import logging

import numpy as np
import pandas as pd

from config.config import get_argparse
from config.pricing_constants import *
from config.db_tables import SMALL_PART_MAX_MIN_PRICE_SQL, FLIGHT_PRICE_BOTTOM_SQL

# 注意：knn_run 的导入已移到函数内部，根据 args.use_v2_predictor 动态选择
from common.get_logger import get_logger
from common.database_oracle import get_data, delete_data, insert_data
import schedule


def small_part_flight_advice_price(args):
    get_logger()
    logging.info(f"【低份额航班定价引擎】{args.version_number} 程序开始！")

    # v2 开关：设置 args.use_v2_predictor=True 启用新版预测器
    if getattr(args, 'use_v2_predictor', False):
        from blocks.SmallPartFlight.SmallPartFlightCapCtrl_v2 import knn_run
    else:
        from blocks.SmallPartFlight.SmallPartFlightCapCtrl import knn_run

    # 1 利用KNN算法获取目标航班的库存水平
    flt_cap_ctrl = knn_run(args)

    # 2 判断当前存量预售水平是否正常，链接航班选择概率
    logging.info(f"【SmallPartFlightAdvicePrice】{args.version_number} 程序开始！")
    # 获取旅客选择概率
    tmp_sql = f"""
        SELECT CATCH_DATE,FLT_DATE,CATCH_DIF AS EX_DIF,CARRIER AS AIR_CODE,FLT_NO,DEP||ARR AS FLT_SEGMENT,--TO_NUMBER(SUBSTR(CATCH_TIME,1,2)) AS TIME_PT,
           SRS_ZL,SRS_ZL_IND,PSG_CHO_PROB,UP_DATE
        FROM 
        (
            SELECT S.*,
            SUM(GREATEST(S.SRS_ZL,0)) OVER (PARTITION BY S.FLT_DATE,S.CATCH_DATE,S.CATCH_TIME,REPLACE(REPLACE(DEP,'PEK','PKX'),'CTU','TFU')||REPLACE(REPLACE(ARR,'PEK','PKX'),'CTU','TFU')) AS SRS_ZL_IND,
            NVL(GREATEST(S.SRS_ZL,0)/NULLIF(SUM(GREATEST(S.SRS_ZL,0)) OVER (PARTITION BY S.FLT_DATE,S.CATCH_DATE,S.CATCH_TIME,REPLACE(REPLACE(DEP,'PEK','PKX'),'CTU','TFU')||REPLACE(REPLACE(ARR,'PEK','PKX'),'CTU','TFU')),0),0) PSG_CHO_PROB
            FROM 
            (
                SELECT S.*,
                NVL(S.SRS - LAG(S.SRS) OVER(PARTITION BY FLT_DATE,S.DEP||S.ARR,S.CARRIER,FLT_NO ORDER BY S.CATCH_DIF DESC,S.CATCH_TIME),0) AS SRS_ZL,
                RANK()OVER(PARTITION BY S.FLT_DATE,S.CARRIER,S.FLT_NO,S.DEP,S.ARR ORDER BY S.CATCH_DATE DESC,S.CATCH_TIME DESC) RANK1
                FROM KD_FUTURE_SJC S 
                WHERE S.FLT_DATE BETWEEN TRUNC(SYSDATE) AND TRUNC(SYSDATE)+30
                AND EXISTS (SELECT FLT_SEGMENT FROM DP_FLT_LIST T WHERE T.FLT_TYPE= 'SMALL_PART' AND REPLACE(REPLACE(DEP,'PEK','PKX'),'CTU','TFU')||REPLACE(REPLACE(ARR,'PEK','PKX'),'CTU','TFU') = T.FLT_SEGMENT)
                AND ((S.CATCH_DIF BETWEEN 1 AND 30) OR (S.CATCH_DIF=0 AND TO_NUMBER(SUBSTR(S.CATCH_TIME,1,2)) + 1 <= TO_NUMBER(SUBSTR(S.DEP_TIME,1,2))))
            ) S
        )
        WHERE RANK1=1 AND CARRIER IN ('MF','NS','RY')
    """
    flt_psg_choice = get_data(tmp_sql)
    flt_price = pd.merge(flt_cap_ctrl, flt_psg_choice, on=['CATCH_DATE', 'FLT_DATE', 'EX_DIF', 'AIR_CODE', 'FLT_NO', 'FLT_SEGMENT'], how='left')[['CATCH_DATE', 'FLT_DATE', 'EX_DIF', 'DOW', 'TIME_PT', 'AIR_CODE', 'FLT_NO',
           'FLT_SEGMENT', 'FLT_ROUTE', 'DEP_HOUR', 'DEP_MINUTE', 'CAP', 'DISCAP', 'PRICE',
           'BKD_LEFT', 'BKD', 'GRS', 'BKD_SK', 'PJPJ', 'HXJG_FLAG',
           'CAP_FINAL', 'CAP_IND_FINAL', 'KZL_ZL_MF',
           'KZL_ZL_IND', 'ARTIFICIAL_CAP_LEFT', 'CAP_LEFT_NEW', 'SRS_ZL', 'SRS_ZL_IND', 'PSG_CHO_PROB']]

    # 当存量预售水平高于库存水平，外放价格建议维持或建议上涨
    flt_price['SRS_ZL_EST'] = np.maximum(flt_price['PSG_CHO_PROB'] * flt_price['KZL_ZL_IND'] * flt_price['CAP_IND_FINAL'] + flt_price['ARTIFICIAL_CAP_LEFT'], 0)
    # 当始发剩余座位数<=0的时候，建议价格维持不变，即系数为1（剩余座位对应客座率）
    flt_price['SALES_SLOPE'] = np.where(flt_price['BKD_LEFT'] > 0, flt_price['SRS_ZL_EST'] / flt_price['BKD_LEFT'], 1)
    # 销售速度偏慢适合的判断逻辑
    flt_price['SALES_SLOPE_SLOW'] = np.where(flt_price['CAP'] > 0,
                                        np.minimum(
                                            np.maximum(flt_price['PSG_CHO_PROB'] * flt_price['KZL_ZL_IND'] * flt_price['CAP_IND_FINAL'] + flt_price['ARTIFICIAL_CAP_LEFT'], flt_price['CAP_LEFT_NEW']) + flt_price['BKD'], flt_price['CAP']) / flt_price['CAP'],
                                        1
                                        )
    # flt_price['SALES_SLOPE_SLOW'] = np.where(flt_price['BKD_LEFT'] > 0,
    #                                     np.maximum(np.maximum(flt_price['PSG_CHO_PROB'] * flt_price['KZL_ZL_IND'] * flt_price['CAP_IND_FINAL'] + flt_price['ARTIFICIAL_CAP_LEFT'], flt_price['CAP_LEFT_NEW']), 0) / flt_price['BKD_LEFT'],
    #                                     1
    #                                     )
    # 锚定特殊底价航班
    flt_price = pd.merge(flt_price, get_data(FLIGHT_PRICE_BOTTOM_SQL),
                                       on=['FLT_SEGMENT', 'FLT_NO'], how='left')
    # 筛选出在日期范围内的记录
    # 先筛选出没有设置兜底价的数据
    isnull_tmp_price_advice = flt_price[flt_price.isnull().any(axis=1)]
    # 剔除不在有效范围内的数据
    flt_price = flt_price[
        (flt_price['FLT_DATE'] >= flt_price['BEGIN_DATE']) & (
                flt_price['FLT_DATE'] <= flt_price['END_DATE'])]
    flt_price = flt_price.append(isnull_tmp_price_advice)
    # 如果没有单独设置的底价，那小份额航班设置绝对底价
    flt_price['PRICE_BOTTOM'].fillna(SMALL_FLT_PRICE_FLOOR_ABSOLUTE, inplace=True)

    # 凌晨时间段保持定价稳定，建议价格与外放价格保持一致ff
    # flt_price['CREATE_HOUR'] = args.file_create_hour
    # flt_price['AI_ADVICE_PRICE'] = np.where((flt_price['CREATE_HOUR'] < 8) & (flt_price['SRS_ZL_IND'] <= 5),
    #                                         flt_price['PJPJ'],
    #                                         np.where(flt_price['CAP_LEFT_NEW'] >= flt_price['BKD_LEFT'],
    #                                                 # 过去一段时间的销售速度（斜率）较快，建议提价（+1折）
    #                                                 np.where((flt_price['SALES_SLOPE'] >= 1) & (flt_price['SRS_ZL'] > 0),
    #                                                          np.where(((flt_price['EX_DIF'] == 2) & (flt_price['TIME_PT'] >= 18)) | ((flt_price['EX_DIF'] == 1) & (flt_price['TIME_PT'] <= 7)), # 单独设置D2夜间
    #                                                                   np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PRICE']),
    #                                                                   np.minimum(np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PJPJ'] + flt_price['PRICE']*0.05), flt_price['PRICE'])),
    #                                                          flt_price['PJPJ']
    #                                                          ),
    #                                                 # np.where((flt_price['SALES_SLOPE'] >= 1) & (flt_price['SRS_ZL'] > 0),
    #                                                 #          np.minimum(np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PJPJ'] + flt_price['PRICE']*0.05), flt_price['PRICE']),
    #                                                 #          flt_price['PJPJ']
    #                                                 #          ),
    #                                                 # 过去一段时间的销售速度（斜率）较慢，建议降价（外放0.9折）
    #                                                 np.where(flt_price['SALES_SLOPE'] < 1,
    #                                                          np.maximum(
    #                                                              np.maximum(flt_price['PJPJ'] * flt_price['SALES_SLOPE_SLOW'], flt_price['PJPJ']*0.9),
    #                                                              flt_price['PRICE_BOTTOM']),
    #                                                          flt_price['PJPJ']
    #                                                          ))
    #                                         )
    flt_price['AI_ADVICE_PRICE'] = np.where(flt_price['CAP_LEFT_NEW'] >= flt_price['BKD_LEFT'],
                                                    # 过去一段时间的销售速度（斜率）较快，建议提价（+1折）
                                                    np.where((flt_price['SALES_SLOPE'] >= 1) & (flt_price['SRS_ZL'] > 0),
                                                             np.where(flt_price['ARTIFICIAL_CAP_LEFT'] > 0,  # 单独设置有库存的航班不限制提价幅度
                                                                      np.minimum(np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PJPJ'] + flt_price['PRICE'] * SMALL_FLT_PRICE_UP_ARTIFICIAL_PCT), flt_price['PRICE']),
                                                                      np.minimum(np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PJPJ'] + flt_price['PRICE'] * SMALL_FLT_PRICE_UP_NORMAL_PCT), flt_price['PRICE'])),
                                                             flt_price['PJPJ']
                                                             ),
                                                    # 过去一段时间的销售速度（斜率）较慢，建议降价
                                                    np.where(flt_price['SALES_SLOPE'] < 1,
                                                             np.maximum(
                                                                 np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE_SLOW'], flt_price['PJPJ'] - flt_price['PRICE'] * SMALL_FLT_PRICE_DOWN_PCT),
                                                                 flt_price['PRICE_BOTTOM']),
                                                             flt_price['PJPJ']
                                                             ))

    # 临时存储不限制涨价的价格
    # flt_price['TMP_ADVICE_PRICE'] = np.where((flt_price['CREATE_HOUR'] < 8) & (flt_price['SRS_ZL_IND'] <= 5),
    #                                         flt_price['PJPJ'],
    #                                         np.where(flt_price['CAP_LEFT_NEW'] >= flt_price['BKD_LEFT'],
    #                                                 # 过去一段时间的销售速度（斜率）较快，建议提价（+1折）
    #                                                 np.where((flt_price['SALES_SLOPE'] >= 1) & (flt_price['SRS_ZL'] > 0),
    #                                                          np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PRICE']),
    #                                                          flt_price['PJPJ']
    #                                                          ),
    #                                                 # 过去一段时间的销售速度（斜率）较慢，建议降价（外放0.9折）
    #                                                 np.where(flt_price['SALES_SLOPE'] < 1,
    #                                                          np.maximum(
    #                                                              np.maximum(flt_price['PJPJ'] * flt_price['SALES_SLOPE_SLOW'], flt_price['PJPJ']*0.9),
    #                                                              flt_price['PRICE_BOTTOM']),
    #                                                          flt_price['PJPJ']
    #                                                          ))
    #                                         )
    flt_price['TMP_ADVICE_PRICE'] = np.where(flt_price['CAP_LEFT_NEW'] >= flt_price['BKD_LEFT'],
                                                    # 过去一段时间的销售速度（斜率）较快，建议提价（+1折）
                                                    np.where((flt_price['SALES_SLOPE'] >= 1) & (flt_price['SRS_ZL'] > 0),
                                                             np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PRICE']),
                                                             flt_price['PJPJ']
                                                             ),
                                                    # 过去一段时间的销售速度（斜率）较慢，建议降价
                                                    np.where(flt_price['SALES_SLOPE'] < 1,
                                                             np.maximum(
                                                                 np.maximum(flt_price['PJPJ'] * flt_price['SALES_SLOPE_SLOW'], flt_price['PJPJ'] - flt_price['PRICE'] * SMALL_FLT_PRICE_DOWN_PCT),
                                                                 flt_price['PRICE_BOTTOM']),
                                                             # 区分D1前后的提价逻辑
                                                             np.where(flt_price['EX_DIF'] <= 1,
                                                                      flt_price['PJPJ'],
                                                                      np.minimum(flt_price['PJPJ'] * flt_price['SALES_SLOPE'], flt_price['PJPJ'] + flt_price['PRICE'] * SMALL_FLT_PRICE_UP_NORMAL_PCT))
                                                             ))

    flt_price = pd.merge(flt_price, get_data(SMALL_PART_MAX_MIN_PRICE_SQL),
                         on=['FLT_DATE', 'FLT_NO', 'AIR_CODE', 'FLT_SEGMENT'],
                         how='left')
    np.where(flt_price['AI_ADVICE_PRICE'] < flt_price['MIN_PRICE'],
             flt_price['MIN_PRICE'],
             flt_price['AI_ADVICE_PRICE'])
    np.where(flt_price['AI_ADVICE_PRICE'] > flt_price['MAX_PRICE'],
             flt_price['MAX_PRICE'],
             flt_price['AI_ADVICE_PRICE'])

    flt_price['AI_ADVICE_PRICE'] = round_to_10(flt_price['AI_ADVICE_PRICE'])
    flt_price['TMP_ADVICE_PRICE'] = round_to_10(flt_price['TMP_ADVICE_PRICE'])
    flt_price['CREATE_TIME'] = args.create_time
    flt_price = flt_price[['CATCH_DATE', 'FLT_DATE', 'EX_DIF', 'DOW', 'TIME_PT', 'AIR_CODE',
       'FLT_NO', 'FLT_SEGMENT', 'FLT_ROUTE', 'DEP_HOUR', 'DEP_MINUTE', 'CAP',
       'DISCAP', 'PRICE', 'BKD_LEFT', 'BKD', 'GRS', 'BKD_SK', 'PJPJ',
       'HXJG_FLAG', 'CAP_FINAL', 'CAP_IND_FINAL', 'KZL_ZL_MF',
       'KZL_ZL_IND', 'ARTIFICIAL_CAP_LEFT', 'CAP_LEFT_NEW', 'SRS_ZL',
       'PSG_CHO_PROB', 'SRS_ZL_EST', 'SALES_SLOPE', 'AI_ADVICE_PRICE', 'CREATE_TIME', 'TMP_ADVICE_PRICE']]
    # 保存定价数据
    delete_data("""DELETE FROM SMALL_PART_ADVICE_PRICE_OUTPUT""")
    # 插入实时数据
    insert_data("SMALL_PART_ADVICE_PRICE_OUTPUT", flt_price)
    # 调整经停长短定价
    tmp_sql = """
    SELECT A.CATCH_DATE,A.FLT_DATE,A.EX_DIF,A.DOW,A.TIME_PT,A.AIR_CODE,A.FLT_NO,A.FLT_SEGMENT,A.FLT_ROUTE,
    A.DEP_HOUR,A.DEP_MINUTE,A.CAP,A.DISCAP,A.PRICE,A.BKD_LEFT,A.BKD,A.GRS,A.BKD_SK,A.PJPJ,A.HXJG_FLAG,
    A.CAP_FINAL,A.CAP_IND_FINAL,A.KZL_ZL_MF,A.KZL_ZL_IND,A.ARTIFICIAL_CAP_LEFT,A.CAP_LEFT,A.SRS_ZL,
    A.PSG_CHO_PROB,A.SRS_ZL_EST,A.SALES_SLOPE,
    CASE WHEN LENGTH(A.FLT_ROUTE)=9 AND A.HXJG_FLAG=0 THEN LEAST(GREATEST(A.AI_ADVICE_PRICE,B.TMP_ADVICE_PRICE_SUM),A.PRICE) ELSE A.AI_ADVICE_PRICE END AS AI_ADVICE_PRICE,
    A.CREATE_TIME,A.TMP_ADVICE_PRICE
    FROM SMALL_PART_ADVICE_PRICE_OUTPUT A
    LEFT JOIN
    (
    --计算经停短段建议价格之和
    SELECT FLT_ROUTE,FLT_NO,EX_DIF,SUM(TMP_ADVICE_PRICE) AS TMP_ADVICE_PRICE_SUM
    FROM SMALL_PART_ADVICE_PRICE_OUTPUT A
    WHERE LENGTH(FLT_ROUTE)=9
    AND HXJG_FLAG=1
    GROUP BY FLT_ROUTE,FLT_NO,EX_DIF
    ORDER BY FLT_ROUTE,FLT_NO,EX_DIF
    )B
    ON A.FLT_ROUTE=B.FLT_ROUTE AND A.FLT_NO=B.FLT_NO AND A.EX_DIF=B.EX_DIF
    """
    flt_price = get_data(tmp_sql)

    delete_data("""DELETE FROM SMALL_PART_ADVICE_PRICE_OUTPUT""")
    insert_data("SMALL_PART_ADVICE_PRICE_OUTPUT", flt_price)

    # 插入累积观察数据
    insert_data("SMALL_PART_ADVICE_PRICE_COPY", flt_price)
    return flt_price

def run():
    args = get_argparse()
    small_part_flight_advice_price(args)

if __name__ == '__main__':
    run()
    # schedule.every().day.at("00:20").do(run)
    # schedule.every().day.at("01:20").do(run)
    # schedule.every().day.at("02:20").do(run)
    # schedule.every().day.at("03:20").do(run)
    # schedule.every().day.at("04:20").do(run)
    # schedule.every().day.at("05:20").do(run)
    # # schedule.every().day.at("06:20").do(run)
    # schedule.every().day.at("07:20").do(run)
    # schedule.every().day.at("08:20").do(run)
    # schedule.every().day.at("09:20").do(run)
    # schedule.every().day.at("10:20").do(run)
    # schedule.every().day.at("11:20").do(run)
    # schedule.every().day.at("12:20").do(run)
    # schedule.every().day.at("13:20").do(run)
    # schedule.every().day.at("14:20").do(run)
    # schedule.every().day.at("15:20").do(run)
    # schedule.every().day.at("16:20").do(run)
    # schedule.every().day.at("17:20").do(run)
    # schedule.every().day.at("18:20").do(run)
    # schedule.every().day.at("19:20").do(run)
    # schedule.every().day.at("20:20").do(run)
    # schedule.every().day.at("21:20").do(run)
    # schedule.every().day.at("22:20").do(run)
    # schedule.every().day.at("23:20").do(run)
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)