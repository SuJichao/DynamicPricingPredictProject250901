"""
【程序目的】
对独飞航班的建议价格进行调整。
"""
import logging
import math
import uuid

import numpy as np
import pandas as pd

from config.config import get_argparse
from config.pricing_constants import *
from data_provider.data_acquisition import get_data
def true_price_up_down(config, flt_type, data):
    logging.info(f"【SoloFltPriceUpDown】{config.version_number} 程序开始！")
    tmp_solo_advice_data = data
    # # 获取实时外放价格
    # rb_ota_data = get_data("oracle", data_sql=config.rb_ota_data)
    # rb_ota_data = rb_ota_data[['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT', 'AIR_CODE',
    #                            'FLT_NO', 'FULL_PRICE', 'AVG_FARE_SK']]
    # sales_ratio = get_data("oracle", data_sql=config.sales_ratio)
    # # 防止外放价格已经到商务舱，触发价格上限缓升
    # rb_ota_data['AVG_FARE_SK'] = np.where(rb_ota_data['AVG_FARE_SK'] > rb_ota_data['FULL_PRICE'],
    #                                       rb_ota_data['FULL_PRICE'], rb_ota_data['AVG_FARE_SK'])
    # rb_ota_data.reset_index(drop=True, inplace=True)
    # advice_price_data.reset_index(drop=True, inplace=True)
    # # 将最新建议价格与上一时点的采集价格进行左连接
    # tmp_price_advice_result = pd.merge(advice_price_data, rb_ota_data,
    #                                    left_on=['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT', 'CARRIER',
    #                                             'FLT_NO', 'PRICE'],
    #                                    right_on=['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT',
    #                                              'AIR_CODE', 'FLT_NO', 'FULL_PRICE'],
    #                                    how='left')

    # 独飞航线处理模式
    if flt_type == 'SOLO_PART':
        tmp_solo_advice_data.rename(
            columns={'AI_ADVICE_PRICE': 'AVG_FARE_SK_x',
                     'AVG_FARE_SK': 'AVG_FARE_SK_y'
                     }, inplace=True)
        tmp_solo_advice_data['AVG_FARE_SK'] = tmp_solo_advice_data['AVG_FARE_SK_x']
        # solo_previous_price = get_data("oracle", data_sql=f"SELECT EX_DIF AS EX_DIF_OLD,TIME_PT_OLD,FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,DEP_TIME,BKD_OLD,PRICE_OTA_OLD FROM {config.solo_previous_price}")
        # bkd_sluggish_record = get_data("oracle", data_sql=f"SELECT * FROM BKD_SLUGGISH_RECORD")
        # # 筛选出有需要的字段，FULL_FARE_VIRTUAL代表建议价格，AVG_FARE_SK代表上一采集点的外放价格
        # tmp_price_advice_result.rename(
        #     columns={'AI_ADVICE_PRICE': 'AVG_FARE_SK_x',
        #              'AVG_FARE_SK': 'AVG_FARE_SK_y'
        #              }, inplace=True)
        # # 【情况1】简单的价格缓升、缓降操作
        # # 【传统缓降机制】
        # tmp_price_advice_result['AVG_FARE_SK'] = tmp_price_advice_result['AVG_FARE_SK_x']
        # tmp_price_advice_result['AVG_FARE_SK'] = np.where((tmp_price_advice_result['AVG_FARE_SK_x'] + tmp_price_advice_result['AVG_FARE_SK_y'] * 0.1 < tmp_price_advice_result['AVG_FARE_SK_y']) & (tmp_price_advice_result['EX_DIF']<=7),
        #                                                   # tmp_price_advice_result['AVG_FARE_SK_x'],
        #                                                   tmp_price_advice_result['AVG_FARE_SK_y'] - np.maximum((tmp_price_advice_result['AVG_FARE_SK_y'] - tmp_price_advice_result['AVG_FARE_SK_x']) * (1/3), tmp_price_advice_result['AVG_FARE_SK_y']*0.1),
        #                                                   tmp_price_advice_result['AVG_FARE_SK_x']
        #                                                   )
        # 【情况1】对订座突增航班进行价格上涨（目前仅限D0-30）
        # solo_previous_price主要用于存储上一采集时点的外放价格和订座人数
        # tmp_solo_advice_data = pd.merge(tmp_price_advice_result, solo_previous_price, how='left',
        #                                 on=['FLT_DATE', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT'])
        # # 防止多个连续时点未采集到数据，导致人数增量计算异常
        # tmp_solo_advice_data['BKD_INC'] = (tmp_solo_advice_data['BKD'] - tmp_solo_advice_data['BKD_OLD'])/(tmp_solo_advice_data['TIME_PT']+(tmp_solo_advice_data['EX_DIF_OLD']-tmp_solo_advice_data['EX_DIF'])*24-tmp_solo_advice_data['TIME_PT_OLD'])
        # # tmp_solo_advice_data['PRICE_INCREASE'] = np.where(tmp_solo_advice_data['EX_DIF'] < 7, tmp_solo_advice_data['BKD_INC'].fillna(0)/5, tmp_solo_advice_data['BKD_INC'].fillna(0)/10)
        # # D0-1
        # tmp_solo_advice_data['PRICE_INCREASE'] = np.where(tmp_solo_advice_data['EX_DIF'] <= 1,
        #                                                   tmp_solo_advice_data['BKD_INC'].fillna(0) // 5,
        #                                                   np.where((tmp_solo_advice_data['EX_DIF'] >= 2) & (tmp_solo_advice_data['EX_DIF'] <= 7),
        #                                                            tmp_solo_advice_data['BKD_INC'].fillna(0) // 3,
        #                                                            np.where(tmp_solo_advice_data['EX_DIF'] >= 8,
        #                                                                     tmp_solo_advice_data['BKD_INC'].fillna(0) // 0.5,
        #                                                                     0)
        #                                                            )
        #                                                   )
        # tmp_solo_advice_data['PRICE_INCREASE'] = np.minimum(tmp_solo_advice_data['PRICE_INCREASE'].apply(math.floor), 2) # 最多涨价2折
        # # 如果当前时点的人数订座增量出现突增，以5个人为一档，增5-9人，加一折;增10-14人，加两折;依此下去（预测客座率大于80%时才触发）
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where((tmp_solo_advice_data['PRICE_INCREASE'] >= 1) & (tmp_solo_advice_data['BKD_PLF_EST'] > 0.9),
        #                                                # 防止突增后的价格低于模型建议价格（20250417改为直接看突增价格）
        #                                                tmp_solo_advice_data['PRICE_OTA'] + round((tmp_solo_advice_data['PRICE_INCREASE'] * tmp_solo_advice_data['PRICE'] * 0.1) / 10) * 10,
        #                                                tmp_solo_advice_data['AVG_FARE_SK'])
        #                                                # np.maximum(tmp_solo_advice_data['AVG_FARE_SK_y'] + round((tmp_solo_advice_data['PRICE_INCREASE'] * tmp_solo_advice_data['FULL_PRICE'] * 0.1)/10)*10, tmp_solo_advice_data['AVG_FARE_SK_x']),
        #                                                # tmp_solo_advice_data['AVG_FARE_SK_x'])
        #
        # # 插入当前时点出现订座突增的航班数据
        # bkd_sudden_increase_record = tmp_solo_advice_data[tmp_solo_advice_data['PRICE_INCREASE'] >= 1][['CATCH_DATE', 'EX_DIF', 'FLT_DATE', 'TIME_PT', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT', 'AVG_FARE_SK', 'BKD_INC', 'UP_DATE']]
        # bkd_sudden_increase_record.reset_index(drop=True, inplace=True)
        # bkd_sudden_increase_record['PID'] = uuid.uuid1()
        # for i in range(len(bkd_sudden_increase_record)):
        #     bkd_sudden_increase_record.at[i, 'PID'] = uuid.uuid1()
        # bkd_sudden_increase_record['PID'] = bkd_sudden_increase_record['PID'].astype('str')
        # insert_predict_data(
        #     """INSERT INTO BKD_SUDDEN_INCREASE_RECORD VALUES(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11)""",
        #     bkd_sudden_increase_record)
        #
        # # 删除多次插入的突增数据（旧数据）
        # tmp_sql = """
        #     DELETE FROM BKD_SUDDEN_INCREASE_RECORD S
        #     WHERE PID IN
        #     (
        #     SELECT PID
        #     FROM
        #     (
        #       SELECT A.*,ROW_NUMBER () OVER (PARTITION BY A.FLT_DATE,A.FLT_NO,A.FLT_SEGMENT ORDER BY A.CREATE_TIME DESC) RN
        #       FROM BKD_SUDDEN_INCREASE_RECORD A
        #     )WHERE RN!=1
        #     )
        # """
        # delete_predict_data(tmp_sql)
        #
        # # 当独飞订座突增表中有数据时，维持定价不变，否则回调价格
        # bkd_sudden_increase_record = get_predict_data("SELECT CATCH_DATE,EX_DIF,FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,ADVICE_PRICE,CREATE_TIME,PID FROM BKD_SUDDEN_INCREASE_RECORD")
        # tmp_solo_advice_data = pd.merge(tmp_solo_advice_data, bkd_sudden_increase_record, how='left', on=['CATCH_DATE', 'EX_DIF', 'FLT_DATE', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT'])
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where(tmp_solo_advice_data['ADVICE_PRICE'] > 0,
        #                                                np.maximum(tmp_solo_advice_data['ADVICE_PRICE'], tmp_solo_advice_data['AVG_FARE_SK']),
        #                                                tmp_solo_advice_data['AVG_FARE_SK'])
        #
        # 【情况3】当预售人数触发库存后，进行价格阶梯上涨（超过1-9人加一折，超过10-19人加2折，否则全价，在建议价格基础上增加）
        tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] = round(tmp_solo_advice_data['SRS_ZL_DETR_LEFT'], 0)
        # 1到9人提价1折
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where((tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] <= tmp_solo_advice_data['DISCAP'] + 9) & (tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] > tmp_solo_advice_data['DISCAP']),
        #                                                   np.where(tmp_solo_advice_data['ADVICE_PRICE'] > 0,
        #                                                            tmp_solo_advice_data['AVG_FARE_SK'],
        #                                                            np.maximum(tmp_solo_advice_data['AVG_FARE_SK_x'] + round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10,
        #                                                                       tmp_solo_advice_data['PJPJ_MIN'] + round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10)),
        #                                                   tmp_solo_advice_data['AVG_FARE_SK']
        #                                                   )
        # 10-19人提价2折
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where((tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] <= tmp_solo_advice_data['DISCAP'] + 19) & (tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] >= tmp_solo_advice_data['DISCAP']+10),
        #                                                   np.where(tmp_solo_advice_data['ADVICE_PRICE'] > 0,
        #                                                            tmp_solo_advice_data['AVG_FARE_SK'],
        #                                                            np.maximum(tmp_solo_advice_data['AVG_FARE_SK_x'] + 2 * round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10,
        #                                                                       tmp_solo_advice_data['PJPJ_MIN'] + 2 * round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10)),
        #                                                   tmp_solo_advice_data['AVG_FARE_SK']
        #                                                   )
        # 20-29人提价3折
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where((tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] <= tmp_solo_advice_data['DISCAP'] + 29) & (tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] >= tmp_solo_advice_data['DISCAP']+20),
        #                                                   np.where(tmp_solo_advice_data['ADVICE_PRICE'] > 0,
        #                                                            tmp_solo_advice_data['AVG_FARE_SK'],
        #                                                            np.maximum(tmp_solo_advice_data['AVG_FARE_SK_x'] + 3 * round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10,
        #                                                                       tmp_solo_advice_data['PJPJ_MIN'] + 3 * round((tmp_solo_advice_data['PRICE'] * 0.1)/10)*10)),
        #                                                   tmp_solo_advice_data['AVG_FARE_SK']
        #                                                   )
        # 超过30人价格拉升至全价
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where(tmp_solo_advice_data['TRUE_MF_CAP_CTRL'] + tmp_solo_advice_data['BKD'] >= tmp_solo_advice_data['DISCAP']+30,
        #                                                   tmp_solo_advice_data['PRICE'],
        #                                                   tmp_solo_advice_data['AVG_FARE_SK']
        #                                                   )


        # 【情况4】2小时内订座无明显变化，价格下调0.5折
        # tmp_solo_advice_data = pd.merge(tmp_solo_advice_data, bkd_sluggish_record, on=['CATCH_DATE', 'EX_DIF', 'FLT_DATE', 'FLT_NO', 'FLT_SEGMENT'], how='left')
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where(tmp_solo_advice_data['BKD_LONG_OLD'] > 0,
        #                                                         # 判定销售停滞，且没有触发库存时执行（在建议价格上减1折）
        #                                                         tmp_solo_advice_data['AVG_FARE_SK'] - round(tmp_solo_advice_data['PRICE'] * 0.05 / 10) * 10,
        #                                                         tmp_solo_advice_data['AVG_FARE_SK'])

        # 【情况5】当预售客座率大于97%时，收到全价
        # tmp_solo_advice_data['AVG_FARE_SK'] = np.where(tmp_solo_advice_data['BKD']/tmp_solo_advice_data['DISCAP'] >= 0.97, tmp_solo_advice_data['PRICE'], tmp_solo_advice_data['AVG_FARE_SK'])

        # 【情况6】当NS航班（不含PKX进出港）建议价格在29-30折之间，提价到31折（31折开始有行李）
        tmp_price_advice_result = tmp_solo_advice_data
        tmp_price_advice_result['AVG_FARE_SK'] = np.where(
            (tmp_price_advice_result['AIR_CODE'] == 'NS') &
            ~(tmp_price_advice_result['ROUTE'].str.contains('PKX')) &
            (tmp_price_advice_result['AVG_FARE_SK'] / tmp_price_advice_result['PRICE'] >= SOLO_NS_BAGGAGE_LOWER_DISCOUNT) &
            (tmp_price_advice_result['AVG_FARE_SK'] / tmp_price_advice_result['PRICE'] < SOLO_NS_BAGGAGE_TARGET_DISCOUNT),
            tmp_price_advice_result['FULL_PRICE'] * SOLO_NS_BAGGAGE_TARGET_DISCOUNT,
            tmp_price_advice_result['AVG_FARE_SK']
        )

        # 【情况7】20241223生效：针对MF航线，0-2天，客座率低于70的，价格不低于400/2折(取高)，客座率高于70的，价格不低于500/3折(取高)。
        # tmp_price_advice_result['AVG_FARE_SK'] = np.where(
        #     (tmp_price_advice_result['AIR_CODE'] == 'MF') &
        #     (tmp_price_advice_result['BKD']/tmp_price_advice_result['DISCAP'] > 0.7) &
        #     (tmp_price_advice_result['EX_DIF'] <= 2),
        #     np.maximum(np.maximum(tmp_price_advice_result['PRICE'] * 0.3, 500), tmp_price_advice_result['AVG_FARE_SK']),
        #     tmp_price_advice_result['AVG_FARE_SK']
        # )
        # tmp_price_advice_result['AVG_FARE_SK'] = np.where(
        #     (tmp_price_advice_result['AIR_CODE'] == 'MF') &
        #     (tmp_price_advice_result['BKD']/tmp_price_advice_result['DISCAP'] <= 0.7) &
        #     (tmp_price_advice_result['EX_DIF'] <= 2),
        #     np.maximum(np.maximum(tmp_price_advice_result['PRICE'] * 0.2, 400), tmp_price_advice_result['AVG_FARE_SK']),
        #     tmp_price_advice_result['AVG_FARE_SK']
        # )



        # 【情况8】航线兜底底价设置
        # 获取提前设置好的航线兜底价格数据
        tmp_price_advice_result = pd.merge(tmp_price_advice_result,
                                           get_data("oracle", data_sql=config.flight_price_bottom),
                                           on=['FLT_SEGMENT', 'FLT_NO'], how='left')
        # 筛选出在日期范围内的记录
        # 先筛选出没有设置兜底价的数据
        isnull_tmp_price_advice = tmp_price_advice_result[(tmp_price_advice_result['PRICE_BOTTOM'].isna())]
        # 剔除不在有效范围内的数据
        tmp_price_advice_result = tmp_price_advice_result[
            (tmp_price_advice_result['FLT_DATE'] >= tmp_price_advice_result['BEGIN_DATE']) & (
                        tmp_price_advice_result['FLT_DATE'] <= tmp_price_advice_result['END_DATE'])]
        tmp_price_advice_result = tmp_price_advice_result.append(isnull_tmp_price_advice)
        # 如果没有单独设置的底价，那独飞航班按折扣计算底价（不低于绝对底价）
        tmp_price_advice_result['PRICE_BOTTOM'].fillna(
            np.maximum(round_to_10(tmp_price_advice_result['PRICE'] * config.solo_flight_bottom_discount), SOLO_FLT_PRICE_FLOOR_ABSOLUTE),
            inplace=True)
        # 最终建议价格不能低于底价
        tmp_price_advice_result['AVG_FARE_SK'] = np.maximum(tmp_price_advice_result['AVG_FARE_SK'], tmp_price_advice_result['PRICE_BOTTOM'])
        # 对建议价格进行数值类型强制转换
        tmp_price_advice_result['AVG_FARE_SK'] = tmp_price_advice_result['AVG_FARE_SK'].astype(float)
        # 价格取整（四舍五入到10的倍数）
        tmp_price_advice_result['AVG_FARE_SK'] = np.minimum(round_to_10(tmp_price_advice_result['AVG_FARE_SK']), tmp_price_advice_result['PRICE'])

        # 【校正数据格式】
        # 独飞航班预测结果表
        solo_flt_advice_price = tmp_price_advice_result[['CATCH_DATE', 'CATCH_TIME', 'TIME_PT', 'EX_DIF', 'DOW', 'FLT_DATE',
                     'CARRIER', 'FLT_NO', 'FLT_SEGMENT', 'ROUTE', 'HXJG_FLAG',
                     'AVG_FARE_SK', 'DEP_HOUR', 'DEP_MINUTE', 'CAP', 'DISCAP', 'BKD', 'PRICE_OTA', 'PRICE', 'PJPJ_MIN','UP_DATE', 'BKD_INCOME_LEFT', 'SRS_ZL_LEFT']]
        solo_flt_advice_price = solo_flt_advice_price.rename(columns={'CARRIER': 'AIR_CODE',
                                                                      'HXJG_FLAG': 'IS_STOPOVER_FLT',
                                                                      'PRICE': 'FULL_PRICE',
                                                                      'BKD_INCOME_LEFT': 'EXPECTED_RETURN',
                                                                      'SRS_ZL_LEFT': 'BKD_ISSUED_NUM_INC',
                                                                      'ROUTE': 'FLT_ROUTE'
                                                                      })
        solo_flt_advice_price.reset_index(drop=True, inplace=True)
        solo_flt_advice_price['WEB_ID'] = 0
        solo_flt_advice_price['AVG_FARE_SK_IND'] = 0
        solo_flt_advice_price['AVG_FARE_DELTA'] = 0
        solo_flt_advice_price['PSG_CHO_PROB'] = 0
        solo_flt_advice_price['PROB_PRIOR'] = 0
        solo_flt_advice_price['PSG_CHO_PROB_DELTA'] = 0
        solo_flt_advice_price['MAX_DEP_HOUR'] = 0
        solo_flt_advice_price['OBJECT_FLT'] = 'MF8888'
        solo_flt_advice_price['IND_BKD_ISSUED_NUM_INC'] = 0
        solo_flt_advice_price['CREATE_TIME'] = config.create_time
        solo_flt_advice_price_result = solo_flt_advice_price[['CATCH_DATE','EX_DIF','TIME_PT','FLT_DATE','AIR_CODE','FLT_NO','FLT_SEGMENT','FLT_ROUTE',
                                        'IS_STOPOVER_FLT','DEP_HOUR','DEP_MINUTE','WEB_ID','CAP','DISCAP','FULL_PRICE','BKD',
                                        'AVG_FARE_SK','AVG_FARE_SK_IND','AVG_FARE_DELTA','PSG_CHO_PROB','PROB_PRIOR',
                                        'PSG_CHO_PROB_DELTA','MAX_DEP_HOUR','OBJECT_FLT','IND_BKD_ISSUED_NUM_INC',
                                        'BKD_ISSUED_NUM_INC','EXPECTED_RETURN','CREATE_TIME']]

    result_data = {
        'solo_flt_advice_price_result': solo_flt_advice_price_result
    }
    return result_data