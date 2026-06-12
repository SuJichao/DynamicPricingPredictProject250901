"""
【程序目的】
实现对独飞航线剩余销售期内的人数增量预测功能。
"""
import logging
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

from config.runtime_args import get_argparse
from config.pricing_constants import SOLO_KNN_FEATURE_COLS, SOLO_KNN_TARGET_COLS
from common.database_oracle import get_data, delete_data, insert_data
from model.KNeighborsRegressor import SoloFltKnnRegressorFunction
from config.db_queries import (SOLO_ADVICE_PRICE_TRAIN_TABLE, SOLO_ADVICE_PRICE_PREDICT_TABLE, SOLO_ADVICE_PRICE_KNN_LIST)


class SoloFlightNumberIncreaseKNN(object):
    def __init__(self, config):
        self.config = config
        self.train_name = SOLO_ADVICE_PRICE_TRAIN_TABLE
        self.predict_name = SOLO_ADVICE_PRICE_PREDICT_TABLE
        self.X_label_col = list(SOLO_KNN_FEATURE_COLS)
        self.Y_label_col = list(SOLO_KNN_TARGET_COLS)
        self.predict_data = get_data("oracle",
                                     data_sql=f"SELECT * FROM {SOLO_ADVICE_PRICE_PREDICT_TABLE}")
        self.knn_result = pd.DataFrame()
        self.tmp_data = None
        self.solo_flight_list = get_data("oracle",
                                         data_sql=f"SELECT * FROM {SOLO_ADVICE_PRICE_KNN_LIST}")
        logging.info(f"【SoloFlightNumberIncreaseKNN】{self.config.version_number} 程序开始！")
        # self.run()

    def data_deal(self, data):
        # 增加月份、日期和时刻的正余弦函数
        data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
        data['DEP_TIME'] = data['DEP_HOUR'] + data['DEP_MINUTE'] / 60
        data.loc[:, 'YEAR'] = data['FLT_DATE'].dt.year
        # data.loc[:, 'MONTH'] = data['FLT_DATE'].dt.month
        # data.loc[:, 'DAY'] = data['FLT_DATE'].dt.day

        # 月份的正余弦函数
        # data['month_sin'] = np.sin(2 * np.pi * data['MONTH'] / 12.0)
        # data['month_cos'] = np.cos(2 * np.pi * data['MONTH'] / 12.0)
        # 日期的正余弦函数
        # data['day_sin'] = np.sin(2 * np.pi * data['DAY'] / 31.0)
        # data['day_cos'] = np.cos(2 * np.pi * data['DAY'] / 31.0)
        # 离港时间的正余弦函数
        data['deptime_sin'] = np.sin(2 * np.pi * data['DEP_TIME'] / 23.0)
        data['deptime_cos'] = np.cos(2 * np.pi * data['DEP_TIME'] / 23.0)
        # 按日期顺序进行正余弦函数
        data['date_sin'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_sin'] = np.sin(2 * np.pi * data['date_sin'] / 366.0)
        data['date_cos'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_cos'] = np.cos(2 * np.pi * data['date_cos'] / 366.0)
        data['chunjie_sin'] = np.sin(2 * np.pi * data['HOLIDAY_RANGE'] / 30.0)
        return data

    def get_data(self, tmp_list):
        # 分批次读取待预测数据
        # 获取待预测数据
        if tmp_list['HOL_FALG'] == 0:
            predict_list_sql = f'''
                SELECT *
                FROM {self.predict_name} A
                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND TIME_PT={tmp_list['TIME_PT']} AND DOW={tmp_list['DOW']}
            '''
            self.predict_data = get_data(predict_list_sql)
            # 获取训练数据（按照分级解除限制的逻辑进行判断）
            if 7 <= tmp_list['MONTH'] <= 8:  # 当目标航班处于暑运时间内，严格对标其历史样本（在可以找到样本数据的前提下）
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END)) AND DOW={tmp_list['DOW']}
                    AND HOL_FALG='{tmp_list['HOL_FALG']}'
                    AND AIR_CODE IN ('MF','NS','RY') AND TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) BETWEEN 7 AND 8
                '''
            elif tmp_list['MONTH'] < 7 or tmp_list['MONTH'] > 8:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END)) AND DOW={tmp_list['DOW']}
                    AND HOL_FALG='{tmp_list['HOL_FALG']}'
                    AND AIR_CODE IN ('MF','NS','RY') AND (TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) > 8 OR TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) < 7)
                '''
            else:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END)) AND DOW={tmp_list['DOW']}
                    AND HOL_FALG='{tmp_list['HOL_FALG']}'
                    AND AIR_CODE IN ('MF','NS','RY')
                '''
            self.train_data = get_data(train_list_sql)
            # 解除1级限制
            if len(self.train_data) <= 1:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                    AND HOL_FALG='{tmp_list['HOL_FALG']}'
                    AND AIR_CODE IN ('MF','NS','RY')
                '''
                self.train_data = get_data(train_list_sql)
                if len(self.train_data) <= 1:
                    logging.warning(
                        f"普通日样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
            else:
                pass

        elif tmp_list['HOL_FALG'] == 1:
            # 获取待预测数据
            predict_list_sql = f'''
                SELECT *
                FROM {self.predict_name} A
                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND TIME_PT={tmp_list['TIME_PT']}
                    AND DOW={tmp_list['DOW']} AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} 
                    AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} 
                    AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                    AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
            '''
            self.predict_data = get_data(predict_list_sql)
            # 放假天数为1天的情况
            if (tmp_list['HOL_LAST'] == 1) & (tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 0):
                # 获取训练数据（按照分级解除限制的逻辑进行判断）
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                        AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']}
                        AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                '''
                self.train_data = get_data(train_list_sql)
                # 解除1级限制
                if len(self.train_data) <= 1:
                    train_list_sql = f'''
                        SELECT *
                        FROM {self.train_name} A
                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                            AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                            AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                        UNION ALL
                        SELECT *
                        FROM {self.train_name} A
                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                        AND A.HOL_FALG=0
                        AND A.DOW=6
                        AND A.EX_DIF=B.EX_DIF 
                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                        AND A.FLT_SEGMENT=B.FLT_SEGMENT
                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                        )
                    '''
                    self.train_data = get_data(train_list_sql)
                    # 解除2级限制
                    if len(self.train_data) <= 1:
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                                AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF 
                            AND B.EX_DIF>0
                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF 
                            AND B.EX_DIF=0 
                            AND B.TIME_PT>=A.TIME_PT
                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                        '''
                        self.train_data = get_data(train_list_sql)
                        if len(self.train_data) <= 1:
                            logging.warning(
                                f"节假日（1天）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
            # 放假天数为3天的情况
            elif (tmp_list['HOL_LAST'] == 2) & (tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 0):
                # 获取训练数据（按照分级解除限制的逻辑进行判断）
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                        AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']}
                        AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                '''
                self.train_data = get_data(train_list_sql)
                # 解除1级限制
                if len(self.train_data) <= 1:
                    train_list_sql = f'''
                        SELECT *
                        FROM {self.train_name} A
                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                            AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                            AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                        UNION ALL
                        SELECT *
                        FROM {self.train_name} A
                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                        AND A.HOL_FALG=0
                        AND DECODE(B.HOLIDAY_RANGE,-2,4,-1,5,1,6,2,6,3,7,4,1,5,2)=A.DOW 
                        AND A.EX_DIF=B.EX_DIF 
                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                        AND A.FLT_SEGMENT=B.FLT_SEGMENT
                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                        )
                    '''
                    self.train_data = get_data(train_list_sql)
                    # 解除2级限制
                    if len(self.train_data) <= 1:
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                                AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF
                            AND B.EX_DIF>0 
                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF
                            AND B.EX_DIF=0 
                            AND B.TIME_PT>=A.TIME_PT
                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                        '''
                        self.train_data = get_data(train_list_sql)
                        if len(self.train_data) <= 1:
                            logging.warning(
                                f"节假日（3天）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
            # 放假天数为3天以上的情况
            else:
                # 非春节的节假日
                if tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 0:
                    # 4天以上（节前）
                    if tmp_list['HOLIDAY_RANGE'] < 0:
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                        '''
                        self.train_data = get_data(train_list_sql)
                        # 解除1级限制
                        if len(self.train_data) <= 1:
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND DECODE(B.HOLIDAY_RANGE,-2,4,-1,5)=A.DOW
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除2级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF>0
                                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                        AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=B.DOW
                                    AND A.EX_DIF=B.EX_DIF
                                    AND B.EX_DIF=0
                                    AND B.TIME_PT>=A.TIME_PT
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                '''
                                self.train_data = get_data(train_list_sql)
                                if len(self.train_data) <= 1:
                                    logging.warning(
                                        f"节假日（4天以上节前）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                    # 4天以上（节后）
                    elif tmp_list['HOL_LAST'] - tmp_list['HOLIDAY_RANGE'] < 0:
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                        '''
                        self.train_data = get_data(train_list_sql)
                        # 解除1级限制
                        if len(self.train_data) <= 1:
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND DECODE(B.HOLIDAY_RANGE,-2,2,-1,1)=A.DOW
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除2级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF>0
                                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                        AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF=0
                                        AND B.TIME_PT>=A.TIME_PT
                                        AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                '''
                                self.train_data = get_data(train_list_sql)
                                if len(self.train_data) <= 1:
                                    logging.warning(
                                        f"节假日（4天以上节后）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                    # 4天以上（节中）
                    else:
                        # 4-8天，节中第1-2天
                        if (tmp_list['HOL_LAST'] >= 4 and tmp_list['HOL_LAST'] <= 5 and tmp_list[
                            'HOLIDAY_RANGE'] == 1) or (
                                tmp_list['HOL_LAST'] >= 6 and tmp_list['HOL_LAST'] <= 8 and tmp_list[
                            'HOLIDAY_RANGE'] <= 2 and tmp_list['HOLIDAY_RANGE'] >= 1):
                            # 获取训练数据（按照分级解除限制的逻辑进行判断）
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE=1) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE=1)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除1级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE=1) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE=1)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                    )
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=5 
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                            AND HOL_FALG={tmp_list['HOL_FALG']}
                                            AND ((HOL_LAST=4 AND HOLIDAY_RANGE=1) 
                                            OR (HOL_LAST=5 AND HOLIDAY_RANGE=1)
                                            OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                            OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                            OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 1 AND 2)
                                            )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF>0 
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF=0 
                                            AND B.TIME_PT>=A.TIME_PT
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中第1-2天）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                        # 4-8天，节中最后第1-2天
                        elif (tmp_list['HOL_LAST'] >= 4 and tmp_list['HOL_LAST'] <= 5 and tmp_list['HOLIDAY_RANGE'] >=
                              tmp_list['HOL_LAST'] - 1 and tmp_list['HOLIDAY_RANGE'] <= tmp_list['HOL_LAST']) or (
                                tmp_list['HOL_LAST'] >= 6 and tmp_list['HOL_LAST'] <= 8 and tmp_list['HOLIDAY_RANGE'] >=
                                tmp_list['HOL_LAST'] - 1 and tmp_list['HOLIDAY_RANGE'] <= tmp_list['HOL_LAST']):
                            # 获取训练数据（按照分级解除限制的逻辑进行判断）
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE=4) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE=5)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 5 AND 6)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 6 AND 7)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 7 AND 8)
                                    )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除1级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE=4) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE=5)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 5 AND 6)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 6 AND 7)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 7 AND 8)
                                    )
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=7
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                            AND HOL_FALG={tmp_list['HOL_FALG']}
                                            AND ((HOL_LAST=4 AND HOLIDAY_RANGE=4) 
                                            OR (HOL_LAST=5 AND HOLIDAY_RANGE=5)
                                            OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 5 AND 6)
                                            OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 6 AND 7)
                                            OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 7 AND 8)
                                            )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF>0
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF=0
                                            AND B.TIME_PT>=A.TIME_PT 
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中最后1-2天）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                        # 4-8天，节中其他天
                        else:
                            # 获取训练数据（按照分级解除限制的逻辑进行判断）
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE BETWEEN 2 AND 3) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 3 AND 5)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 3 AND 6)
                                    )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除1级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND ((HOL_LAST=4 AND HOLIDAY_RANGE BETWEEN 2 AND 3) 
                                    OR (HOL_LAST=5 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                    OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                    OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 3 AND 5)
                                    OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 3 AND 6)
                                    )
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=6
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                            AND HOL_FALG={tmp_list['HOL_FALG']}
                                            AND ((HOL_LAST=4 AND HOLIDAY_RANGE BETWEEN 2 AND 3) 
                                            OR (HOL_LAST=5 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                            OR (HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 2 AND 4)
                                            OR (HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 3 AND 5)
                                            OR (HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 3 AND 6)
                                            )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF 
                                            AND B.EX_DIF>0
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM SOLO_FLT_KNN_LIST B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF 
                                            AND B.EX_DIF=0
                                            AND B.TIME_PT>=A.TIME_PT
                                            AND A.FLT_SEGMENT=B.FLT_SEGMENT
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中其他天）样本池数据寻找失败，建议重新检查寻找逻辑！航段信息：{tmp_list['FLT_SEGMENT']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                # 春节假日
                elif tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 1:
                    # 获取训练数据（按照分级解除限制的逻辑进行判断）
                    train_list_sql = f'''
                        SELECT *
                        FROM {self.train_name} A
                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                    '''
                    self.train_data = get_data(train_list_sql)
                    if len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] < -14:  # 除夕前3周
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE<-14
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= -14 and tmp_list['HOLIDAY_RANGE'] <= -8:  # 除夕前2周（含除夕）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE>=-14
                                AND HOLIDAY_RANGE<=-8
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= -7 and tmp_list['HOLIDAY_RANGE'] <= -1:  # 除夕前1周
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE>=-7
                                AND HOLIDAY_RANGE<=-1
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 0 and tmp_list['HOLIDAY_RANGE'] <= 5:  # 节中（除夕-初四）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND HOLIDAY_RANGE>=0
                                    AND HOLIDAY_RANGE<=5
                            '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 6 and tmp_list['HOLIDAY_RANGE'] <= 10:  # 节后高峰（初五到初九）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                        AND HOL_FALG={tmp_list['HOL_FALG']}
                                        AND HOLIDAY_RANGE>=6
                                        AND HOLIDAY_RANGE<=10
                                '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 11 and tmp_list[
                        'HOLIDAY_RANGE'] <= 15:  # 初十至十四
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                            AND HOL_FALG={tmp_list['HOL_FALG']}
                                            AND HOLIDAY_RANGE>=11
                                            AND HOLIDAY_RANGE<=15
                                    '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] == 16:  # 元宵节
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                            SELECT *
                                            FROM {self.train_name} A
                                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                                AND HOLIDAY_RANGE>=15
                                                AND HOLIDAY_RANGE<=17
                                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] > 16:  # 元宵节后
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                            SELECT *
                                            FROM {self.train_name} A
                                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                                AND HOLIDAY_RANGE>16
                                        '''
                        self.train_data = get_data(train_list_sql)
                    else:
                        pass

                    # 二次放松限制条件（对节前节后放宽限制）
                    if len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] <= -1:
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE<=-1
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 6:
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['FLT_SEGMENT']}' AND EX_DIF={tmp_list['EX_DIF']} AND (EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {tmp_list['TIME_PT']} END))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE>=6
                        '''
                        self.train_data = get_data(train_list_sql)

                    if len(self.train_data) < 1:
                        logging.warning(
                            f"春节假期样本池数据寻找失败，建议重新检查寻找逻辑！序号：'{tmp_list['HX']}'，航段信息：'{tmp_list['FLT_SEGMENT']}'，春运日期标识：{tmp_list['HOLIDAY_RANGE']}，采集时点：{tmp_list['TIME_PT']}。")

        else:
            pass

    def clean_data(self, data):
        data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
        data = self.data_deal(data)
        data.reset_index(drop=True, inplace=True)
        Y = data[self.Y_label_col]
        X = data[self.X_label_col]
        return X, Y

    def knn_est(self, knn_list):
        if knn_list['HOLIDAY_SPRING_FESTIVAL'] == 1:
            self.X_label_col = ['HOLIDAY_RANGE', 'deptime_sin', 'deptime_cos', 'HXJG_FLAG']
        X, Y = self.clean_data(self.train_data)
        # 标准化处理
        scaler_x = StandardScaler()
        x_std = scaler_x.fit_transform(X.to_numpy())
        x_train = x_std
        y_train = Y.to_numpy()

        # 设置好KNN模型参数，并训练模型
        if (knn_list['HOL_FALG'] == 0) & (knn_list['HOLIDAY_SPRING_FESTIVAL'] == 0):  # 普通日
            knn = SoloFltKnnRegressorFunction(n_neighbors=5)
        elif (knn_list['HOL_FALG'] == 1) & (knn_list['HOLIDAY_SPRING_FESTIVAL'] == 0):  # 节假日非春运
            knn = SoloFltKnnRegressorFunction(n_neighbors=1)
        else:  # 春运
            knn = SoloFltKnnRegressorFunction(n_neighbors=1)
        knn.fit(x_train, y_train)

        # 补齐待预测数据的相关字段
        X_predict, Y_predict = self.clean_data(self.predict_data)
        X_predict_std = scaler_x.transform(X_predict.to_numpy())

        # 预测行业人数增量数据
        y_pred, target_index = knn.predict(X_predict_std, Y_predict)
        target_data = self.train_data.loc[target_index]
        target_data['CREATE_TIME'] = self.config.create_time
        target_data = target_data.iloc[:, :43]
        insert_data("TMP_SOLO_FLIGHT_KNN_TARGET", target_data)

        # 将预测数据写回待预测数据（对D0数据进行特殊处理，防止D0样本都是起飞时间靠后的样本，导致D0人数增量预测偏高）/
        for i, col in enumerate(self.Y_label_col):
            if self.predict_data['EX_DIF'].values[0] <= 1 and col == 'SRS_ZL_DETR_LEFT':
                self.predict_data[col] = y_pred[:, i]
                # self.predict_data[col] = (np.maximum(sum(target_data['SRS_ZL_DETR_LEFT']) / sum(target_data['EX_DIF'] * 24 + target_data['DEP_HOUR'] - target_data['TIME_PT']),1) *
                #                           (self.predict_data['EX_DIF'] * 24 + self.predict_data['DEP_HOUR'] - self.predict_data['TIME_PT']))
            else:
                self.predict_data[col] = y_pred[:, i]
        self.tmp_data = self.predict_data

    def worker(self, i):
        data = pd.DataFrame()
        tmp_sql = f"SELECT * FROM {SOLO_ADVICE_PRICE_KNN_LIST} WHERE HX = {i + 1}"
        knn_list = get_data(tmp_sql).iloc[0]
        self.get_data(knn_list)
        if len(self.train_data) != 0:
            self.knn_est(knn_list)
            data = self.tmp_data
        return data

    def run(self):
        # 利用KNN算法预测当前预售水平下，剩余销售期内最低价格的销售增量
        delete_data("""DELETE FROM TMP_SOLO_FLIGHT_KNN_TARGET""")
        # if 1 <= self.config.file_create_hour <= 3:
        #     delete_data("""DELETE FROM TMP_SOLO_FLIGHT_KNN_TARGET""")

        # 当self.solo_flight_list中数量不足40条时，采用单进程模式，否则触发多进程模式
        if len(self.solo_flight_list) < 40:
            # 单进程模式
            logging.info(f"【SoloFlightNumberIncreaseKNN】使用单线程模式进行计算，数据量为：{len(self.solo_flight_list)}。")
            single_results = []
            for index, knn_list in self.solo_flight_list.iterrows():
                self.get_data(knn_list)
                if len(self.train_data) != 0:
                    self.knn_est(knn_list)
                    single_results.append(self.tmp_data)
            self.knn_result = pd.concat(single_results, ignore_index=True) if single_results else pd.DataFrame()
        else:
            # 多进程模式
            # 创建Manager实例
            num_cores = min(mp.cpu_count(), 4)  # 限制最大4进程
            logging.info(
                f"【SoloFlightNumberIncreaseKNN】使用{num_cores}进程模式进行计算，数据量为：{len(self.solo_flight_list)}。")
            try:
                # 创建进程池但不创建Manager（不必要开销）
                with Pool(processes=num_cores) as pool:
                    # 优化点1：使用imap_unordered提高效率（顺序无关时）
                    results = list(pool.imap_unordered(self.worker, range(len(self.solo_flight_list))))

                    # 优化点2：使用concat一次性合并（比循环append快10倍+）
                    if results:
                        self.knn_result = pd.concat(results, ignore_index=True)

            except Exception as e:
                logging.error(f"多进程处理失败: {str(e)}")

        self.knn_result = self.knn_result.sort_values(by=['FLT_SEGMENT', 'FLT_DATE'], ascending=[True, True])
        self.knn_result.reset_index(drop=True, inplace=True)
        self.knn_result['SRS_ZL_DETR_LEFT'] = np.maximum(self.knn_result['SRS_ZL_DETR_LEFT'], 1)
        return self.knn_result


def solo_knn_est_run(args):
    # Windows 需要设置启动方法
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    model = SoloFlightNumberIncreaseKNN(args)
    show_data = model.run()
    return show_data


if __name__ == '__main__':
    mp.freeze_support()
    solo_knn_est_run()
