"""
【程序目的】
针对小份额航线，实现对剩余销售期内行业人数增量的预测。
"""
# -*- coding: utf-8 -*-
import logging
import multiprocessing as mp
import sys
import warnings

import numpy as np
import pandas as pd
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler

from config.config import get_argparse
from config.pricing_constants import (SMALL_PART_KNN_TARGET_COLS, SMALL_PART_KNN_FEATURE_COLS, SMALL_PART_KNN_OUTPUT_COLS)
from config.db_tables import (
    SMALL_PART_KNN_IDENTIFIER,
    SMALL_PART_KNN_TRAIN_TABLE,
    SMALL_PART_KNN_PREDICT_TABLE,
    SMALL_PART_KNN_PREDICT_LIST,
    SMALL_PART_KNN_LOG_TABLE,
)
from common.database_oracle import get_data, delete_data, insert_data
from common.get_logger import get_logger
from model.KNeighborsRegressor import SmallFltKnnRegressorFunction

warnings.filterwarnings('ignore', category=Warning)

get_logger()

def data_deal(data):
    # 增加月份、日期和时刻的正余弦函数
    data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
    # 按日期顺序进行正余弦函数
    data['date_sin'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
    data['date_sin'] = np.sin(2 * np.pi * data['date_sin'] / 366.0)
    data['date_cos'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
    data['date_cos'] = np.cos(2 * np.pi * data['date_cos'] / 366.0)
    data['chunjie_sin'] = np.sin(2 * np.pi * data['HOLIDAY_RANGE'] / 30.0)
    return data


def est_data_same(train_data, est_data):
    # 获取训练集列名集合
    train_columns = train_data.columns.values.tolist()
    miss_columns = set(train_data.columns) - set(est_data.columns)
    # 补足训练集中缺失的列
    for col in miss_columns:
        est_data[col] = 0
    # 删除训练集中多余的列
    adu_columns = set(train_data.columns) - set(est_data.columns)
    est_data.drop(list(adu_columns), axis=1, inplace=True)
    est_data = est_data.reindex(train_columns, axis=1)
    return est_data


class FlightCapControlKnn():

    def __init__(self, config):
        self.config = config
        self.object_name = SMALL_PART_KNN_IDENTIFIER
        self.train_name = SMALL_PART_KNN_TRAIN_TABLE
        self.predict_name = SMALL_PART_KNN_PREDICT_TABLE
        self.list_name = SMALL_PART_KNN_PREDICT_LIST
        self.knn_list = get_data(f"SELECT * FROM {self.list_name}")
        self.train_data = None
        self.predict_data = None

        self.root_dic = config.root_dir
        self.log_dic = config.root_dir + '/log'
        self.log_name = SMALL_PART_KNN_LOG_TABLE

        self.output_name = config.FlightCapControlKnnOutput
        self.X_columns = list(SMALL_PART_KNN_FEATURE_COLS)
        self.Y_columns = list(SMALL_PART_KNN_TARGET_COLS)
        self.output_columns = list(SMALL_PART_KNN_OUTPUT_COLS)
        self.show_data = pd.DataFrame()
        self.tmp_data = None

    def get_data(self, tmp_list):
        # 分批次读取待预测数据
        # 获取待预测数据
        # 普通日
        if tmp_list['HOL_FALG'] == 0:
            predict_list_sql = f'''
                SELECT *
                FROM {self.predict_name} A
                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND TIME_PT={tmp_list['TIME_PT']} AND DOW={tmp_list['DOW']}
                AND FLT_NO='{tmp_list['FLT_NO']}' AND HXJG_FLAG={tmp_list['HXJG_FLAG']} AND HOL_FALG={tmp_list['HOL_FALG']}
            '''
            self.predict_data = get_data(predict_list_sql)
            # 获取训练数据（按照分级解除限制的逻辑进行判断）
            if 7 <= tmp_list['MONTH'] <= 8:  # 当目标航班处于暑运时间内，严格对标其历史样本（在可以找到样本数据的前提下）
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']})) AND DOW={tmp_list['DOW']} 
                    AND HXJG_FLAG={tmp_list['HXJG_FLAG']} AND HOL_FALG='{tmp_list['HOL_FALG']}' AND TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) BETWEEN 7 AND 8
                '''
            elif tmp_list['MONTH'] < 7 or tmp_list['MONTH'] > 8:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']})) AND DOW={tmp_list['DOW']}
                    AND HXJG_FLAG={tmp_list['HXJG_FLAG']} AND HOL_FALG='{tmp_list['HOL_FALG']}' AND (TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) > 8 OR TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) < 7)
                '''
            else:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']})) AND DOW={tmp_list['DOW']}
                    AND HXJG_FLAG={tmp_list['HXJG_FLAG']} AND HOL_FALG='{tmp_list['HOL_FALG']}'
                '''
            self.train_data = get_data(train_list_sql)
            # 解除1级限制
            if len(self.train_data) <= 1:
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                    AND HOL_FALG='{tmp_list['HOL_FALG']}'
                '''
                self.train_data = get_data(train_list_sql)
                # 解除2级限制
                if len(self.train_data) <= 1:
                    train_list_sql = f'''
                        SELECT *
                        FROM {self.train_name} A
                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                        AND HOL_FALG='{tmp_list['HOL_FALG']}'
                    '''
                    self.train_data = get_data(train_list_sql)
                    if len(self.train_data) <= 1:
                        logging.warning(
                            f"普通日样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
            else:
                pass

        # 节假日
        elif tmp_list['HOL_FALG'] == 1:
            # 获取待预测数据
            predict_list_sql = f'''
                SELECT *
                FROM {self.predict_name} A
                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND TIME_PT={tmp_list['TIME_PT']} AND FLT_NO='{tmp_list['FLT_NO']}'
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
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                            AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                            AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                        UNION ALL
                        SELECT *
                        FROM {self.train_name} A
                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                        AND A.HOL_FALG=0
                        AND A.DOW=6
                        AND A.EX_DIF=B.EX_DIF 
                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                        AND A.FLT_SEGMENT=B.DEP||B.ARR
                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                        )
                    '''
                    self.train_data = get_data(train_list_sql)
                    # 解除2级限制
                    if len(self.train_data) <= 1:
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                                AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF 
                            AND B.EX_DIF>0
                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF 
                            AND B.EX_DIF=0 
                            AND B.TIME_PT>=A.TIME_PT
                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                        '''
                        self.train_data = get_data(train_list_sql)
                        if len(self.train_data) <= 1:
                            logging.warning(
                                f"节假日（1天）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
            # 放假天数为3天的情况
            elif (tmp_list['HOL_LAST'] == 2) & (tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 0):
                # 获取训练数据（按照分级解除限制的逻辑进行判断）
                train_list_sql = f'''
                    SELECT *
                    FROM {self.train_name} A
                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                            AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                            AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                        UNION ALL
                        SELECT *
                        FROM {self.train_name} A
                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                        AND A.HOL_FALG=0
                        AND DECODE(B.HOLIDAY_RANGE,-2,4,-1,5,1,6,2,6,3,7,4,1,5,2)=A.DOW 
                        AND A.EX_DIF=B.EX_DIF 
                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                        AND A.FLT_SEGMENT=B.DEP||B.ARR
                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                        )
                    '''
                    self.train_data = get_data(train_list_sql)
                    # 解除2级限制
                    if len(self.train_data) <= 1:
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOL_BEFORE_TWO_DAY={tmp_list['HOL_BEFORE_TWO_DAY']} AND HOL_BEFORE_ONE_DAY={tmp_list['HOL_BEFORE_ONE_DAY']} 
                                AND HOL_AFTER_ONE_DAY={tmp_list['HOL_AFTER_ONE_DAY']} AND HOL_AFTER_TWO_DAY={tmp_list['HOL_AFTER_TWO_DAY']} 
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF
                            AND B.EX_DIF>0 
                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                            UNION ALL
                            SELECT *
                            FROM {self.train_name} A
                            WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                            AND A.HOL_FALG=0
                            AND A.DOW=B.DOW
                            AND A.EX_DIF=B.EX_DIF
                            AND B.EX_DIF=0 
                            AND B.TIME_PT>=A.TIME_PT
                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                            )
                        '''
                        self.train_data = get_data(train_list_sql)
                        if len(self.train_data) <= 1:
                            logging.warning(
                                f"节假日（3天）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
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
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                        '''
                        self.train_data = get_data(train_list_sql)
                        # 解除1级限制
                        if len(self.train_data) <= 1:
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND DECODE(B.HOLIDAY_RANGE,-2,4,-1,5)=A.DOW
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除2级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOLIDAY_RANGE<0
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF>0
                                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                        AND A.FLT_SEGMENT=B.DEP||B.ARR
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=B.DOW
                                    AND A.EX_DIF=B.EX_DIF
                                    AND B.EX_DIF=0
                                    AND B.TIME_PT>=A.TIME_PT
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                '''
                                self.train_data = get_data(train_list_sql)
                                if len(self.train_data) <= 1:
                                    logging.warning(
                                        f"节假日（4天以上节前）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                    # 4天以上（节后）
                    elif tmp_list['HOL_LAST'] - tmp_list['HOLIDAY_RANGE'] < 0:
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                        '''
                        self.train_data = get_data(train_list_sql)
                        # 解除1级限制
                        if len(self.train_data) <= 1:
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                    AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                                UNION ALL
                                SELECT *
                                FROM {self.train_name} A
                                WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND DECODE(B.HOLIDAY_RANGE,-2,2,-1,1)=A.DOW
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                            '''
                            self.train_data = get_data(train_list_sql)
                            # 解除2级限制
                            if len(self.train_data) <= 1:
                                train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                        AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                                        AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF>0
                                        AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                        AND A.FLT_SEGMENT=B.DEP||B.ARR
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                    UNION ALL
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                        AND A.HOL_FALG=0
                                        AND A.DOW=B.DOW
                                        AND A.EX_DIF=B.EX_DIF 
                                        AND B.EX_DIF=0
                                        AND B.TIME_PT>=A.TIME_PT
                                        AND A.FLT_SEGMENT=B.DEP||B.ARR
                                        AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                    )
                                '''
                                self.train_data = get_data(train_list_sql)
                                if len(self.train_data) <= 1:
                                    logging.warning(
                                        f"节假日（4天以上节后）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
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
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=5 
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF>0 
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF=0 
                                            AND B.TIME_PT>=A.TIME_PT
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中第1-2天）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                        # 4-8天，节中最后第1-2天
                        elif (tmp_list['HOL_LAST'] >= 4 and tmp_list['HOL_LAST'] <= 5 and tmp_list[
                            'HOLIDAY_RANGE'] == tmp_list['HOLIDAY_RANGE']) or (
                                tmp_list['HOL_LAST'] >= 6 and tmp_list['HOL_LAST'] <= 8 and tmp_list[
                            'HOLIDAY_RANGE'] >= tmp_list['HOL_LAST'] - 1 and tmp_list['HOLIDAY_RANGE'] <=
                                tmp_list['HOL_LAST']):
                            # 获取训练数据（按照分级解除限制的逻辑进行判断）
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=7
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF>0
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF
                                            AND B.EX_DIF=0
                                            AND B.TIME_PT>=A.TIME_PT 
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中最后1-2天）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                        # 4-8天，节中其他天
                        else:
                            # 获取训练数据（按照分级解除限制的逻辑进行判断）
                            train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                    AND A.HOL_FALG=0
                                    AND A.DOW=6
                                    AND A.EX_DIF=B.EX_DIF 
                                    AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                    AND A.FLT_SEGMENT=B.DEP||B.ARR
                                    AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                )
                                '''
                                self.train_data = get_data(train_list_sql)
                                # 解除2级限制
                                if len(self.train_data) <= 1:
                                    train_list_sql = f'''
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF 
                                            AND B.EX_DIF>0
                                            AND CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END=CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END 
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                        UNION ALL
                                        SELECT *
                                        FROM {self.train_name} A
                                        WHERE EXISTS (SELECT * FROM {SMALL_PART_KNN_PREDICT_LIST} B WHERE B.HX={tmp_list['HX']}
                                            AND A.HOL_FALG=0
                                            AND A.DOW=B.DOW
                                            AND A.EX_DIF=B.EX_DIF 
                                            AND B.EX_DIF=0
                                            AND B.TIME_PT>=A.TIME_PT
                                            AND A.FLT_SEGMENT=B.DEP||B.ARR
                                            AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL
                                        )
                                    '''
                                    self.train_data = get_data(train_list_sql)
                                    if len(self.train_data) <= 1:
                                        logging.warning(
                                            f"节假日（4天以上节中其他天）样本池数据寻找失败，建议重新检查寻找逻辑！序号：{tmp_list['HX']}：航段信息：{tmp_list['DEP']}{tmp_list['ARR']}，距离起飞天数{tmp_list['EX_DIF']}，采集时点{tmp_list['TIME_PT']}。")
                # 春节假日
                elif tmp_list['HOLIDAY_SPRING_FESTIVAL'] == 1:
                    # 获取训练数据（按照分级解除限制的逻辑进行判断）
                    train_list_sql = f'''
                        SELECT *
                        FROM {self.train_name} A
                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                            AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']} 
                            AND HOL_FALG={tmp_list['HOL_FALG']} AND HOL_LAST={tmp_list['HOL_LAST']} AND HOLIDAY_RANGE={tmp_list['HOLIDAY_RANGE']}
                    '''
                    self.train_data = get_data(train_list_sql)

                    if len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] < -7:  # 除夕前2周
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE<-7
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= -7 and tmp_list[
                        'HOLIDAY_RANGE'] <= -1:  # 除夕前1周（含除夕）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                            SELECT *
                            FROM {self.train_name} A
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE>=-7
                                AND HOLIDAY_RANGE<=1
                        '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 2 and tmp_list[
                        'HOLIDAY_RANGE'] <= 5:  # 节中（初一-初四）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                SELECT *
                                FROM {self.train_name} A
                                WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                    AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                    AND HOL_FALG={tmp_list['HOL_FALG']}
                                    AND HOLIDAY_RANGE>=2
                                    AND HOLIDAY_RANGE<=5
                            '''
                        self.train_data = get_data(train_list_sql)
                    elif len(self.train_data) < 1 and tmp_list['HOLIDAY_RANGE'] >= 6 and tmp_list[
                        'HOLIDAY_RANGE'] <= 10:  # 节后高峰（初五到初九）
                        # 获取训练数据（按照分级解除限制的逻辑进行判断）
                        train_list_sql = f'''
                                    SELECT *
                                    FROM {self.train_name} A
                                    WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                        WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
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
                            WHERE FLT_SEGMENT='{tmp_list['DEP']}{tmp_list['ARR']}' AND EX_DIF={tmp_list['EX_DIF']} AND ((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={tmp_list['TIME_PT']}))
                                AND HOLIDAY_SPRING_FESTIVAL={tmp_list['HOLIDAY_SPRING_FESTIVAL']}
                                AND HOL_FALG={tmp_list['HOL_FALG']}
                                AND HOLIDAY_RANGE>=6
                        '''
                        self.train_data = get_data(train_list_sql)

                    if len(self.train_data) < 1:
                        logging.warning(
                            f"春节假期样本池数据寻找失败，建议重新检查寻找逻辑！序号：'{tmp_list['HX']}'：航段信息：'{tmp_list['DEP']}{tmp_list['ARR']}'，春运日期标识：{tmp_list['HOLIDAY_RANGE']}，采集时点：{tmp_list['TIME_PT']}。")
                else:
                    pass

        else:
            pass

    def clean_data(self, data):
        data.reset_index(drop=True, inplace=True)
        data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
        data = data_deal(data)
        Y = data[self.Y_columns]
        data = data[self.X_columns]
        X = data.copy()
        return X, Y

    def knn_est(self, knn_list):
        X, Y = self.clean_data(self.train_data)
        # 标准化处理
        scaler_x = StandardScaler()
        x_std = scaler_x.fit_transform(X.to_numpy())
        x_train = x_std
        y_train = Y.to_numpy()
        # 设置好KNN模型参数，并训练模型
        if knn_list['HOL_FALG'] == 0:
            knn = SmallFltKnnRegressorFunction(n_neighbors=3)
        else:
            knn = SmallFltKnnRegressorFunction(n_neighbors=1)
        knn.fit(x_train, y_train)

        # 补齐待预测数据的相关字段
        X_predict, Y_predict = self.clean_data(self.predict_data)
        X_predict = est_data_same(X, X_predict)
        X_predict_std = scaler_x.transform(X_predict.to_numpy())

        # 预测行业客座率增量数据
        y_pred, target_index = knn.predict(X_predict_std, Y_predict)
        target_data = self.train_data.loc[target_index]
        target_data = target_data.iloc[:, :36]
        target_data.loc[:, 'HX'] = knn_list[0]
        insert_data("TMP_SELECT_HIS_DEMO", target_data)

        # 寻找当前样本剩余销售期内的客座率增量
        tmp_sql = f"""
        SELECT A.CATCH_DATE,A.FLT_DATE,A.EX_DIF,A.TIME_PT,A.AIR_CODE,A.FLT_NO,A.FLT_SEGMENT,A.FLT_ROUTE,A.DEP_HOUR,A.DEP_MINUTE,A.CAP,A.DISCAP,A.PRICE,A.BKD,A.GRS,A.BKD_SK,A.PJPJ,
            GREATEST(SUM(C.KZL_ZL_MF),0) AS KZL_ZL_MF,SUM(C.KZL_ZL_IND) AS KZL_ZL_IND,NVL(AVG(A.CAP_LEFT),0) AS CAP_LEFT
        FROM
        (
          SELECT A.CATCH_DATE,A.FLT_DATE,A.EX_DIF,A.TIME_PT,A.AIR_CODE,A.FLT_NO,A.FLT_SEGMENT,A.FLT_ROUTE,A.DEP_HOUR,A.DEP_MINUTE,A.CAP,A.DISCAP,A.PRICE,A.BKD,A.GRS,A.BKD_SK,A.PJPJ,
                 A.EX_DIF AS EX_DIF_START,NVL(B.EX_DIF_E,0) AS EX_DIF_END,
                 A.TIME_PT AS TIME_PT_START,CASE WHEN B.CAP_LEFT IS NOT NULL AND B.TIME_PT_E IS NOT NULL THEN B.TIME_PT_E ELSE A.DEP_HOUR-1 END AS TIME_PT_END,
                 B.CAP_LEFT
          FROM (SELECT * FROM TMP_SELECT_HIS_DEMO WHERE HX={knn_list[0]}) A--样本选择结果
          LEFT JOIN (SELECT * FROM TMP_FUT_EST_INPUT A WHERE EXISTS (SELECT * FROM TMP_FLT_LIST B WHERE B.HX={knn_list[0]} AND A.FLT_SEGMENT=B.DEP||B.ARR AND A.FLT_NO=B.FLT_NO AND A.TIME_PT=B.TIME_PT AND A.EX_DIF=B.EX_DIF)) C
          ON A.EX_DIF=C.EX_DIF
          LEFT JOIN RM_MXZ_AI_CTRL B--收益管理系统动态托管规则数据
          ON INSTR(B.FLT_SEGMENT,A.FLT_SEGMENT)>0 AND INSTR(B.DOW,A.DOW)>0 AND A.EX_DIF*24-A.TIME_PT BETWEEN B.EX_DIF_E*24-B.TIME_PT_E AND B.EX_DIF_B*24-B.TIME_PT_B AND C.FLT_DATE BETWEEN B.FLT_DATE_B AND B.FLT_DATE_E
          AND C.AIR_CODE||C.FLT_NO = CASE WHEN B.FLT_NO IS NOT NULL THEN B.FLT_NO ELSE C.AIR_CODE||C.FLT_NO END
        )A
        LEFT JOIN
        HIS_FUT_FLT_LIST C--样本池
        ON A.FLT_DATE=C.FLT_DATE AND A.AIR_CODE=C.AIR_CODE AND A.FLT_NO=C.FLT_NO AND A.FLT_SEGMENT=C.FLT_SEGMENT 
           AND C.EX_DIF*24-C.TIME_PT BETWEEN A.EX_DIF_END*24-A.TIME_PT_END AND A.EX_DIF_START*24-A.TIME_PT_START
        GROUP BY A.CATCH_DATE,A.FLT_DATE,A.EX_DIF,A.TIME_PT,A.AIR_CODE,A.FLT_NO,A.FLT_SEGMENT,A.FLT_ROUTE,A.DEP_HOUR,A.DEP_MINUTE,A.CAP,A.DISCAP,A.PRICE,A.BKD,A.GRS,A.BKD_SK,A.PJPJ
        """
        tmp_result = get_data(tmp_sql)

        # 将预测数据写回待预测数据
        for i, col in enumerate(self.Y_columns):
            self.predict_data[col] = np.average(tmp_result[col])
        self.predict_data = self.predict_data.iloc[:, :26]
        self.predict_data['ARTIFICIAL_CAP_LEFT'] = np.average(tmp_result['CAP_LEFT'])
        # self.predict_data['CAP_LEFT'] = (self.predict_data['CAP_FINAL'] *
        #                                  np.maximum(self.predict_data['KZL_ZL_MF'], self.predict_data['KZL_ZL_IND']) +
        #                                  self.predict_data['ARTIFICIAL_CAP_LEFT'])
        # 在夜间需要单独对KZL_ZL_MF和KZL_ZL_IND取平均值
        # self.predict_data['CAP_LEFT'] = self.predict_data['CAP_FINAL'] * (self.predict_data['KZL_ZL_MF'] + self.predict_data['KZL_ZL_IND']) / 2 + self.predict_data['ARTIFICIAL_CAP_LEFT']
        self.predict_data['CAP_LEFT'] = np.where(((self.predict_data['EX_DIF'] == 1) & (self.predict_data['TIME_PT'] >= 8)) | ((self.predict_data['EX_DIF'] == 0) & (self.predict_data['TIME_PT'] <= 7)),
                                                 (self.predict_data['CAP_FINAL'] * self.predict_data['KZL_ZL_IND'] +
                                                  self.predict_data['ARTIFICIAL_CAP_LEFT']),
                                                 (self.predict_data['CAP_FINAL'] * (self.predict_data['KZL_ZL_MF'] + self.predict_data['KZL_ZL_IND']) / 2 +
                                                  self.predict_data['ARTIFICIAL_CAP_LEFT'])
                                                 )
        self.tmp_data = self.predict_data[self.output_columns]

    def worker(self, i):
        data = pd.DataFrame()
        tmp_sql = f"SELECT * FROM {self.list_name} WHERE HX = {i + 1}"
        knn_list = get_data(tmp_sql).iloc[0]
        self.get_data(knn_list)
        if len(self.train_data) > 0:
            self.knn_est(knn_list)
            data = self.tmp_data
        return data

    def run(self):
        logging.info(f"{SMALL_PART_KNN_IDENTIFIER}{self.config.version_number} 程序开始！")
        delete_data("""DELETE FROM TMP_SELECT_HIS_DEMO""")
        delete_data("""DELETE FROM SMALL_FLT_CAP_CONTRAL_RESULT""")

        # 当self.knn_list中数量不足30条时，采用单进程模式，否则触发多进程模式
        if len(self.knn_list) < 10:
            # 单进程模式
            logging.info(f"{SMALL_PART_KNN_IDENTIFIER}使用单进程模式进行计算，数据量为：{len(self.knn_list)}。")
            single_results = []
            for index, knn_list in self.knn_list.iterrows():
                self.get_data(knn_list)
                if len(self.train_data) > 0:
                    self.knn_est(knn_list)
                    single_results.append(self.tmp_data)
            self.show_data = pd.concat(single_results, ignore_index=True) if single_results else pd.DataFrame()
        else:
            # 多进程模式
            # 创建Manager实例
            num_cores = min(mp.cpu_count(), 4)  # 限制最大4进程
            logging.info(
                f"{SMALL_PART_KNN_IDENTIFIER}使用{num_cores}进程模式进行计算，数据量为：{len(self.knn_list)}。")
            try:
                # 创建进程池但不创建Manager（不必要开销）
                with Pool(processes=num_cores) as pool:
                    # 优化点1：使用imap_unordered提高效率（顺序无关时）
                    results = list(pool.imap_unordered(self.worker, range(len(self.knn_list))))

                    # 优化点2：使用concat一次性合并（比循环append快10倍+）
                    if results:
                        self.show_data = pd.concat(results, ignore_index=True)

            except Exception as e:
                logging.error(f"多进程处理失败: {str(e)}")

        self.show_data = self.show_data.sort_values(by=['FLT_SEGMENT', 'FLT_DATE'], ascending=[True, True])
        self.show_data.reset_index(drop=True, inplace=True)
        return self.show_data


def knn_run(args):
    # Windows 需要设置启动方法
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    model = FlightCapControlKnn(args)
    show_data = model.run()
    insert_data("SMALL_FLT_CAP_CONTRAL_RESULT", show_data)
    # 更新经停航班CAP_LEFT
    tmp_sql = f"""
    SELECT A.*,--B.SHORT_CAP_LEFT,NVL(C.LONG_CAP_LEFT,0) AS LONG_CAP_LEFT,
       CASE WHEN LENGTH(A.FLT_ROUTE)=9 AND A.HXJG_FLAG=1 THEN A.CAP_LEFT+NVL(C.LONG_CAP_LEFT,0)
         WHEN LENGTH(A.FLT_ROUTE)=9 AND A.HXJG_FLAG=0 THEN A.CAP_LEFT+B.SHORT_CAP_LEFT
           ELSE  A.CAP_LEFT END 
         AS CAP_LEFT_NEW
    FROM SMALL_FLT_CAP_CONTRAL_RESULT A--KNN计算后的原始结果
    LEFT JOIN
    --短段MAX
    (
    SELECT CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE,MAX(CAP_LEFT) AS SHORT_CAP_LEFT
    FROM SMALL_FLT_CAP_CONTRAL_RESULT
    WHERE LENGTH(FLT_ROUTE)=9 AND HXJG_FLAG=1
    GROUP BY CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE
    )B
    ON A.CATCH_DATE=B.CATCH_DATE AND A.FLT_DATE=B.FLT_DATE AND A.EX_DIF=B.EX_DIF AND A.TIME_PT=B.TIME_PT AND A.AIR_CODE=B.AIR_CODE AND A.FLT_NO=B.FLT_NO AND A.FLT_ROUTE=B.FLT_ROUTE
    LEFT JOIN
    --长段
    (
    SELECT CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE,CAP_LEFT AS LONG_CAP_LEFT
    FROM SMALL_FLT_CAP_CONTRAL_RESULT
    WHERE LENGTH(FLT_ROUTE)=9 AND HXJG_FLAG=0
    )C
    ON A.CATCH_DATE=C.CATCH_DATE AND A.FLT_DATE=C.FLT_DATE AND A.EX_DIF=C.EX_DIF AND A.TIME_PT=C.TIME_PT AND A.AIR_CODE=C.AIR_CODE AND A.FLT_NO=C.FLT_NO AND A.FLT_ROUTE=C.FLT_ROUTE
    """
    tmp_result = get_data(tmp_sql)
    delete_data("""DELETE FROM SMALL_FLT_CAP_CONTRAL_RESULT2""")
    insert_data("SMALL_FLT_CAP_CONTRAL_RESULT2", tmp_result)

    return tmp_result


if __name__ == '__main__':
    mp.freeze_support()
    args = get_argparse()
    knn_run(args)
