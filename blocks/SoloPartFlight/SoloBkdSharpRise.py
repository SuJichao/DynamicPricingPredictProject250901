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
from config.db_tables import RB_OTA_DATA_SQL, SOLO_PREVIOUS_PRICE_TABLE
from common.database_oracle import get_data, delete_data, insert_data
def bkd_sharp_rise(config, data):
    advice_price_data = data
    # 获取实时外放价格
    rb_ota_data = get_data(RB_OTA_DATA_SQL)
    rb_ota_data = rb_ota_data[['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT', 'AIR_CODE',
                               'FLT_NO', 'FULL_PRICE', 'AVG_FARE_SK']]
    # 防止外放价格已经到商务舱，触发价格上限缓升
    rb_ota_data['AVG_FARE_SK'] = np.where(rb_ota_data['AVG_FARE_SK'] > rb_ota_data['FULL_PRICE'],
                                          rb_ota_data['FULL_PRICE'], rb_ota_data['AVG_FARE_SK'])
    rb_ota_data.reset_index(drop=True, inplace=True)
    advice_price_data.reset_index(drop=True, inplace=True)
    # 将最新建议价格与上一时点的采集价格进行左连接
    tmp_price_advice_result = pd.merge(advice_price_data, rb_ota_data,
                                       left_on=['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT', 'CARRIER',
                                                'FLT_NO', 'PRICE'],
                                       right_on=['FLT_DATE', 'CATCH_DATE', 'FLT_SEGMENT', 'EX_DIF', 'TIME_PT',
                                                 'AIR_CODE', 'FLT_NO', 'FULL_PRICE'],
                                       how='left')


    solo_previous_price = get_data(f"SELECT EX_DIF AS EX_DIF_OLD,TIME_PT_OLD,FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,BKD_OLD,PRICE_OTA_OLD FROM {SOLO_PREVIOUS_PRICE_TABLE}")
    bkd_sluggish_record = get_data(f"SELECT * FROM BKD_SLUGGISH_RECORD")

    # 【情况1】对订座突增航班进行价格上涨（目前仅限D0-30）
    # solo_previous_price主要用于存储上一采集时点的外放价格和订座人数
    tmp_solo_advice_data = pd.merge(tmp_price_advice_result, solo_previous_price, how='left',
                                    on=['FLT_DATE', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT'])
    # 防止多个连续时点未采集到数据，导致人数增量计算异常
    tmp_solo_advice_data['BKD_INC'] = (tmp_solo_advice_data['BKD'] - tmp_solo_advice_data['BKD_OLD'])/(tmp_solo_advice_data['TIME_PT']+(tmp_solo_advice_data['EX_DIF_OLD']-tmp_solo_advice_data['EX_DIF'])*24-tmp_solo_advice_data['TIME_PT_OLD'])
    # tmp_solo_advice_data['PRICE_INCREASE'] = np.where(tmp_solo_advice_data['EX_DIF'] < 7, tmp_solo_advice_data['BKD_INC'].fillna(0)/5, tmp_solo_advice_data['BKD_INC'].fillna(0)/10)
    # D0-1
    tmp_solo_advice_data['PRICE_INCREASE'] = np.where(tmp_solo_advice_data['EX_DIF'] <= 1,
                                                      tmp_solo_advice_data['BKD_INC'].fillna(0) // SOLO_BKD_SURGE_D0_DIVISOR,
                                                      np.where((tmp_solo_advice_data['EX_DIF'] >= 2) & (tmp_solo_advice_data['EX_DIF'] <= 7),
                                                               tmp_solo_advice_data['BKD_INC'].fillna(0) // SOLO_BKD_SURGE_D2_D7_DIVISOR,
                                                               np.where(tmp_solo_advice_data['EX_DIF'] >= 8,
                                                                        tmp_solo_advice_data['BKD_INC'].fillna(0) // SOLO_BKD_SURGE_D8_DIVISOR,
                                                                        0)
                                                               )
                                                      )
    tmp_solo_advice_data['PRICE_INCREASE'] = tmp_solo_advice_data['PRICE_INCREASE'].fillna(0)
    tmp_solo_advice_data['PRICE_INCREASE'] = np.minimum(tmp_solo_advice_data['PRICE_INCREASE'].apply(math.floor), SOLO_BKD_SURGE_STEP_CAP) # 最多涨价N折
    # 如果当前时点的人数订座增量出现突增（预计客座率大于90%时才触发）
    tmp_solo_advice_data['AVG_FARE_SK'] = np.where((tmp_solo_advice_data['PRICE_INCREASE'] >= 1) & (tmp_solo_advice_data['BKD_PLF_EST'] > SOLO_BKD_SURGE_LOAD_THRESHOLD),
                                                   # 防止突增后的价格低于模型建议价格（20250417改为直接看突增价格）
                                                   tmp_solo_advice_data['PRICE_OTA'] + round_to_10(tmp_solo_advice_data['PRICE_INCREASE'] * tmp_solo_advice_data['PRICE'] * SOLO_FLT_DISCOUNT_PER_TFLAG),
                                                   tmp_solo_advice_data['AVG_FARE_SK'])
                                                   # np.maximum(tmp_solo_advice_data['AVG_FARE_SK_y'] + round((tmp_solo_advice_data['PRICE_INCREASE'] * tmp_solo_advice_data['FULL_PRICE'] * 0.1)/10)*10, tmp_solo_advice_data['AVG_FARE_SK_x']),
                                                   # tmp_solo_advice_data['AVG_FARE_SK_x'])

    # 插入当前时点出现订座突增的航班数据
    bkd_sudden_increase_record = tmp_solo_advice_data[tmp_solo_advice_data['PRICE_INCREASE'] >= 1][['CATCH_DATE', 'EX_DIF', 'FLT_DATE', 'TIME_PT', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT', 'AVG_FARE_SK', 'BKD_INC', 'UP_DATE']]
    bkd_sudden_increase_record.reset_index(drop=True, inplace=True)
    bkd_sudden_increase_record['PID'] = uuid.uuid1()
    for i in range(len(bkd_sudden_increase_record)):
        bkd_sudden_increase_record.at[i, 'PID'] = uuid.uuid1()
    bkd_sudden_increase_record['PID'] = bkd_sudden_increase_record['PID'].astype('str')
    insert_data("BKD_SUDDEN_INCREASE_RECORD", bkd_sudden_increase_record)

    # 删除多次插入的突增数据（旧数据）
    tmp_sql = """
        DELETE FROM BKD_SUDDEN_INCREASE_RECORD S
        WHERE PID IN
        (
        SELECT PID
        FROM
        (
          SELECT A.*,ROW_NUMBER () OVER (PARTITION BY A.FLT_DATE,A.FLT_NO,A.FLT_SEGMENT ORDER BY A.CREATE_TIME DESC) RN
          FROM BKD_SUDDEN_INCREASE_RECORD A
        )WHERE RN!=1
        )
    """
    delete_data(tmp_sql)

    # 当独飞订座突增表中有数据时，维持定价不变，否则回调价格
    bkd_sudden_increase_record = get_data("SELECT CATCH_DATE,EX_DIF,FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,ADVICE_PRICE,PID FROM BKD_SUDDEN_INCREASE_RECORD")
    tmp_solo_advice_data = pd.merge(tmp_solo_advice_data, bkd_sudden_increase_record, how='left', on=['CATCH_DATE', 'EX_DIF', 'FLT_DATE', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT'])
    tmp_solo_advice_data['AVG_FARE_SK'] = np.where(tmp_solo_advice_data['ADVICE_PRICE'] > 0,
                                                   np.maximum(tmp_solo_advice_data['ADVICE_PRICE'], tmp_solo_advice_data['AVG_FARE_SK']),
                                                   tmp_solo_advice_data['AVG_FARE_SK'])
    result_data = tmp_solo_advice_data

    return result_data