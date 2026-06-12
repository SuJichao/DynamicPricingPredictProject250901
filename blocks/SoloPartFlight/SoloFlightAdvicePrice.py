"""
【程序目的】
针对独飞航线，实现对最优价格的选择，涉及时间范围D0-30均可。
"""
import logging

import numpy as np
import pandas as pd

from config.runtime_args import get_argparse
from config.pricing_constants import (
    SOLO_FLT_BOTTOM_DISCOUNT,
    SOLO_FLT_DISCOUNT_PER_TFLAG,
    SOLO_FLT_FULL_PRICE_FALLBACK,
    SOLO_FLT_PRICE_MULTIPLIER_MAX,
    SOLO_FLT_PRICE_MULTIPLIER_MIN,
    SOLO_FLT_TARGET_LOAD_FACTOR,
)
from config.db_queries import (SOLO_SALES_RATIO_SQL, SOLO_BOTTOM_PRICE_TABLE, SOLO_PREVIOUS_PRICE_TABLE, RB_OTA_DATA_SQL, SOLO_CHUNYUN_ZFX_LIST)
from common.database_oracle import get_data, insert_data
from blocks.SoloPartFlight.SoloBkdSharpRise import bkd_sharp_rise


class SoloFltAdvicePrice(object):
    def __init__(self, config, bottom_price_demand):
        self.config = config
        # 获取独飞航线在最低价水平下的人数增量预测结果
        self.bottom_price_demand = bottom_price_demand
        # 获取库存限制数据（目前只有D0-2的航班数据）
        self.sales_ratio = get_data(SOLO_SALES_RATIO_SQL)
        self.sales_ratio = self.sales_ratio[['FLT_DATE', 'EX_DIF', 'FLT_NO', 'FLT_SEGMENT', 'MF_ZL_AHEAD']]
        self.bottom_price_demand = pd.merge(self.bottom_price_demand, self.sales_ratio, on=['FLT_DATE', 'EX_DIF', 'FLT_NO', 'FLT_SEGMENT'], how='left')
        self.bottom_price_demand['MF_ZL_AHEAD'].fillna(0, inplace=True)
        # self.bottom_price_demand['SRS_ZL_DETR_LEFT'] = np.where(self.bottom_price_demand['EX_DIF'] <= 0,
        #                                                         self.bottom_price_demand['MF_ZL_AHEAD'],
        #                                                         np.maximum(self.bottom_price_demand['SRS_ZL_DETR_LEFT'], self.bottom_price_demand['MF_ZL_AHEAD'])
        # )

        # self.bottom_price_demand['SRS_ZL_DETR_LEFT'] = np.where((self.bottom_price_demand['HOL_FALG'] == 0) | ((self.bottom_price_demand['HOLIDAY_RANGE'] <= 1) | (self.bottom_price_demand['HOLIDAY_RANGE']-self.bottom_price_demand['HOLIDAY_BEFORE_AND_AFTER'] >= 0)),
        #                                                         np.maximum(self.bottom_price_demand['SRS_ZL_DETR_LEFT'], self.bottom_price_demand['MF_ZL_AHEAD']),
        #                                                         self.bottom_price_demand['SRS_ZL_DETR_LEFT']
        # )

        # 获取独飞航班最低价格表（目前只对D0-1的独飞航班做限制，防止价格倒放）
        self.solo_bottom_price = get_data(f"SELECT CATCH_DATE,FLT_DATE,EX_DIF,FLT_NO,FLT_SEGMENT,PRICE FROM {SOLO_BOTTOM_PRICE_TABLE}")
        # 获取独飞航班上一时点的外放价格（目前只有D0-7的航班数据）
        self.solo_previous_price = get_data("oracle",
                                            data_sql=f"SELECT CATCH_DATE,EX_DIF,FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,DEP_TIME,BKD_OLD,PRICE_OTA_OLD FROM {SOLO_PREVIOUS_PRICE_TABLE}")
        # 获取当前OTA外放价格水平
        self.rb_ota_data = get_data(RB_OTA_DATA_SQL)

        self.tmp_full_fare_knn_data = None
        self.solo_flt_advice_price = None
        self.solo_flight_bottom_discount = SOLO_FLT_BOTTOM_DISCOUNT
        self.chunyun_zfx_list = get_data(f"SELECT T_DATE AS FLT_DATE,HC AS FLT_SEGMENT,正反向标识 FROM {SOLO_CHUNYUN_ZFX_LIST} WHERE 正反向标识='正向'")
        logging.info(f"【SoloFltAdvicePrice】{self.config.version_number} 程序开始！")
        self.solo_flt_max_min_price()

    def solo_flt_max_min_price(self):
        # 1 确定价格范围（上下限）
        '''
        最低价：D0-1的价格下限由前一天外放价格的中位数或平均数确定，D2以外的直接按2折作为底价；
        最高价：按全票价设置
        '''
        # 获取价格扩展列表
        self.bottom_price_demand['PRICE'].fillna(SOLO_FLT_FULL_PRICE_FALLBACK, inplace=True) # 防止全票价数据缺失
        # 独飞历史1天销售数据
        tmp_solo_flt_sales_price = get_data('SELECT FLT_DATE,CARRIER,FLT_NO,FLT_SEGMENT,SRS_SALES,PJPJ_SALES,T_FLAG FROM TMP_DP_SOLO_SRS_HIS_PH')
        self.bottom_price_demand = pd.merge(self.bottom_price_demand, tmp_solo_flt_sales_price, on=['FLT_DATE', 'CARRIER', 'FLT_NO', 'FLT_SEGMENT'])
        self.bottom_price_demand['PJPJ_SALES'].fillna(0, inplace=True)  # 防止销售票价数据缺失
        # 航班预测客座率
        self.bottom_price_demand['BKD_PLF_EST'] = (self.bottom_price_demand['SRS_ZL_DETR_LEFT'] + self.bottom_price_demand['BKD'])/self.bottom_price_demand['CAP']
        # 剩余座位对应客座率
        self.bottom_price_demand['SYZW_PLF'] = np.where(self.bottom_price_demand['CAP'] - self.bottom_price_demand['BKD'] > 0,
                                                        self.bottom_price_demand['SRS_ZL_DETR_LEFT']/(self.bottom_price_demand['CAP'] - self.bottom_price_demand['BKD']),
                                                        0)
        self.bottom_price_demand['SRS_ZL_LEFT'] = self.bottom_price_demand['SRS_ZL_DETR_LEFT']
        self.bottom_price_demand['BKD_INCOME_LEFT'] = 0

        self.bottom_price_demand = bkd_sharp_rise(self.config, self.bottom_price_demand)
        # 给出建议价格
        '''
        D29-30：直接根据历史放舱记录放舱
        D0-28：
        独飞：成交价算出来多少，就是多少，不按照T_FLAG调整。
        1、预留座位数：RM_MXZ_MF_CAP_CTRL中的MF_ZL_AHEAD字段和KNN结果取大。剩余座位对应客座率 = 预留座位数/始发剩余。
        2、突增提价正常保留。
        3、加个最终的预计客座率  =  (预留座位数 + BKD）/CAP 。 没有突增的时候，根据T_FLAG判断速度快慢：
        1）T_FLAG>=0，定价 = MAX( min(最终的预计客座率 /0.95,1.5) , 0.9 ) *  (成交价+T_FLAG /10 * 全票价），然后 和 当前外放价格 - 0.5折  取个MAX；
        2）T_FLAG  <0 ,  定价 = MAX( min(最终的预计客座率，1) , 0.9 ) *  (1+T_FLAG /10）*成交价，然后 限制范围在 当前外放价格 - 0.5折 和 当前外放价格 的区间里。
        '''
        self.bottom_price_demand['AI_ADVICE_PRICE'] = np.where(self.bottom_price_demand['PRICE_INCREASE'] <= 0,
                                                       np.where(self.bottom_price_demand['EX_DIF'] >= 29,
                                                               np.minimum(self.bottom_price_demand['PJPJ_MIN'], self.bottom_price_demand['PRICE_OTA']), # 历史成交平均价格
                                                               np.where(self.bottom_price_demand['T_FLAG'] >= 0,
                                                                        np.maximum(np.maximum(np.minimum(self.bottom_price_demand['BKD_PLF_EST'] / SOLO_FLT_TARGET_LOAD_FACTOR, SOLO_FLT_PRICE_MULTIPLIER_MAX), 1) * (self.bottom_price_demand['PJPJ_SALES'] + self.bottom_price_demand['T_FLAG'] * SOLO_FLT_DISCOUNT_PER_TFLAG * self.bottom_price_demand['PRICE']),
                                                                                   self.bottom_price_demand['PRICE_OTA'] - self.bottom_price_demand['PRICE'] * SOLO_FLT_DISCOUNT_PER_TFLAG),
                                                                        np.maximum(np.minimum(np.maximum(np.minimum(self.bottom_price_demand['BKD_PLF_EST'], 1), SOLO_FLT_PRICE_MULTIPLIER_MIN) * (1 + self.bottom_price_demand['T_FLAG'] * SOLO_FLT_DISCOUNT_PER_TFLAG) * self.bottom_price_demand['PJPJ_SALES'],
                                                                                   self.bottom_price_demand['PRICE_OTA']),
                                                                                   self.bottom_price_demand['PRICE_OTA'] - self.bottom_price_demand['PRICE'] * SOLO_FLT_DISCOUNT_PER_TFLAG)
                                                                        )
                                                               ),
                                                               self.bottom_price_demand['AVG_FARE_SK']
        )
        # self.bottom_price_demand['AI_ADVICE_PRICE'] = np.where((self.bottom_price_demand['FLT_DATE'] >= '2026-01-17') & (self.bottom_price_demand['FLT_DATE'] <= '2026-01-22') & (self.bottom_price_demand['CARRIER'] == 'MF'),
        #                                                        np.where(self.bottom_price_demand['PJPJ_FINAL'] / self.bottom_price_demand['PRICE'] > 0.3,
        #                                                                 # 正向
        #                                                                 np.maximum(self.bottom_price_demand['PRICE'] * 0.3, self.bottom_price_demand['AI_ADVICE_PRICE']),
        #                                                                 # 反向
        #                                                                 np.maximum(200, self.bottom_price_demand['AI_ADVICE_PRICE'])),
        #                                                        self.bottom_price_demand['AI_ADVICE_PRICE']
        # )
        # self.bottom_price_demand['AI_ADVICE_PRICE'] = np.where((self.bottom_price_demand['FLT_DATE'] >= '2026-01-23') & (self.bottom_price_demand['FLT_DATE'] <= '2026-01-29') & (self.bottom_price_demand['CARRIER'] == 'MF'),
        #                                                        np.where(self.bottom_price_demand['PJPJ_FINAL'] / self.bottom_price_demand['PRICE'] > 0.4,
        #                                                                 # 正向
        #                                                                 np.maximum(self.bottom_price_demand['PRICE'] * 0.4, self.bottom_price_demand['AI_ADVICE_PRICE']),
        #                                                                 # 反向
        #                                                                 np.maximum(200, self.bottom_price_demand['AI_ADVICE_PRICE'])),
        #                                                        self.bottom_price_demand['AI_ADVICE_PRICE']
        # )
        # # 春运正式开始
        # self.bottom_price_demand['AI_ADVICE_PRICE'] = np.where((self.bottom_price_demand['FLT_DATE'] >= '2026-01-30') & (self.bottom_price_demand['FLT_DATE'] <= '2026-02-19') & (self.bottom_price_demand['CARRIER'] == 'MF'),
        #                                                        np.where(self.bottom_price_demand['PJPJ_FINAL'] / self.bottom_price_demand['PRICE'] > 0.4,
        #                                                                 # 正向
        #                                                                 np.maximum(self.bottom_price_demand['PJPJ_FINAL'], self.bottom_price_demand['AI_ADVICE_PRICE']),
        #                                                                 # 反向
        #                                                                 np.maximum(200, self.bottom_price_demand['AI_ADVICE_PRICE'])),
        #                                                        self.bottom_price_demand['AI_ADVICE_PRICE']
        # )

        # self.bottom_price_demand['AI_ADVICE_PRICE'] = np.where(self.bottom_price_demand['T_FLAG'] >= 0,
        #                                                        # np.maximum(np.maximum(self.bottom_price_demand['T_FLAG'] * 0.1 * self.bottom_price_demand['PRICE'] + self.bottom_price_demand['PJPJ_SALES'], self.bottom_price_demand['SYZW_PLF'] / 0.95 * self.bottom_price_demand['PJPJ_SALES']), self.bottom_price_demand['PRICE_OTA']),
        #                                                        np.maximum(np.minimum(self.bottom_price_demand['BKD_PLF_EST'] / 0.95, 1.5),0.9) * self.bottom_price_demand['PJPJ_SALES'],
        #                                                        np.where(self.bottom_price_demand['BKD_PLF_EST'] >= 0.95,
        #                                                                 self.bottom_price_demand['PRICE_OTA'],
        #                                                                 np.minimum(np.maximum(self.bottom_price_demand['T_FLAG'] * 0.1 * self.bottom_price_demand['PRICE'] + self.bottom_price_demand['PJPJ_SALES'],
        #                                                                                       np.maximum(np.minimum(self.bottom_price_demand['BKD_PLF_EST']/0.95, 1),0.9) * self.bottom_price_demand['PJPJ_SALES']),
        #                                                                            self.bottom_price_demand['PRICE_OTA'])
        #                                                                 )
        #                                                       )
        self.result_data = self.bottom_price_demand
        tmp_data = self.bottom_price_demand[['CATCH_DATE', 'CATCH_TIME', 'TIME_PT', 'EX_DIF', 'DOW', 'FLT_DATE',
       'CARRIER', 'FLT_NO', 'FLT_SEGMENT', 'ROUTE', 'HXJG_FLAG', 'DEP_HOUR',
       'DEP_MINUTE', 'CAP', 'DISCAP', 'BKD', 'PRICE_OTA', 'PRICE', 'UP_DATE',
       'PJPJ_MIN', 'PJPJ_RATIO', 'SRS_ZL_DETR_LEFT', 'PJPJ_FINAL',
       'HOL_BEFORE_TWO_DAY', 'HOL_BEFORE_ONE_DAY', 'HOL_AFTER_ONE_DAY',
       'HOL_AFTER_TWO_DAY', 'HOLIDAY_EXACT_DAY', 'HOLIDAY_SPRING_FESTIVAL',
       'HOLIDAY_RANGE', 'HOLIDAY_BEFORE_AND_AFTER', 'HOL_FALG', 'HOL_LAST',
       'LATITUDE_DEP', 'LONGITUDE_DEP', 'LATITUDE_ARR', 'LONGITUDE_ARR',
       'BKD_DEP', 'BKD_ARR', 'CREATE_TIME', 'DEP_TIME', 'YEAR', 'MF_ZL_AHEAD',
       'SRS_SALES', 'PJPJ_SALES', 'T_FLAG', 'BKD_PLF_EST', 'SYZW_PLF',
       'SRS_ZL_LEFT', 'BKD_INCOME_LEFT', 'AI_ADVICE_PRICE']]
        insert_data("SOLO_FLIGHT_ADVICE_DATA_COPY", tmp_data)


if __name__ == '__main__':
    args = get_argparse()
    SoloFltAdvicePrice(args)