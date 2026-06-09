"""
【程序目的】
实现对独飞航线剩余销售期内的人数增量预测功能（v2 重构版）。
继承 KNNBasePredictor，仅保留独飞特有的 data_deal / predict_write_back。

与 v1 (SoloFlightNumberIncreaseKNN.py) 的区别：
  - get_data() 由 DataFetchRules 规则链统一管理
  - clean_data / worker / run 由 KNNBasePredictor 统一管理
  - 本文件仅约 150 行，原文件 941 行
"""

import logging
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

from config.pricing_constants import (
    SOLO_FLT_KNN_NORMAL_K, SOLO_FLT_KNN_HOLIDAY_K, SOLO_FLT_KNN_SPRING_FESTIVAL_K
)
from common.database_oracle import get_predict_data, delete_predict_data, insert_predict_data
from model.KNeighborsRegressor_v2 import SoloFltKnnRegressorFunction_v2
from blocks.UniversalModule.KNNBasePredictor import KNNBasePredictor
from blocks.UniversalModule.DataFetchRules import SOLO_FLT_FETCH_CONTEXT


class SoloFlightNumberIncreaseKNN_v2(KNNBasePredictor):
    """独飞航线 KNN 预测器 v2"""

    # === 类级配置 ===
    KNN_MODEL_CLASS = SoloFltKnnRegressorFunction_v2
    DEFAULT_K = SOLO_FLT_KNN_NORMAL_K
    HOLIDAY_K = SOLO_FLT_KNN_HOLIDAY_K
    SPRING_FESTIVAL_K = SOLO_FLT_KNN_SPRING_FESTIVAL_K
    MULTIPROCESS_THRESHOLD = 40
    FETCH_CONTEXT = SOLO_FLT_FETCH_CONTEXT

    def __init__(self, config):
        self.config = config
        self.train_data = pd.DataFrame()
        self.predict_data = pd.DataFrame()
        self.tmp_data = None
        self.result_data = pd.DataFrame()

        # 独飞特有的属性
        self.X_label_col = config.solo_x_label_col.split(",")
        self.Y_label_col = config.solo_y_label_col.split(",")

        self._setup_context()
        logging.info(
            f"【SoloFlightNumberIncreaseKNN_v2】{config.version_number} 程序开始！")

    # --- 表名获取 ---
    def _get_train_table(self):
        return self.config.solo_flight_advice_price_train_table

    def _get_predict_table(self):
        return self.config.solo_flight_advice_price_predict_table

    def _get_list_table(self):
        return self.config.solo_flight_advice_price_knn_predict_list

    def _get_list_name(self):
        return self.config.solo_flight_advice_price_knn_predict_list

    def _get_cleanup_sql(self):
        return "DELETE FROM TMP_SOLO_FLIGHT_KNN_TARGET"

    # --- 特征工程 ---
    def data_deal(self, data):
        """独飞特征：deptime_sin/cos + date_sin/cos + chunjie_sin"""
        data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
        data['DEP_TIME'] = data['DEP_HOUR'] + data['DEP_MINUTE'] / 60
        data.loc[:, 'YEAR'] = data['FLT_DATE'].dt.year

        # 离港时间的正余弦函数
        data['deptime_sin'] = np.sin(2 * np.pi * data['DEP_TIME'] / 23.0)
        data['deptime_cos'] = np.cos(2 * np.pi * data['DEP_TIME'] / 23.0)
        # 按日期顺序的正余弦函数
        data['date_sin'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_sin'] = np.sin(2 * np.pi * data['date_sin'] / 366.0)
        data['date_cos'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_cos'] = np.cos(2 * np.pi * data['date_cos'] / 366.0)
        data['chunjie_sin'] = np.sin(2 * np.pi * data['HOLIDAY_RANGE'] / 30.0)
        return data

    # --- 特征覆盖（春运） ---
    def _override_features(self, knn_list):
        """春运期间使用不同的特征列"""
        if knn_list.get('HOLIDAY_SPRING_FESTIVAL') == 1:
            self.X_label_col = ['HOLIDAY_RANGE', 'deptime_sin', 'deptime_cos', 'HXJG_FLAG']

    # --- 预测写回 ---
    def predict_write_back(self, y_pred, target_index, knn_list):
        """独飞特有的写回逻辑：插入 TMP_SOLO_FLIGHT_KNN_TARGET + D0 特殊处理"""

        # 插入近邻样本数据
        target_data = self.train_data.loc[target_index]
        target_data['CREATE_TIME'] = self.config.create_time
        target_data = target_data.iloc[:, :43]
        insert_predict_data(
            """INSERT INTO TMP_SOLO_FLIGHT_KNN_TARGET VALUES(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15, :16, :17, :18, :19, :20, :21, :22, :23, :24, :25, :26, :27, :28, :29, :30, :31, :32, :33, :34, :35, :36, :37, :38, :39, :40, :41, :42, :43)""",
            target_data)

        # 将预测数据写回待预测数据
        # D0数据特殊处理：防止D0样本都是起飞时间靠后的样本导致人数增量预测偏高
        for i, col in enumerate(self.Y_label_col):
            if self.predict_data['EX_DIF'].values[0] <= 1 and col == 'SRS_ZL_DETR_LEFT':
                self.predict_data[col] = y_pred[:, i]
            else:
                self.predict_data[col] = y_pred[:, i]
        self.tmp_data = self.predict_data

    # --- 后处理 ---
    def _post_process(self):
        """独飞后处理：SRS_ZL_DETR_LEFT 不低于 1"""
        self.result_data['SRS_ZL_DETR_LEFT'] = np.maximum(
            self.result_data['SRS_ZL_DETR_LEFT'], 1)


# ============================================================
# 对外入口（与原 solo_knn_est_run 接口兼容）
# ============================================================
def solo_knn_est_run(args):
    """独飞 KNN 预测入口（v2），接口与 v1 完全兼容"""
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    mp.freeze_support()

    model = SoloFlightNumberIncreaseKNN_v2(args)
    show_data = model.run()
    return show_data


if __name__ == '__main__':
    mp.freeze_support()
    solo_knn_est_run()
