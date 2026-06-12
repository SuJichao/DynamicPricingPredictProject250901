"""
【程序目的】
针对小份额航线，实现对剩余销售期内行业人数增量的预测（v2 重构版）。
继承 KNNBasePredictor，仅保留小份额特有的 data_deal / predict_write_back。

与 v1 (SmallPartFlightCapCtrl.py) 的区别：
  - get_data() 由 DataFetchRules 规则链统一管理
  - clean_data / worker / run 由 KNNBasePredictor 统一管理
  - 本文件仅约 150 行，原文件 1022 行
"""

import logging
import multiprocessing as mp
import os
import sys
import warnings

# 确保项目根目录在 sys.path 中（支持直接运行此文件）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from config.pricing_constants import (SMALL_FLT_KNN_NORMAL_K, SMALL_FLT_KNN_HOLIDAY_K, SMALL_PART_KNN_TARGET_COLS, SMALL_PART_KNN_FEATURE_COLS, SMALL_PART_KNN_OUTPUT_COLS)
from config.db_queries import (SMALL_PART_KNN_LOG_TABLE, SMALL_PART_KNN_TRAIN_TABLE, SMALL_PART_KNN_PREDICT_TABLE, SMALL_PART_KNN_PREDICT_LIST)
from common.database_oracle import get_data, delete_data, insert_data
from common.get_logger import get_logger
from model.KNeighborsRegressor_v2 import SmallFltKnnRegressorFunction_v2
from blocks.UniversalModule.KNNBasePredictor import KNNBasePredictor
from blocks.UniversalModule.DataFetchRules import SMALL_FLT_FETCH_CONTEXT

warnings.filterwarnings('ignore', category=Warning)
get_logger()


class FlightCapControlKnn_v2(KNNBasePredictor):
    """小份额航线 KNN 预测器 v2"""

    # === 类级配置 ===
    KNN_MODEL_CLASS = SmallFltKnnRegressorFunction_v2
    DEFAULT_K = SMALL_FLT_KNN_NORMAL_K
    HOLIDAY_K = SMALL_FLT_KNN_HOLIDAY_K
    MULTIPROCESS_THRESHOLD = 10
    FETCH_CONTEXT = SMALL_FLT_FETCH_CONTEXT
    NEED_EST_DATA_SAME = True

    def __init__(self, config):
        super().__init__(config)

        # 小份额特有的属性
        self.X_label_col = list(SMALL_PART_KNN_FEATURE_COLS)
        self.Y_label_col = list(SMALL_PART_KNN_TARGET_COLS)
        self.output_columns = list(SMALL_PART_KNN_OUTPUT_COLS)
        self.log_name = SMALL_PART_KNN_LOG_TABLE

        logging.info(
            f"【FlightCapControlKnn_v2】{config.version_number} 程序开始！")

    # --- 表名获取 ---
    def _get_train_table(self):
        return SMALL_PART_KNN_TRAIN_TABLE

    def _get_predict_table(self):
        return SMALL_PART_KNN_PREDICT_TABLE

    def _get_list_table(self):
        return SMALL_PART_KNN_PREDICT_LIST

    def _get_list_name(self):
        return SMALL_PART_KNN_PREDICT_LIST

    def _get_cleanup_sql(self):
        return "DELETE FROM TMP_SELECT_HIS_DEMO"

    # --- 特征工程 ---
    def data_deal(self, data):
        """小份额特征：date_sin/cos + chunjie_sin"""
        data['FLT_DATE'] = pd.to_datetime(data['FLT_DATE'])
        # 按日期顺序进行正余弦函数
        data['date_sin'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_sin'] = np.sin(2 * np.pi * data['date_sin'] / 366.0)
        data['date_cos'] = data['FLT_DATE'].apply(lambda x: x.timetuple().tm_yday)
        data['date_cos'] = np.cos(2 * np.pi * data['date_cos'] / 366.0)
        data['chunjie_sin'] = np.sin(2 * np.pi * data['HOLIDAY_RANGE'] / 30.0)
        return data

    # --- 预测写回 ---
    def predict_write_back(self, y_pred, target_index, knn_list):
        """小份额特有的写回逻辑：插入 TMP_SELECT_HIS_DEMO + 客座率增量计算"""

        # 插入近邻样本数据
        target_data = self.train_data.iloc[target_index]
        target_data = target_data.iloc[:, :36]
        target_data.loc[:, 'HX'] = knn_list.iloc[0] if hasattr(knn_list, 'iloc') else knn_list[0]
        insert_data("TMP_SELECT_HIS_DEMO", target_data)

        hx_val = knn_list.iloc[0] if hasattr(knn_list, 'iloc') else knn_list[0]

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
          FROM (SELECT * FROM TMP_SELECT_HIS_DEMO WHERE HX={hx_val}) A
          LEFT JOIN (SELECT * FROM TMP_FUT_EST_INPUT A WHERE EXISTS (SELECT * FROM TMP_FLT_LIST B WHERE B.HX={hx_val} AND A.FLT_SEGMENT=B.DEP||B.ARR AND A.FLT_NO=B.FLT_NO AND A.TIME_PT=B.TIME_PT AND A.EX_DIF=B.EX_DIF)) C
          ON A.EX_DIF=C.EX_DIF
          LEFT JOIN RM_MXZ_AI_CTRL B
          ON INSTR(B.FLT_SEGMENT,A.FLT_SEGMENT)>0 AND INSTR(B.DOW,A.DOW)>0 AND A.EX_DIF*24-A.TIME_PT BETWEEN B.EX_DIF_E*24-B.TIME_PT_E AND B.EX_DIF_B*24-B.TIME_PT_B AND C.FLT_DATE BETWEEN B.FLT_DATE_B AND B.FLT_DATE_E
          AND C.AIR_CODE||C.FLT_NO = CASE WHEN B.FLT_NO IS NOT NULL THEN B.FLT_NO ELSE C.AIR_CODE||C.FLT_NO END
        )A
        LEFT JOIN
        HIS_FUT_FLT_LIST C
        ON A.FLT_DATE=C.FLT_DATE AND A.AIR_CODE=C.AIR_CODE AND A.FLT_NO=C.FLT_NO AND A.FLT_SEGMENT=C.FLT_SEGMENT
           AND C.EX_DIF*24-C.TIME_PT BETWEEN A.EX_DIF_END*24-A.TIME_PT_END AND A.EX_DIF_START*24-A.TIME_PT_START
        GROUP BY A.CATCH_DATE,A.FLT_DATE,A.EX_DIF,A.TIME_PT,A.AIR_CODE,A.FLT_NO,A.FLT_SEGMENT,A.FLT_ROUTE,A.DEP_HOUR,A.DEP_MINUTE,A.CAP,A.DISCAP,A.PRICE,A.BKD,A.GRS,A.BKD_SK,A.PJPJ
        """
        tmp_result = get_data(tmp_sql)

        # 将预测数据写回待预测数据
        for i, col in enumerate(self.Y_label_col):
            self.predict_data[col] = np.average(tmp_result[col])
        self.predict_data = self.predict_data.iloc[:, :26]
        self.predict_data['ARTIFICIAL_CAP_LEFT'] = np.average(tmp_result['CAP_LEFT'])
        self.predict_data['CAP_LEFT'] = np.where(
            ((self.predict_data['EX_DIF'] == 1) & (self.predict_data['TIME_PT'] >= 8)) |
            ((self.predict_data['EX_DIF'] == 0) & (self.predict_data['TIME_PT'] <= 7)),
            (self.predict_data['CAP_FINAL'] * self.predict_data['KZL_ZL_IND'] +
             self.predict_data['ARTIFICIAL_CAP_LEFT']),
            (self.predict_data['CAP_FINAL'] * (self.predict_data['KZL_ZL_MF'] + self.predict_data['KZL_ZL_IND']) / 2 +
             self.predict_data['ARTIFICIAL_CAP_LEFT'])
        )
        self.tmp_data = self.predict_data[self.output_columns]


# ============================================================
# 对外入口（与原 knn_run 接口兼容）
# ============================================================
def knn_run(args):
    """小份额 KNN 预测入口（v2），接口与 v1 完全兼容"""
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    mp.freeze_support()

    model = FlightCapControlKnn_v2(args)
    show_data = model.run()

    # 插入结果表
    insert_data("SMALL_FLT_CAP_CONTRAL_RESULT", show_data)

    # 更新经停航班 CAP_LEFT（与原逻辑完全一致）
    tmp_sql = f"""
    SELECT A.*,
       CASE WHEN LENGTH(A.FLT_ROUTE)=9 AND A.HXJG_FLAG=1 THEN A.CAP_LEFT+NVL(C.LONG_CAP_LEFT,0)
         WHEN LENGTH(A.FLT_ROUTE)=9 AND A.HXJG_FLAG=0 THEN A.CAP_LEFT+B.SHORT_CAP_LEFT
           ELSE  A.CAP_LEFT END
         AS CAP_LEFT_NEW
    FROM SMALL_FLT_CAP_CONTRAL_RESULT A
    LEFT JOIN
    (
    SELECT CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE,MAX(CAP_LEFT) AS SHORT_CAP_LEFT
    FROM SMALL_FLT_CAP_CONTRAL_RESULT
    WHERE LENGTH(FLT_ROUTE)=9 AND HXJG_FLAG=1
    GROUP BY CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE
    )B
    ON A.CATCH_DATE=B.CATCH_DATE AND A.FLT_DATE=B.FLT_DATE AND A.EX_DIF=B.EX_DIF AND A.TIME_PT=B.TIME_PT AND A.AIR_CODE=B.AIR_CODE AND A.FLT_NO=B.FLT_NO AND A.FLT_ROUTE=B.FLT_ROUTE
    LEFT JOIN
    (
    SELECT CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,FLT_ROUTE,CAP_LEFT AS LONG_CAP_LEFT
    FROM SMALL_FLT_CAP_CONTRAL_RESULT
    WHERE LENGTH(FLT_ROUTE)=9 AND HXJG_FLAG=0
    )C
    ON A.CATCH_DATE=C.CATCH_DATE AND A.FLT_DATE=C.FLT_DATE AND A.EX_DIF=C.EX_DIF AND A.TIME_PT=C.TIME_PT AND A.AIR_CODE=C.AIR_CODE AND A.FLT_NO=C.FLT_NO AND A.FLT_ROUTE=C.FLT_ROUTE
    """
    tmp_result = get_data(tmp_sql)
    delete_data("DELETE FROM SMALL_FLT_CAP_CONTRAL_RESULT2")
    insert_data("SMALL_FLT_CAP_CONTRAL_RESULT2", tmp_result)

    return tmp_result


if __name__ == '__main__':
    mp.freeze_support()
    from config.runtime_args import get_argparse
    args = get_argparse()
    knn_run(args)
