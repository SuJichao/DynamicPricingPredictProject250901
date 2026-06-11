"""
【程序目的】
获取待预测数据。
"""
from common.database_oracle import get_data
from config.config import get_argparse
from config.db_tables import (FLT_LIST_TABLE, SMALL_PART_KNN_PREDICT_TABLE, SOLO_ADVICE_PRICE_PREDICT_TABLE)


def GetPredictData(args):
    # 1 获取小份额航线待预测数据
    predict_data = get_data(f"SELECT * FROM {SMALL_PART_KNN_PREDICT_TABLE} WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {FLT_LIST_TABLE} WHERE FLT_TYPE='SMALL_PART' AND EXECUTION_STATUS=1)")
    small_part_flt_data = {
        'predict_data': predict_data
    }

    # 2 获取大份额航线待预测数据

    # 3 获取独飞航线待预测数据
    solo_flt_predict_data = get_data(f"SELECT * FROM {SOLO_ADVICE_PRICE_PREDICT_TABLE}")
    solo_part_flt_data = {
        'predict_data': solo_flt_predict_data
    }

    # 4 整合不同类型的数据
    data_set = {
        'SMALL_PART': small_part_flt_data,
        'SOLO_PART': solo_part_flt_data
    }

    # 获取所有航线类型
    flt_list = get_data(f"SELECT DISTINCT FLT_TYPE FROM {FLT_LIST_TABLE}").values.tolist()

    return data_set, flt_list