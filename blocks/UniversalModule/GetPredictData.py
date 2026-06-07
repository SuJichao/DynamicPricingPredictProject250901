"""
【程序目的】
获取待预测数据。
"""
from data_provider.data_acquisition import get_data
from config.config import get_argparse


def GetPredictData(args):
    # 1 获取小份额航线待预测数据
    predict_data = get_data("oracle",
                        data_sql=f"SELECT * FROM {args.FlightCapControlKnnPredictData} WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {args.flt_list} WHERE FLT_TYPE='SMALL_PART' AND EXECUTION_STATUS=1)")
    small_part_flt_data = {
        'predict_data': predict_data
    }

    # 2 获取大份额航线待预测数据

    # 3 获取独飞航线待预测数据
    solo_flt_predict_data = get_data("oracle", data_sql=f"SELECT * FROM {args.solo_flight_advice_price_predict_table}")
    solo_part_flt_data = {
        'predict_data': solo_flt_predict_data
    }

    # 4 整合不同类型的数据
    data_set = {
        'SMALL_PART': small_part_flt_data,
        'SOLO_PART': solo_part_flt_data
    }

    # 获取所有航线类型
    flt_list = get_data("oracle", data_sql=f"SELECT DISTINCT FLT_TYPE FROM {args.flt_list}").values.tolist()

    return data_set, flt_list