"""
【程序目的】
实现Python对Oracle数据库 或 Excel数据 各种形式的获取。
"""
import logging

import pandas as pd

from common.database_oracle import get_data as get_oracle_data


def get_train_predict_data(data_flag, tarin_input_dir=None, train_dt_name=None, predict_input_dir=None, predict_dt_name=None,
             train_data_sql=None, predict_data_sql=None):
    train_data = pd.DataFrame()
    predict_data = pd.DataFrame()
    if data_flag == "csv":
        train_data = pd.read_csv(f'{tarin_input_dir}/{train_dt_name}.csv', encoding='gbk')
        predict_data = pd.read_csv(f'{predict_input_dir}/{predict_dt_name}.csv', encoding='gbk')
        logging.info("从csv格式文件获取到数据！")
    elif data_flag == "oracle":
        train_data = get_oracle_data(train_data_sql)
        predict_data = get_oracle_data(predict_data_sql)
        # logging.info("从Oracle数据库获取到数据！")
    else:
        logging.warning("未获取到任何数据！")
        pass
    return train_data, predict_data

def get_data(data_flag, input_dir=None, dt_name=None, data_sql=None):
    data = pd.DataFrame()
    if data_flag == "csv":
        data = pd.read_csv(f'{input_dir}/{dt_name}.csv', encoding='gbk')
        logging.info("从csv格式文件获取到数据！")
    elif data_flag == "oracle":
        data = get_oracle_data(data_sql)
        # logging.info("从Oracle数据库获取到数据！")
    else:
        logging.warning("未获取到任何数据！")
        pass
    return data