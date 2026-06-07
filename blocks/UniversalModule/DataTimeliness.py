"""
【程序目的】
实现对数据及时性的检查。
"""
import logging

import numpy as np

from config.config import get_argparse
from common.oracle.database_oracle import delete_predict_data
from data_provider.data_acquisition import get_data
from common.get_logger import get_logger
from common.send_mail import send_mail

def data_timeliness(args):
    get_logger()

    # 获取时间戳（获取程序运行时间、小时、分钟等信息）
    date = args.file_create_date
    hour = str(args.file_create_hour).zfill(2)
    minute = args.file_create_minute

    # 采集数据最新批次
    now_catch_time = get_data("oracle", data_sql='SELECT MAX(CATCH_TIME) FROM KD_FUTURE_TMP_SJC_NEW').values[0][0]
    # 采集数据最新数据创建时间
    now_create_time = get_data("oracle", data_sql='SELECT MAX(UP_DATE) FROM KD_FUTURE_TMP_SJC_NEW').values[0][0]
    # 预测结果表（最新）
    result_data = get_data("oracle", data_sql='SELECT * FROM TMP_MAX_RETURN_ADVICE_PRICE')
    # 当处于夜间托管时间范围 | 表中数据创建时间已过去11分钟 | 预测结果表存在数据 时，删除预测结果表数据
    if (args.file_create_hour >= 18 or args.file_create_hour <= 7) and (
            ((np.datetime64(args.create_time) - np.datetime64(now_create_time)) / np.timedelta64(1, 'm')).astype(
                    int) >= 11) and len(result_data) > 0:
        delete_predict_data(f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE")
        logging.warning('预测结果数据过期，将结果表数据删除，防止误执行。')

    # 获取当前时间应该有的最新数据时间戳标记（一旦最小采集频率有变化，就要修改此处代码）
    minute = '00'
    # if minute >= 0 and minute < 30:
    #     minute = '00'
    # elif minute >= 30 and minute <= 59:
    #     minute = '30'
    # else:
    #     pass

    # 根据当前应该有的时间戳看采集源头表是否数据，如果有，则执行，否则跳过。
    catch_time = f'{hour}:{minute}'
    tmp_time_flag = get_data("oracle",
                             data_sql=f"SELECT * FROM TB_DAQ_LOG WHERE CATCH_DATE=DATE'{date}' AND CATCH_TIME='{catch_time}' AND TITLE='已整合'")
    if tmp_time_flag.empty or now_catch_time == catch_time:
        return False
    else:
        return True

def catch_data_timeliness():
    args = get_argparse()
    get_logger()
    # 采集数据最新数据创建时间
    now_create_time = get_data("oracle", data_sql='SELECT MAX(UP_DATE) FROM KD_FUTURE_TMP_SJC_NEW').values[0][0]
    # 获取当前系统时间和数据采集时间
    sysdate = np.datetime64(args.create_time)
    catch_time = np.datetime64(now_create_time)

    # 创建一个timedelta64对象
    td = sysdate - catch_time  # 719581280000纳秒，约等于2小时27分钟
    # 将timedelta64转换为秒
    td_in_seconds = td / np.timedelta64(1, 's')
    # 将秒转换为分钟
    td_in_minutes = td_in_seconds // 60
    # 如果间隔时间超过100分钟，返回真，否则返回假
    if td_in_minutes > 150:
        logging.warning('动态定价采集程序过期超过150分钟，后续数据缺失，请注意检查相关程序！！！')
        send_mail('【动态定价采集程序报错】', f'动态定价采集程序过期超过150分钟！请及时前往云桌面检查。')
    else:
        pass
