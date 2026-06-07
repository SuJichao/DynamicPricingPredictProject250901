# 辅助主程序的相关子模块
import datetime
import logging
import multiprocessing as mp
import schedule
import sys
import time

from config.config import get_argparse
from common.get_logger import get_logger
from common.oracle.database_oracle import callproc, delete_predict_data
from common.send_mail import send_mail, clean_sent_mails
from blocks.UniversalModule.DataStorage import data_storage, delete_deduplication_data
from blocks.UniversalModule.DataTemporaryProcessing import advice_price_output
from blocks.UniversalModule.DataTimeliness import timeliness_assessment, catch_data_timeliness
from blocks.UniversalModule.GetPredictData import GetPredictData


# 主要业务场景
from blocks.SmallPartFlight.SmallPartFlightAdvicePrice import small_part_flight_advice_price
from blocks.SoloPartFlight.SoloPartFlightAdvicePrice import solo_part_flight_advice_price

def run(args):
    # Windows 特殊设置
    if sys.platform.startswith('win'):
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)
    # 1 获取待预测数据
    data_set, flt_list = GetPredictData(args)
    # 2 根据不同类型的航班进行处理
    for tmp_flt_list in flt_list:
        flt_type = tmp_flt_list[0]
        try:
            # 小份额航线
            if flt_type == 'SMALL_PART':
                # pass
                # 1 数据校验（校验成功后继续执行预测操作，否则将所属航班类型的数据进行清除）
                if data_set[flt_type]['predict_data'].empty:
                    delete_predict_data(
                        f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {args.flt_list} WHERE FLT_TYPE='{flt_type}')")
                    logging.warning(f"{flt_type}航线待预测数据缺失，程序跳过不予执行！")
                # 2 执行航班预测程序
                else:
                    result_data = small_part_flight_advice_price(args)
                    data_storage(args, flt_type, result_data)
                    logging.info(f'==============={flt_type}航班价格建议完成！===============')
            # 协议航线
            elif flt_type == 'BIG_PART':
                pass
            # 独飞航线
            elif flt_type == 'SOLO_PART':
                # 1 数据校验（校验成功后继续执行预测操作，否则将所属航班类型的数据进行清除）
                if data_set[flt_type]['predict_data'].empty:
                    delete_predict_data(
                        f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {args.flt_list} WHERE FLT_TYPE='{flt_type}')")
                    logging.warning(f"{flt_type}航线待预测数据缺失，程序跳过不予执行！")
                # 2 执行航班预测程序
                else:
                    result_data = solo_part_flight_advice_price(args)
                    data_storage(args, flt_type, result_data)
                    logging.info(f'==============={flt_type}航班价格建议完成！===============')
            else:
                pass

        except Exception as e:
            logging.error(e, exc_info=True)
            send_mail('【动态定价程序报错】', f'动态定价程序报错！请及时前往云桌面检查。\n\n错误信息为：{e}')
            delete_predict_data(
                f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {args.flt_list})")
            logging.warning('航班动态定价托管程序执行失败！')

    # 建议价格数据输出和处理
    try:
        pass
        advice_price_output()
    except Exception as e:
        logging.error(e, exc_info=True)
        send_mail('【动态定价程序报错】', f'动态定价程序报错！请及时前往云桌面检查。\n\n错误信息为：{e}')

def timeliness_assessment():
    # Windows 特殊设置
    if sys.platform.startswith('win'):
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)
    args = get_argparse()
    get_logger()
    # 判断为真，说明最新批次的数据已经到位，否则不予执行
    # 剔除6点的执行时点
    # if args.file_create_hour != 6:
    if data_timeliness(args):
        logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<【航班动态定价托管】{args.version_number} 程序开始！>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        try:
            callproc('RM_DP_FLT_EST_INPUT_CHG')
        except Exception as e:
            logging.error(e, exc_info=True)
            send_mail('【动态定价程序报错】', f'动态定价程序报错！请及时前往云桌面检查。\n\n错误信息为：{e}')

        # 进程执行时间不超过5s，则判定进程未执行完毕，暂停20s后继续执行后续程序
        if (datetime.datetime.now() - args.create_time).total_seconds() < 5:
            logging.warning('===进程执行完毕后暂停20s！===')
            time.sleep(20)
            run(args)
        elif (datetime.datetime.now() - args.create_time).total_seconds() > 901:
            logging.warning('===进程执行时间超过15分钟，自动跳过本次程序执行！===')
            pass
        else:
            run(args)
    else:
        pass
        # logging.warning('===数据过期，自动跳过本次程序执行！！！===')
    # else:
    #     pass




if __name__ == '__main__':
    mp.freeze_support()
    judge_num = int(input("=====任务列表=====\n==【1】立即执行动态定价程序（不刷新数据）\n==【2】立即执行动态定价程序（刷新数据）\n==【3】定时执行动态定价程序（刷新数据）\n==【4】清理发件箱\n==输入序号选择要执行的任务："))
    if judge_num == 1:
        # 【手动触发作业任务】
        args = get_argparse()
        get_logger()
        logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<【航班动态定价托管】{args.version_number} 程序开始！>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        run(args)
        delete_deduplication_data()
        # 【定时作业任务】
        schedule.every(1).minutes.at(":00").do(timeliness_assessment)
        schedule.every(1).hour.at(":00").do(catch_data_timeliness)
        schedule.every().day.at("03:00").do(clean_sent_mails)
        while True:
            schedule.run_pending()
            time.sleep(1)

    elif judge_num == 2:
        # 【手动触发作业任务】
        args = get_argparse()
        get_logger()
        logging.info(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<【航班动态定价托管】{args.version_number} 程序开始！>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        callproc('RM_DP_FLT_EST_INPUT_CHG')
        run(args)
        delete_deduplication_data()
        # 【定时作业任务】
        schedule.every(1).minutes.at(":00").do(timeliness_assessment)
        schedule.every(1).hour.at(":00").do(catch_data_timeliness)
        schedule.every().day.at("03:00").do(clean_sent_mails)
        while True:
            schedule.run_pending()
            time.sleep(1)

    elif judge_num == 3:
        # 【定时作业任务】
        schedule.every(1).minutes.at(":00").do(timeliness_assessment)
        schedule.every(1).hour.at(":00").do(catch_data_timeliness)
        schedule.every().day.at("03:00").do(clean_sent_mails)
        while True:
            schedule.run_pending()
            time.sleep(1)

    elif judge_num == 4:
        # 【手动清理发件箱】
        clean_sent_mails()

    else:
        pass