# 辅助主程序的相关子模块
import datetime
import logging
import multiprocessing as mp
import schedule
import signal
import sys
import time

from config.config import get_argparse
from config.pricing_constants import (
    TIMING_MIN_EXECUTION_SECONDS,
    TIMING_SLEEP_SECONDS,
    TIMING_MAX_EXECUTION_SECONDS,
)
from config.db_tables import FLT_LIST_TABLE
from common.get_logger import get_logger
from common.database_oracle import callproc, delete_data
from common.send_mail import send_mail, clean_sent_mails
from blocks.UniversalModule.DataStorage import data_storage, delete_deduplication_data
from blocks.UniversalModule.DataTemporaryProcessing import advice_price_output
from blocks.UniversalModule.DataTimeliness import data_timeliness, catch_data_timeliness
from blocks.UniversalModule.GetPredictData import GetPredictData


# 主要业务场景
from blocks.SmallPartFlight.SmallPartFlightAdvicePrice import small_part_flight_advice_price
from blocks.SoloPartFlight.SoloPartFlightAdvicePrice import solo_part_flight_advice_price

# 航班类型 → 预测执行函数 dispatch 表
_FLIGHT_TYPE_HANDLERS = {
    'SMALL_PART': small_part_flight_advice_price,
    'SOLO_PART': solo_part_flight_advice_price,
}

def _process_flight_type(args, flt_type, data_set):
    """验证数据完整性，执行预测，写入存储。

    Args:
        args: argparse 命名空间
        flt_type: 航班类型（'SMALL_PART' / 'SOLO_PART'）
        data_set: GetPredictData 返回的字典，按 flt_type 索引
    """
    predict_func = _FLIGHT_TYPE_HANDLERS.get(flt_type)
    if predict_func is None:
        logging.warning(f"未知航线类型 {flt_type}，跳过。")
        return

    if data_set[flt_type]['predict_data'].empty:
        delete_data(
            f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE WHERE FLT_SEGMENT IN "
            f"(SELECT FLT_SEGMENT FROM {FLT_LIST_TABLE} WHERE FLT_TYPE='{flt_type}')")
        logging.warning(f"{flt_type}航线待预测数据缺失，程序跳过不予执行！")
    else:
        result_data = predict_func(args)
        data_storage(args, flt_type, result_data)
        logging.info(f'==============={flt_type}航班价格建议完成！===============')

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
            _process_flight_type(args, flt_type, data_set)
        except Exception as e:
            logging.error(e, exc_info=True)
            send_mail('【动态定价程序报错】', f'动态定价程序报错！请及时前往云桌面检查。\n\n错误信息为：{e}')
            delete_data(
                f"DELETE FROM TMP_MAX_RETURN_ADVICE_PRICE WHERE FLT_SEGMENT IN (SELECT FLT_SEGMENT FROM {FLT_LIST_TABLE})")
            logging.warning('航班动态定价托管程序执行失败！')

    # 建议价格数据输出和处理
    try:
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
        if (datetime.datetime.now() - args.create_time).total_seconds() < TIMING_MIN_EXECUTION_SECONDS:
            logging.warning('===进程执行完毕后暂停20s！===')
            time.sleep(TIMING_SLEEP_SECONDS)
            run(args)
        elif (datetime.datetime.now() - args.create_time).total_seconds() > TIMING_MAX_EXECUTION_SECONDS:
            logging.warning('===进程执行时间超过15分钟，自动跳过本次程序执行！===')
            pass
        else:
            run(args)
    else:
        logging.warning('===数据过期，自动跳过本次程序执行！！！===')




def _show_menu():
    """显示交互菜单，返回用户选择。无效输入返回 -1。"""
    prompt = (
        "=====任务列表=====\n"
        "==【1】立即执行动态定价程序（不刷新数据）\n"
        "==【2】立即执行动态定价程序（刷新数据）\n"
        "==【3】定时执行动态定价程序（刷新数据）\n"
        "==【4】清理发件箱\n"
        "==输入序号选择要执行的任务："
    )
    try:
        return int(input(prompt))
    except ValueError:
        return -1


def _execute_manual_run(args, refresh_data=False):
    """执行单次手动定价（菜单 1 和 2 共用）。

    Args:
        args: argparse 命名空间（已创建）
        refresh_data: True 时先调用 RM_DP_FLT_EST_INPUT_CHG 刷新数据
    """
    logging.info(
        f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<【航班动态定价托管】'
        f'{args.version_number} 程序开始！>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if refresh_data:
        callproc('RM_DP_FLT_EST_INPUT_CHG')
    run(args)
    delete_deduplication_data()


def _start_scheduler_loop():
    """注册定时任务并进入 schedule 轮询循环。

    注册三个定时任务：
      - 每分钟 :00 执行 timeliness_assessment
      - 每小时 :00 执行 catch_data_timeliness
      - 每天 03:00 清理发件箱
    支持 SIGTERM / SIGINT 优雅退出。
    """
    schedule.every(1).minutes.at(":00").do(timeliness_assessment)
    schedule.every(1).hour.at(":00").do(catch_data_timeliness)
    schedule.every().day.at("03:00").do(clean_sent_mails)

    shutdown_flag = [False]

    def _on_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logging.info(f"收到信号 {sig_name}，正在优雅退出...")
        shutdown_flag[0] = True

    # Windows 只支持 SIGINT / SIGBREAK；类 Unix 支持 SIGINT / SIGTERM
    _signals = [signal.SIGINT]
    if sys.platform.startswith('win'):
        _signals.append(getattr(signal, 'SIGBREAK', None))
    else:
        _signals.append(signal.SIGTERM)
    for sig in filter(None, _signals):
        try:
            signal.signal(sig, _on_shutdown)
        except (ValueError, OSError):
            pass

    while not shutdown_flag[0]:
        schedule.run_pending()
        time.sleep(1)

    logging.info("调度循环已退出。")


if __name__ == '__main__':
    mp.freeze_support()
    choice = _show_menu()

    if choice in (1, 2):
        args = get_argparse()
        get_logger()
        _execute_manual_run(args, refresh_data=(choice == 2))
        _start_scheduler_loop()
    elif choice == 3:
        _start_scheduler_loop()
    elif choice == 4:
        clean_sent_mails()
    else:
        logging.warning(f"无效的菜单选项: {choice}")