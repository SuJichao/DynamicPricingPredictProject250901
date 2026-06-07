"""
【程序目的】
存储程序文件中的各类参数信息。
"""
import argparse
import datetime
import os
def get_argparse():
    parser = argparse.ArgumentParser(description='DynamicPricingPredictProject')

    # 1 basic config
    parser.add_argument('--data_source', type=str, default='oracle', help='data source')
    parser.add_argument('--version_number', type=str, default='6.0.0', help='version number')
    parser.add_argument('--root_dir', type=str, default='/DynamicPricingPredictProject250901', help='root directory')
    parser.add_argument('--multi_process_flag', type=bool, default=False, help='multi process flag')
    parser.add_argument('--whether_execute_process', type=bool, default=True, help='whether to execute the process')
    parser.add_argument('--file_create_date', type=str, default=datetime.datetime.now().strftime('%Y-%m-%d'),
                        help='file create date')
    parser.add_argument('--file_create_hour', type=int, default=int(datetime.datetime.now().strftime('%H')),
                        help='file create hour')
    parser.add_argument('--file_create_minute', type=int, default=int(datetime.datetime.now().strftime('%M')),
                        help='file create minute')
    parser.add_argument('--create_time', type=str, default=datetime.datetime.now(),
                        help='file create time')
    parser.add_argument('--weekday', type=str, default=datetime.datetime.now().isoweekday(),
                        help='weekday')
    parser.add_argument('--small_flight_bottom_discount', type=float, default=0.1,
                        help='small flight bottom discount')
    parser.add_argument('--big_flight_bottom_discount', type=float, default=0.2,
                        help='big flight bottom discount')
    parser.add_argument('--solo_flight_bottom_discount', type=float, default=0.1,
                        help='solo flight bottom discount')
    parser.add_argument('--flt_list', type=str, default='DP_FLT_LIST', help='flt list')
    parser.add_argument('--rb_ota_data', type=str,
                        default="SELECT FLT_DATE,CATCH_DATE,DEP||ARR AS FLT_SEGMENT,CATCH_DIF AS EX_DIF,TO_NUMBER(SUBSTR(CATCH_TIME,1,2)) AS TIME_PT,CARRIER AS AIR_CODE,CARRIER||FLT_NO AS FLT_NO,PRICE_OTA AS AVG_FARE_SK,PRICE AS FULL_PRICE FROM KD_FUTURE_TMP_SJC_NEW WHERE CARRIER IN ('MF','NS','RY')",
                        help='rb ota data')
    parser.add_argument('--flight_price_bottom', type=str, default='SELECT * FROM MXZ_DP_PRICE_CONTRAL',
                        help='flight price bottom data')

    # 1 small part flight
    parser.add_argument('--FlightCapControlKnn', type=str,
                        default='【FlightCapControlKnn】')
    parser.add_argument('--FlightCapControlKnnTrainData', type=str,
                        default='HIS_FUT_FLT_LIST')
    parser.add_argument('--FlightCapControlKnnPredictData', type=str,
                        default='TMP_FUT_EST_INPUT')
    parser.add_argument('--FlightCapControlKnnPredictList', type=str,
                        default='TMP_FLT_LIST')
    parser.add_argument('--SmallPartMaxMInPrice', type=str,
                        default='SELECT FLT_DATE,AIR_CODE,FLT_NO,FLT_SEGMENT,MAX_PRICE,MIN_PRICE FROM SMALL_PART_MAX_MIN_PRICE')
    parser.add_argument('--PredictLabel', type=str,
                        default='KZL_ZL_MF,KZL_ZL_IND')
    parser.add_argument('--FlightCapControlKnnLog', type=str,
                        default='FlightCapControlKnnLog')
    parser.add_argument('--FlightCapControlKnnParameter', type=str,
                        default='date_sin,date_cos,HOL_LAST,TIME_PT,DEP_HOUR,CAP,BKD_SK,PJPJ,chunjie_sin')
    parser.add_argument('--FlightCapControlKnnOutput', type=str,
                        default='CATCH_DATE,FLT_DATE,EX_DIF,TIME_PT,AIR_CODE,FLT_NO,'
                                'FLT_SEGMENT,FLT_ROUTE,DEP_HOUR,DEP_MINUTE,CAP,DISCAP,'
                                'BKD_LEFT,BKD,GRS,BKD_SK,PJPJ,SZS_ZL,HXJG_FLAG,'
                                'DOW,CAP_FINAL,CAP_IND_FINAL,KZL_ZL_MF,'
                                'KZL_ZL_IND,PRICE,ARTIFICIAL_CAP_LEFT,CAP_LEFT')


    # 5 revenue management agreement K
    parser.add_argument('--rm_agreement_K', type=str,
                        default='SELECT * FROM RM_AGREEMENT_FLT_K', help='rm agreement K')
    parser.add_argument('--sales_ratio', type=str, default='SELECT * FROM RM_MXZ_MF_CAP_CTRL', help='sales ratio sql')

    # 6 Solo Flight Advice Price
    parser.add_argument('--solo_flight_advice_price_train_table', type=str,
                        default='TMP_DP_SOLO_PROB_PRED2_3', help='solo flight advice price train table')
    parser.add_argument('--solo_flight_advice_price_predict_table', type=str,
                        default='TMP_SOLO_RB_DATA', help='solo flight advice price predict table')
    parser.add_argument('--solo_flight_advice_price_knn_predict_list', type=str,
                        default='SOLO_FLT_KNN_LIST', help='solo flight predict list')
    parser.add_argument('--solo_x_label_col', type=str,
                        default='YEAR,date_sin,date_cos,HOL_LAST,TIME_PT,chunjie_sin', help='X label_col')#month_sin,month_cos,day_sin,day_cos,
    parser.add_argument('--solo_y_label_col', type=str, default='SRS_ZL_DETR_LEFT,PJPJ_MIN,EX_DIF,PJPJ_FINAL', help='Y label_col')#SRS_ZL_DETR_LEFT,K,B
    parser.add_argument('--log_func_result', type=str, default='LOG_FUC_RESULT', help='log func result table')
    parser.add_argument('--solo_bottom_price', type=str, default='TMP_SOLO_BOTTOM_PRICE', help='solo bottom price')
    parser.add_argument('--solo_previous_price', type=str, default='TMP_SOLO_FLT_OLD_DATA', help='solo flight previous price')
    parser.add_argument('--chunyun_zfx_list', type=str, default='JKXZ_CY_ZFX_DATE_HC', help='chunyun zfx list')

    args = parser.parse_args()
    return args


# 日志文件路径（所有模块统一从此读取）
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'app.log')

# ===== 邮件配置 =====
EMAIL_USERNAME = 'ps_rms@xiamenair.com'
EMAIL_PASSWORD = '!111qqqq08'
EMAIL_SMTP_HOST = 'mail.xiamenair.com.cn'
EMAIL_POP_HOST = 'mail.xiamenair.com.cn'
EMAIL_IMAP_HOST = 'mail.xiamenair.com.cn'
EMAIL_SENT_KEEP_DAYS = 30      # 发件箱邮件默认保留天数
EMAIL_SENT_KEEP_MAX = 200      # 发件箱邮件默认保留最大封数