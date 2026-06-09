from blocks.SoloPartFlight.SoloFlightAdvicePrice import SoloFltAdvicePrice
from blocks.SoloPartFlight.SoloFlightUniversalModule import true_price_up_down

def solo_part_flight_advice_price(config):
    # v2 开关：设置 config.use_v2_predictor=True 启用新版预测器
    if getattr(config, 'use_v2_predictor', False):
        from blocks.SoloPartFlight.SoloFlightNumberIncreaseKNN_v2 import solo_knn_est_run
    else:
        from blocks.SoloPartFlight.SoloFlightNumberIncreaseKNN import solo_knn_est_run

    # 1 利用KNN算法计算剩余销售期内的人数增量情况
    result_data = solo_knn_est_run(config)
    # 2 进行价格扩展并给出航班建议价格
    result_data = SoloFltAdvicePrice(config, result_data).result_data
    # 3 修正建议价格
    result_data = true_price_up_down(config, 'SOLO_PART', result_data)
    return result_data