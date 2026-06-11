"""
【程序目的】
KNN 训练数据获取的规则链模块（v2 重构版）。
用责任链模式替代 SmallPartFlightCapCtrl.get_data() 和
SoloFlightNumberIncreaseKNN.get_data() 中合计约 1500 行的 if-else 树。

每条规则链对应一种场景，内部包含 2-3 级回退：
  Level 0 → 严格匹配 → Level 1 → 解除限制 → Level 2 → 最终回退

使用方式：
  from blocks.UniversalModule.DataFetchRules import (
      FetchContext, fetch_train_data, fetch_predict_data,
      SMALL_FLT_FETCH_CONTEXT, SOLO_FLT_FETCH_CONTEXT
  )
"""

import logging
import os
import sys

# 确保项目根目录在 sys.path 中（支持直接运行此文件）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from common.database_oracle import get_data


# ============================================================
# FetchContext：封装小份额/独飞的所有SQL差异
# ============================================================
class FetchContext:
    """
    封装小份额 (SMALL_PART) 和独飞 (SOLO_PART) 在 SQL 构建时的差异。

    关键差异点：
      segment(t)         航段字段值        t['DEP']+t['ARR'] / t['FLT_SEGMENT']
      time_pt(t)         TIME_PT 条件     两套不同的 EX_DIF→TIME_PT 映射
      exists_segment     EXISTS 中段引用   B.DEP||B.ARR / B.FLT_SEGMENT
      holiday_pred_extra 节假日预测额外条件 小份额有 FLT_NO 过滤
      normal_levels      普通日回退级数     3(小份额) / 2(独飞)
    """
    def __init__(self, train_table, predict_table, list_table,
                 segment_fn, time_pt_fn,
                 exists_segment, holiday_pred_extra_fn,
                 log_label_fn, normal_levels=2):
        self.train_table = train_table
        self.predict_table = predict_table
        self.list_table = list_table
        self._segment = segment_fn
        self._time_pt = time_pt_fn
        self.exists_segment = exists_segment    # e.g. "B.DEP||B.ARR" or "B.FLT_SEGMENT"
        self._holiday_pred_extra = holiday_pred_extra_fn
        self._log_label = log_label_fn
        self.normal_levels = normal_levels

    def seg(self, t):
        return self._segment(t)

    def tp(self, t):
        return self._time_pt(t)

    def hpe(self, t):
        return self._holiday_pred_extra(t)

    def label(self, t):
        return self._log_label(t)


# ============================================================
# 小份额上下文
# ============================================================
def _small_seg(t):
    return f"{t['DEP']}{t['ARR']}"

def _small_tp(t):
    return f"((EX_DIF>7 AND TIME_PT=0) OR (EX_DIF<=7 AND TIME_PT={t['TIME_PT']}))"

_SMALL_EXISTS_SEG = "B.DEP||B.ARR"

def _small_hpe(t):
    return f"AND FLT_NO='{t['FLT_NO']}'"

def _small_label(t):
    return (f"序号：{t.get('HX','?')}：航段信息：{t.get('DEP','')}{t.get('ARR','')}，"
            f"距离起飞天数{t.get('EX_DIF','?')}，采集时点{t.get('TIME_PT','?')}")

SMALL_FLT_FETCH_CONTEXT = FetchContext(
    train_table=None, predict_table=None, list_table=None,
    segment_fn=_small_seg, time_pt_fn=_small_tp,
    exists_segment=_SMALL_EXISTS_SEG,
    holiday_pred_extra_fn=_small_hpe,
    log_label_fn=_small_label,
    normal_levels=3,
)


# ============================================================
# 独飞上下文
# ============================================================
def _solo_seg(t):
    return t['FLT_SEGMENT']

def _solo_tp(t):
    return f"(EX_DIF=0 OR (EX_DIF!=0 AND TIME_PT=CASE WHEN EX_DIF>7 THEN 1 ELSE {t['TIME_PT']} END))"

_SOLO_EXISTS_SEG = "B.FLT_SEGMENT"

def _solo_hpe(t):
    return ""

def _solo_label(t):
    return (f"航段信息：{t.get('FLT_SEGMENT','?')}，"
            f"距离起飞天数{t.get('EX_DIF','?')}，采集时点{t.get('TIME_PT','?')}")

SOLO_FLT_FETCH_CONTEXT = FetchContext(
    train_table=None, predict_table=None, list_table=None,
    segment_fn=_solo_seg, time_pt_fn=_solo_tp,
    exists_segment=_SOLO_EXISTS_SEG,
    holiday_pred_extra_fn=_solo_hpe,
    log_label_fn=_solo_label,
    normal_levels=2,
)


# ============================================================
# 辅助：月份/季节 SQL 片段
# ============================================================
def _season_clause(t):
    month = t.get('MONTH', 0)
    if 7 <= month <= 8:
        return "AND TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) BETWEEN 7 AND 8"
    elif month < 7 or month > 8:
        return "AND (TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) > 8 OR TO_NUMBER(TO_CHAR(FLT_DATE,'MM')) < 7)"
    return ""


# ============================================================
# 辅助：TIME_PT 比较子句（EXISTS 中通用）
# ============================================================
_EXISTS_TP_CMP = (
    "CASE WHEN A.EX_DIF>7 THEN 1 ELSE A.TIME_PT END"
    " = "
    "CASE WHEN B.EX_DIF>7 THEN 1 ELSE B.TIME_PT END"
)


# ============================================================
# 辅助：节假日字段的 SQL 片段
# ============================================================
def _hol_fields(t):
    """返回 5 个节假日字段的 SQL 条件"""
    return (f"HOL_BEFORE_TWO_DAY={t['HOL_BEFORE_TWO_DAY']} AND "
            f"HOL_BEFORE_ONE_DAY={t['HOL_BEFORE_ONE_DAY']} AND "
            f"HOL_AFTER_ONE_DAY={t['HOL_AFTER_ONE_DAY']} AND "
            f"HOL_AFTER_TWO_DAY={t['HOL_AFTER_TWO_DAY']} AND "
            f"HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']} AND "
            f"HOL_FALG={t['HOL_FALG']} AND "
            f"HOL_LAST={t['HOL_LAST']} AND "
            f"HOLIDAY_RANGE={t['HOLIDAY_RANGE']}")


def _hol_base_where(ctx, t):
    """节假日基础 WHERE（不含额外过滤）"""
    return (f"FLT_SEGMENT='{ctx.seg(t)}' AND "
            f"EX_DIF={t['EX_DIF']} AND "
            f"{ctx.tp(t)} AND "
            f"{_hol_fields(t)}")


# ============================================================
# 辅助：生成 EXISTS 子查询的 3 种回退模式
# ============================================================

def _exists_l1_specific_dow(ctx, t, dow_value):
    """Level 1 回退：特定 DOW + UNION 普通日"""
    return (
        f"SELECT * FROM {ctx.train_table} A "
        f"WHERE EXISTS ("
        f"SELECT * FROM {ctx.list_table} B WHERE B.HX={t['HX']} "
        f"AND A.HOL_FALG=0 "
        f"AND A.DOW={dow_value} "
        f"AND A.EX_DIF=B.EX_DIF "
        f"AND {_EXISTS_TP_CMP} "
        f"AND A.FLT_SEGMENT={ctx.exists_segment} "
        f"AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL"
        f")"
    )


def _exists_l1_decode_dow(ctx, t, decode_expr):
    """Level 1 回退：DECODE 映射 DOW + UNION 普通日"""
    return (
        f"SELECT * FROM {ctx.train_table} A "
        f"WHERE EXISTS ("
        f"SELECT * FROM {ctx.list_table} B WHERE B.HX={t['HX']} "
        f"AND A.HOL_FALG=0 "
        f"AND {decode_expr} "
        f"AND A.EX_DIF=B.EX_DIF "
        f"AND {_EXISTS_TP_CMP} "
        f"AND A.FLT_SEGMENT={ctx.exists_segment} "
        f"AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL"
        f")"
    )


def _exists_l2_part1(ctx, t):
    """Level 2 回退 Part 1：DOW 放宽 + EX_DIF>0"""
    return (
        f"SELECT * FROM {ctx.train_table} A "
        f"WHERE EXISTS ("
        f"SELECT * FROM {ctx.list_table} B WHERE B.HX={t['HX']} "
        f"AND A.HOL_FALG=0 "
        f"AND A.DOW=B.DOW "
        f"AND A.EX_DIF=B.EX_DIF "
        f"AND B.EX_DIF>0 "
        f"AND {_EXISTS_TP_CMP} "
        f"AND A.FLT_SEGMENT={ctx.exists_segment} "
        f"AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL"
        f")"
    )


def _exists_l2_part2(ctx, t):
    """Level 2 回退 Part 2：DOW + TIME_PT 同时放宽"""
    return (
        f"SELECT * FROM {ctx.train_table} A "
        f"WHERE EXISTS ("
        f"SELECT * FROM {ctx.list_table} B WHERE B.HX={t['HX']} "
        f"AND A.HOL_FALG=0 "
        f"AND A.DOW=B.DOW "
        f"AND A.EX_DIF=B.EX_DIF "
        f"AND B.EX_DIF=0 "
        f"AND B.TIME_PT>=A.TIME_PT "
        f"AND A.FLT_SEGMENT={ctx.exists_segment} "
        f"AND A.HOLIDAY_SPRING_FESTIVAL=B.HOLIDAY_SPRING_FESTIVAL"
        f")"
    )


# ============================================================
# 规则节点
# ============================================================
class DataFetchRule:
    """一条数据获取规则。本级查不到数据时自动回退到 fallback。"""

    def __init__(self, name, build_sql, fallback=None):
        self.name = name
        self._build_sql = build_sql   # (ctx, tmp_list) -> sql 字符串
        self.fallback = fallback

    def fetch(self, ctx, tmp_list):
        sql = self._build_sql(ctx, tmp_list)
        data = get_data(sql)
        if len(data) <= 1 and self.fallback:
            logging.info(
                f"【DataFetchRules】[{self.name}] 样本不足({len(data)}条)，"
                f"回退到 {self.fallback.name}")
            return self.fallback.fetch(ctx, tmp_list)
        return data


# ============================================================
# 工厂函数：普通日规则链
# ============================================================
def make_normal_day_chain(ctx):
    """普通日（HOL_FALG=0）数据获取链"""

    level0 = DataFetchRule(
        name="普通日-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT *
            FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND DOW={t['DOW']}
              {f"AND HXJG_FLAG={t['HXJG_FLAG']}" if c.normal_levels >= 3 else ""}
              AND HOL_FALG='{t['HOL_FALG']}'
              {c.hpe(t) if c.normal_levels < 3 else ""}
              {"AND AIR_CODE IN ('MF','NS','RY')" if c.normal_levels < 3 else ""}
              {_season_clause(t)}
        """,
    )

    level1 = DataFetchRule(
        name="普通日-解除1级(去DOW/季节)",
        build_sql=lambda c, t: f"""
            SELECT *
            FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND HOL_FALG='{t['HOL_FALG']}'
              {c.hpe(t) if c.normal_levels < 3 else ""}
              {"AND AIR_CODE IN ('MF','NS','RY')" if c.normal_levels < 3 else ""}
        """,
    )

    if ctx.normal_levels >= 3:
        # 小份额：第2级与第1级相同（保持原行为）
        level2 = DataFetchRule(
            name="普通日-解除2级(与L1同)",
            build_sql=lambda c, t: f"""
                SELECT *
                FROM {c.train_table} A
                WHERE FLT_SEGMENT='{c.seg(t)}'
                  AND EX_DIF={t['EX_DIF']}
                  AND {c.tp(t)}
                  AND HOL_FALG='{t['HOL_FALG']}'
            """,
            fallback=None,
        )
        level1.fallback = level2
    else:
        level1.fallback = None

    level0.fallback = level1
    return level0


# ============================================================
# 工厂函数：节假日 1 天规则链
# ============================================================
def make_holiday_1day_chain(ctx):
    """节假日放假 1 天（HOL_LAST=1, HOLIDAY_SPRING_FESTIVAL=0）"""

    hol_where = lambda c, t: _hol_base_where(c, t)

    level0 = DataFetchRule(
        name="1天假-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
        """,
    )

    level1 = DataFetchRule(
        name="1天假-解除1级(DOW→6+UNION)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
            UNION ALL
            {_exists_l1_specific_dow(c, t, 6)}
        """,
    )

    level2 = DataFetchRule(
        name="1天假-解除2级(全面放宽)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
            UNION ALL
            {_exists_l2_part1(c, t)}
            UNION ALL
            {_exists_l2_part2(c, t)}
        """,
        fallback=None,
    )

    level0.fallback = level1
    level1.fallback = level2
    return level0


# ============================================================
# 工厂函数：节假日 3 天规则链
# ============================================================
def make_holiday_3day_chain(ctx):
    """节假日放假 3 天（HOL_LAST=2, HOLIDAY_SPRING_FESTIVAL=0）"""

    hol_where = lambda c, t: _hol_base_where(c, t)

    decode_3day = (
        "DECODE(B.HOLIDAY_RANGE,"
        "-2,4, -1,5, 1,6, 2,6, 3,7, 4,1, 5,2"
        ")=A.DOW"
    )

    level0 = DataFetchRule(
        name="3天假-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
        """,
    )

    level1 = DataFetchRule(
        name="3天假-解除1级(DECODE+UNION)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
            UNION ALL
            {_exists_l1_decode_dow(c, t, decode_3day)}
        """,
    )

    level2 = DataFetchRule(
        name="3天假-解除2级(全面放宽)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {hol_where(c, t)}
            UNION ALL
            {_exists_l2_part1(c, t)}
            UNION ALL
            {_exists_l2_part2(c, t)}
        """,
        fallback=None,
    )

    level0.fallback = level1
    level1.fallback = level2
    return level0


# ============================================================
# 工厂函数：4天以上节前规则链
# ============================================================
def make_holiday_4plus_pre_chain(ctx):
    """4天以上假期节前（HOL_LAST>=3, HOLIDAY_RANGE<0）"""

    decode_pre = "DECODE(B.HOLIDAY_RANGE,-2,4,-1,5)=A.DOW"

    # Level 0 where: HOL_LAST>=3 AND HOLIDAY_RANGE<0 (不加精确 HOLIDAY_RANGE)
    def _l0_where(c, t):
        return (f"FLT_SEGMENT='{c.seg(t)}' AND "
                f"EX_DIF={t['EX_DIF']} AND "
                f"{c.tp(t)} AND "
                f"HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']} AND "
                f"HOL_FALG={t['HOL_FALG']} AND "
                f"HOL_LAST>=3 AND HOLIDAY_RANGE<0")

    level0 = DataFetchRule(
        name="4天+节前-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
        """,
    )

    level1 = DataFetchRule(
        name="4天+节前-解除1级(DECODE+UNION)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
            UNION ALL
            {_exists_l1_decode_dow(c, t, decode_pre)}
        """,
    )

    level2 = DataFetchRule(
        name="4天+节前-解除2级(全面放宽)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
            UNION ALL
            {_exists_l2_part1(c, t)}
            UNION ALL
            {_exists_l2_part2(c, t)}
        """,
        fallback=None,
    )

    level0.fallback = level1
    level1.fallback = level2
    return level0


# ============================================================
# 工厂函数：4天以上节后规则链
# ============================================================
def make_holiday_4plus_post_chain(ctx):
    """4天以上假期节后（HOL_LAST>=3, HOL_LAST-HOLIDAY_RANGE<0）"""

    decode_post = "DECODE(B.HOLIDAY_RANGE,-2,2,-1,1)=A.DOW"

    def _l0_where(c, t):
        return (f"FLT_SEGMENT='{c.seg(t)}' AND "
                f"EX_DIF={t['EX_DIF']} AND "
                f"{c.tp(t)} AND "
                f"HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']} AND "
                f"HOL_FALG={t['HOL_FALG']} AND "
                f"HOL_LAST>=3 AND HOL_LAST-HOLIDAY_RANGE<0")

    level0 = DataFetchRule(
        name="4天+节后-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
        """,
    )

    level1 = DataFetchRule(
        name="4天+节后-解除1级(DECODE+UNION)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
            UNION ALL
            {_exists_l1_decode_dow(c, t, decode_post)}
        """,
    )

    level2 = DataFetchRule(
        name="4天+节后-解除2级(全面放宽)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE {_l0_where(c, t)}
            UNION ALL
            {_exists_l2_part1(c, t)}
            UNION ALL
            {_exists_l2_part2(c, t)}
        """,
        fallback=None,
    )

    level0.fallback = level1
    level1.fallback = level2
    return level0


# ============================================================
# 工厂函数：4天以上节中规则链
# ============================================================
def make_holiday_4plus_mid_chain(ctx):
    """
    4天以上假期节中（HOL_LAST>=3, 非节前非节后）。
    内部按节日天数分 3 个子场景：
      - 节中第1-2天（HOL_LAST 4-5: RANGE=1; HOL_LAST 6-8: RANGE 1-2）
      - 节中最后1-2天（RANGE 接近 HOL_LAST）
      - 节中其他天
    """

    # 我们给每个子场景构建它自己的规则链，然后在 fetch 时通过条件判断选择。
    # 由于 DataFetchRule 只负责回退，条件分支需要在上层处理。
    # 这里采用组合模式：节中的 3 个子场景共用同一个外部接口，
    # 但内部根据 HOL_LAST 和 HOLIDAY_RANGE 选择不同的 SQL 模板。

    # 为了保持与原代码行为一致，在 build_sql 中内联子场景判断。
    def _mid_where(c, t):
        hol_last = t['HOL_LAST']
        hol_range = t['HOLIDAY_RANGE']

        # 节中第1-2天
        if (4 <= hol_last <= 5 and hol_range == 1) or \
           (6 <= hol_last <= 8 and 1 <= hol_range <= 2):
            return _mid_first_where(c, t, hol_last)
        # 节中最后1-2天
        elif (4 <= hol_last <= 5 and hol_range >= hol_last - 1 and hol_range <= hol_last) or \
             (6 <= hol_last <= 8 and hol_range >= hol_last - 1 and hol_range <= hol_last):
            return _mid_last_where(c, t, hol_last)
        # 节中其他天
        else:
            return _mid_other_where(c, t, hol_last)

    def _mid_extra_clause(t, hol_last):
        """仅用于 Level 1 回退中的特定 DOW 映射"""
        hol_range = t['HOLIDAY_RANGE']
        # 节中第1-2天 → DOW=5；节中最后1-2天 → DOW=7；其他天 → DOW=6
        if (4 <= hol_last <= 5 and hol_range == 1) or \
           (6 <= hol_last <= 8 and 1 <= hol_range <= 2):
            return (5, "A.DOW=5")
        elif (4 <= hol_last <= 5 and hol_range >= hol_last - 1 and hol_range <= hol_last) or \
             (6 <= hol_last <= 8 and hol_range >= hol_last - 1 and hol_range <= hol_last):
            return (7, "A.DOW=7")
        else:
            return (6, "A.DOW=6")

    # 由于 build_sql 是 lambda，节中场景需要动态选择 DOW 值，
    # 所以在 build_sql 内部调用 _mid_extra_clause
    level0 = DataFetchRule(
        name="4天+节中-严格匹配",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
              AND HOL_FALG={t['HOL_FALG']}
              AND {_mid_where(c, t)}
        """,
    )

    level1 = DataFetchRule(
        name="4天+节中-解除1级(特定DOW+UNION)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
              AND HOL_FALG={t['HOL_FALG']}
              AND {_mid_where(c, t)}
            UNION ALL
            {_exists_l1_specific_dow(c, t, _mid_extra_clause(t, t['HOL_LAST'])[0])}
        """,
    )

    level2 = DataFetchRule(
        name="4天+节中-解除2级(全面放宽)",
        build_sql=lambda c, t: f"""
            SELECT * FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
              AND HOL_FALG={t['HOL_FALG']}
              AND {_mid_where(c, t)}
            UNION ALL
            {_exists_l2_part1(c, t)}
            UNION ALL
            {_exists_l2_part2(c, t)}
        """,
        fallback=None,
    )

    level0.fallback = level1
    level1.fallback = level2
    return level0


# ---- 节中 3 个子场景的 WHERE 条件 ----

def _mid_first_where(c, t, hol_last):
    """4-8天假期节中第1-2天"""
    if hol_last == 4:
        return "HOL_LAST=4 AND HOLIDAY_RANGE=1"
    elif hol_last == 5:
        return "HOL_LAST=5 AND HOLIDAY_RANGE=1"
    elif hol_last == 6:
        return "HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 1 AND 2"
    elif hol_last == 7:
        return "HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 1 AND 2"
    else:  # hol_last == 8
        return "HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 1 AND 2"


def _mid_last_where(c, t, hol_last):
    """4-8天假期节中最后1-2天"""
    if hol_last == 4:
        return "HOL_LAST=4 AND HOLIDAY_RANGE=4"
    elif hol_last == 5:
        return "HOL_LAST=5 AND HOLIDAY_RANGE=5"
    elif hol_last == 6:
        return "HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 5 AND 6"
    elif hol_last == 7:
        return "HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 6 AND 7"
    else:  # hol_last == 8
        return "HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 7 AND 8"


def _mid_other_where(c, t, hol_last):
    """4-8天假期节中其他天"""
    if hol_last == 4:
        return "HOL_LAST=4 AND HOLIDAY_RANGE BETWEEN 2 AND 3"
    elif hol_last == 5:
        return "HOL_LAST=5 AND HOLIDAY_RANGE BETWEEN 2 AND 4"
    elif hol_last == 6:
        return "HOL_LAST=6 AND HOLIDAY_RANGE BETWEEN 2 AND 4"
    elif hol_last == 7:
        return "HOL_LAST=7 AND HOLIDAY_RANGE BETWEEN 3 AND 5"
    else:  # hol_last == 8
        return "HOL_LAST=8 AND HOLIDAY_RANGE BETWEEN 3 AND 6"


# ============================================================
# 工厂函数：春节规则链
# ============================================================
def make_spring_festival_chain(ctx):
    """
    春节（HOLIDAY_SPRING_FESTIVAL=1）。
    按 HOLIDAY_RANGE 分 7 个时段，每一级都有不同的放宽范围。
    """

    # 春节的 Level 0 回退方案按 HOLIDAY_RANGE 区间划分
    # 原代码中的春节处理：先按精确 HOLIDAY_RANGE 查，查不到按区间放宽
    def _sf_base_where(c, t):
        return (f"FLT_SEGMENT='{c.seg(t)}' AND "
                f"EX_DIF={t['EX_DIF']} AND "
                f"{c.tp(t)} AND "
                f"HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']} AND "
                f"HOL_FALG={t['HOL_FALG']} AND "
                f"HOLIDAY_RANGE={t['HOLIDAY_RANGE']}")

    # 小份额春节多一个 HOL_LAST 条件
    def _sf_base_where_small(c, t):
        return (_sf_base_where(c, t) +
                f" AND HOL_LAST={t['HOL_LAST']}")

    def _sf_where(c, t):
        if c.normal_levels >= 3:
            return _sf_base_where_small(c, t)
        return _sf_base_where(c, t)

    # 按 HOLIDAY_RANGE 区间放宽（level 0 的各子回退）
    def _sf_range_clause(holiday_range):
        if holiday_range < -14:
            return "HOLIDAY_RANGE<-14"
        elif -14 <= holiday_range <= -8:
            return "HOLIDAY_RANGE>=-14 AND HOLIDAY_RANGE<=-8"
        elif -7 <= holiday_range <= -1:
            return "HOLIDAY_RANGE>=-7 AND HOLIDAY_RANGE<=-1"
        elif 0 <= holiday_range <= 5:
            return "HOLIDAY_RANGE>=0 AND HOLIDAY_RANGE<=5"
        elif 6 <= holiday_range <= 10:
            return "HOLIDAY_RANGE>=6 AND HOLIDAY_RANGE<=10"
        elif 11 <= holiday_range <= 15:
            return "HOLIDAY_RANGE>=11 AND HOLIDAY_RANGE<=15"
        elif holiday_range == 16:
            return "HOLIDAY_RANGE>=15 AND HOLIDAY_RANGE<=17"
        else:  # >16
            return "HOLIDAY_RANGE>16"

    # 小份额春节区间映射（与原代码不同）
    def _sf_range_clause_small(holiday_range):
        if holiday_range < -7:
            return "HOLIDAY_RANGE<-7"
        elif -7 <= holiday_range <= 1:  # 注意，小份额节前含除夕的区间不同
            return "HOLIDAY_RANGE>=-7 AND HOLIDAY_RANGE<=1" if holiday_range <= -1 else "HOLIDAY_RANGE>=-7 AND HOLIDAY_RANGE<=-1"
        elif 2 <= holiday_range <= 5:
            return "HOLIDAY_RANGE>=2 AND HOLIDAY_RANGE<=5"
        elif 6 <= holiday_range <= 10:
            return "HOLIDAY_RANGE>=6 AND HOLIDAY_RANGE<=10"
        elif 11 <= holiday_range <= 15:
            return "HOLIDAY_RANGE>=11 AND HOLIDAY_RANGE<=15"
        elif holiday_range == 16:
            return "HOLIDAY_RANGE>=15 AND HOLIDAY_RANGE<=17"
        else:
            return "HOLIDAY_RANGE>16"

    # 春节 Level 0 本身就有多级回退（按区间），但在原代码中是通过 elif len<1 串联的。
    # 这里我们构建一个更扁平的链条：

    # 构建春节的子规则链：精确匹配 → 区间放宽 → 二次放宽(节前/节后合并)
    # 由于春节逻辑极其复杂，为保持与原行为一致，在 build_sql 中直接复刻原逻辑

    # 采用一个特殊的春节规则：在 build_sql 中按 HOLIDAY_RANGE 动态选择区间
    def _build_sf_level0(c, t):
        hr = t['HOLIDAY_RANGE']
        return f"""
            SELECT * FROM {c.train_table} A
            WHERE {_sf_where(c, t)}
        """

    def _build_sf_fallback1(c, t):
        """第一次回退：按 HOLIDAY_RANGE 区间放宽"""
        hr = t['HOLIDAY_RANGE']
        range_clause = _sf_range_clause(hr)
        if c.normal_levels >= 3:
            # 小份额：更简单的区间逻辑
            range_clause = _sf_range_clause_small(hr)
        return f"""
            SELECT * FROM {c.train_table} A
            WHERE FLT_SEGMENT='{c.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND {c.tp(t)}
              AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
              AND HOL_FALG={t['HOL_FALG']}
              AND {"HOL_LAST=" + str(t['HOL_LAST']) + " AND " if c.normal_levels >= 3 else ""}{range_clause}
        """

    def _build_sf_fallback2(c, t):
        """第二次回退：合并节前/节后所有"""
        hr = t['HOLIDAY_RANGE']
        if c.normal_levels >= 3:
            # 小份额
            if hr <= -1:
                range_clause = "HOLIDAY_RANGE<=-1"
            else:  # hr >= 6
                range_clause = "HOLIDAY_RANGE>=6"
            return f"""
                SELECT * FROM {c.train_table} A
                WHERE FLT_SEGMENT='{c.seg(t)}'
                  AND EX_DIF={t['EX_DIF']}
                  AND {c.tp(t)}
                  AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
                  AND HOL_FALG={t['HOL_FALG']}
                  AND {range_clause}
            """
        else:
            # 独飞
            if hr <= -1:
                range_clause = "HOLIDAY_RANGE<=-1"
            elif hr >= 6:
                range_clause = "HOLIDAY_RANGE>=6"
            else:
                range_clause = "1=1"  # 不会到这里(节中不回退到 merge)
            return f"""
                SELECT * FROM {c.train_table} A
                WHERE FLT_SEGMENT='{c.seg(t)}'
                  AND EX_DIF={t['EX_DIF']}
                  AND {c.tp(t)}
                  AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
                  AND HOL_FALG={t['HOL_FALG']}
                  AND {range_clause}
            """

    # 注意春节的回退比较特殊：Level 0 内部就有按区间的回退逻辑，
    # 这里通过多个 DataFetchRule 层叠来实现。
    # 先做精确匹配，再做区间放宽，再做节前/节后合并。
    level0 = DataFetchRule(
        name="春节-精确匹配",
        build_sql=_build_sf_level0,
    )

    level0b = DataFetchRule(
        name="春节-区间放宽",
        build_sql=_build_sf_fallback1,
    )

    level0c = DataFetchRule(
        name="春节-二次放宽(节前/节后合并)",
        build_sql=_build_sf_fallback2,
        fallback=None,
    )

    # 链接：精确 → 区间 → 合并
    # 注意：原代码中区间回退只对 len<1 触发，
    # 但第一次如果 len=0(不是<=1)也会触发，
    # 所以这里按 DataFetchRule 的标准 (len<=1) 来回退是合理的。
    level0.fallback = level0b
    level0b.fallback = level0c

    return level0


# ============================================================
# 调度入口
# ============================================================
def _select_holiday_chain(ctx, tmp_list):
    """根据 tmp_list 的节假日特征选择对应的规则链"""
    t = tmp_list
    hol_last = t.get('HOL_LAST', 0)

    if t.get('HOLIDAY_SPRING_FESTIVAL') == 1:
        return make_spring_festival_chain(ctx)
    if hol_last == 1:
        return make_holiday_1day_chain(ctx)
    if hol_last == 2:
        return make_holiday_3day_chain(ctx)
    if hol_last >= 3:
        holiday_range = t.get('HOLIDAY_RANGE', 0)
        if holiday_range < 0:
            return make_holiday_4plus_pre_chain(ctx)
        if hol_last - holiday_range < 0:
            return make_holiday_4plus_post_chain(ctx)
        return make_holiday_4plus_mid_chain(ctx)
    return None


def fetch_train_data(ctx, tmp_list):
    """
    统一的训练数据获取入口，替代两个类各自的 get_data() 中的训练数据部分。

    参数：
      ctx: FetchContext 实例
      tmp_list: 单条预测列表记录（pd.Series）

    返回：
      pd.DataFrame — 训练数据（可能为空）
    """
    if tmp_list.get('HOL_FALG') == 0:
        chain = make_normal_day_chain(ctx)
    else:
        chain = _select_holiday_chain(ctx, tmp_list)
        if chain is None:
            logging.warning(
                f"【DataFetchRules】未找到匹配的节假日规则链: {ctx.label(tmp_list)}")
            return pd.DataFrame()

    return chain.fetch(ctx, tmp_list)


def fetch_predict_data(ctx, tmp_list):
    """
    统一的待预测数据获取入口。

    参数：
      ctx: FetchContext 实例
      tmp_list: 单条预测列表记录（pd.Series）

    返回：
      pd.DataFrame — 待预测数据
    """
    t = tmp_list
    if t.get('HOL_FALG') == 0:
        sql = f"""
            SELECT *
            FROM {ctx.predict_table} A
            WHERE FLT_SEGMENT='{ctx.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND TIME_PT={t['TIME_PT']}
              AND DOW={t['DOW']}
              {ctx.hpe(t)}
              {"AND HOL_FALG=" + str(t['HOL_FALG']) if ctx.normal_levels < 3 else ""}
        """
    else:
        sql = f"""
            SELECT *
            FROM {ctx.predict_table} A
            WHERE FLT_SEGMENT='{ctx.seg(t)}'
              AND EX_DIF={t['EX_DIF']}
              AND TIME_PT={t['TIME_PT']}
              AND DOW={t['DOW']}
              {ctx.hpe(t)}
              AND HOL_BEFORE_TWO_DAY={t['HOL_BEFORE_TWO_DAY']}
              AND HOL_BEFORE_ONE_DAY={t['HOL_BEFORE_ONE_DAY']}
              AND HOL_AFTER_ONE_DAY={t['HOL_AFTER_ONE_DAY']}
              AND HOL_AFTER_TWO_DAY={t['HOL_AFTER_TWO_DAY']}
              AND HOLIDAY_SPRING_FESTIVAL={t['HOLIDAY_SPRING_FESTIVAL']}
              AND HOL_FALG={t['HOL_FALG']}
              AND HOL_LAST={t['HOL_LAST']}
              AND HOLIDAY_RANGE={t['HOLIDAY_RANGE']}
        """
    return get_data(sql)
