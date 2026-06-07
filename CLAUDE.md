# 航班动态定价托管系统 — 项目说明

## 项目概述
厦门航空动态定价系统。通过 KNN 机器学习 + 业务规则引擎，自动预测航班未来销售情况，给出建议票价。

## 技术栈
- Python 3.7+
- Oracle 数据库 (cx_Oracle 连接池)
- KNN 回归（自实现，欧氏距离）
- PyInstaller 打包 exe
- 常驻进程轮询 + schedule 库

## 项目结构
```
run.py                                # 主入口
config/
  config.py                           # argparse 参数
  pricing_constants.py                # 定价参数常量（魔法数字集中管理）
  db_config.ini                       # Oracle 连接配置
data_provider/
  data_acquisition.py                 # 数据获取（Oracle / CSV）
  data_integrity.py                   # 数据完整性检查
common/
  oracle/database_oracle.py           # Oracle 连接池 + CRUD 封装
  get_logger.py                       # 日志配置
  send_mail.py                        # 邮件报警
  save_excel.py                       # Excel VBA 报表
  request.py                          # 收益管理 API 接口
model/
  KNeighborsRegressor.py              # KNN 回归实现
blocks/
  UniversalModule/                    # 通用模块（数据获取/存储/时效性）
  SmallPartFlight/                    # 小份额航线定价
  SoloPartFlight/                     # 独飞航线定价
```

## 两种航线
| 类型 | 预测目标 | KNN K值 |
|------|---------|--------|
| 小份额 (SMALL_PART) | 厦航客座率(KZL_ZL_MF) + 行业客座率(KZL_ZL_IND) | 3(普通日)/1(节假日) |
| 独飞 (SOLO_PART) | 剩余人数增量 + 最低均价 + 最终均价 | 5(普通日)/1(节假日及春运) |

## 代码约束
### 不改的决策
- **SQL 不抽离** — 保持原位，调试时直接复制 SQL 到 PL/SQL Developer 执行
- **f-string 拼接 SQL** — 可复制性 > 抽象整洁
- **调度方式不动** — 保持常驻进程 + schedule + while True 模式

### 已完成的 Phase 1 改进 (2026-06)
1. 魔法数字 → `config/pricing_constants.py` 命名常量
2. `from xxx import *` → 显式导入
3. `except BaseException` → `except Exception`
4. `DataFrame.append()` → list 收集 + `pd.concat()`

### 待做（优先级由高到低）
5. **Phase 2**：提取 `KNNBasePredictor` 基类（消除约 1500 行重复代码）
6. **Phase 2**：分级解除限制 → 规则链（替代 800 行 if-else）
7. **Phase 3**：定价规则引擎（策略模式替代 np.where 嵌套）
8. **Phase 3**：特征工程管道化
