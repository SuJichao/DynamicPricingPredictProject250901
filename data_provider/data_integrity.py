"""
【程序目的】
实现对数据完整性的检查，主要检查数据是否存在空值，并返回存在空值的列。
"""

def get_null_columns(df):
    null_columns = df.isnull().any()
    null_columns = null_columns[null_columns].index

    if null_columns.empty:
        return True
    else:
        print(null_columns)
        return False
