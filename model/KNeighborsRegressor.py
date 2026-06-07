"""
【程序目的】
实现KNN机器学习算法的核心功能。
"""

import numpy as np
import pandas as pd
import statistics
# 欧氏距离
def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

# KNN核心类
class KnnRegressorFunction(object):
    # 定义初始化方法,初始化KNN需要的参数
    def __init__(self, n_neighbors=3, dist_func=distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 训练模型方法
    def fit(self, x, y):
        # 将x，y传进来即可
        self.x = x
        self.y = y

    # 模型预测方法
    def predict(self, x, y):
        # 初始化预测分类数组
        # y = np.zeros((y.shape[0]), dtype=self.y.dtype)
        if isinstance(y, pd.DataFrame):
            y = np.zeros((y.shape[0], y.shape[1]), dtype=self.y.dtype)
        else:
            y = np.zeros((y.shape[0]), dtype=self.y.dtype)

        # 遍历输入的x数据点，取出每一个数据点的i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有的训练数据计算距离
            distances = self.dist_func(self.x, x_test)

            # 得到的距离按照由近到远排序
            nn_index = np.argsort(distances)

            # 选取最近的k个点
            nn_index = nn_index[:self.n_neighbors]
            nn_y = self.y[nn_index[:self.n_neighbors]]#.ravel()

            # 计算选取样本的均值
            y[i] = np.mean(nn_y, axis=0)

        return y, nn_index

# KNN独飞航线类
class SoloFltKnnRegressorFunction(KnnRegressorFunction):
    def __init__(self, n_neighbors=3, dist_func=distance):
        # 调用父类的实例化方法
        KnnRegressorFunction.__init__(self, n_neighbors=3, dist_func=distance)
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 预测结果取中位数
    def predict(self, x, y):
        # 初始化预测分类数组
        # y = np.zeros((y.shape[0]), dtype=self.y.dtype)
        if isinstance(y, pd.DataFrame):
            y = np.zeros((y.shape[0], y.shape[1]), dtype=self.y.dtype)
        else:
            y = np.zeros((y.shape[0]), dtype=self.y.dtype)

        # 遍历输入的x数据点，取出每一个数据点的i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有的训练数据计算距离
            distances = self.dist_func(self.x, x_test)

            # 得到的距离按照由近到远排序
            nn_index = np.argsort(distances)

            # 选取最近的k个点
            nn_index = nn_index[:self.n_neighbors]
            nn_y = self.y[nn_index[:self.n_neighbors]]

            # y[0:, 0] = np.minimum(statistics.median(nn_y[:, 0]), np.mean(nn_y[:, 0], axis=0))
            # y[0:, 1] = np.minimum(statistics.median(nn_y[:, 1]), np.mean(nn_y[:, 1], axis=0))
            # 当D8以后价格和人数都取MAX,0时人数，1是价格，2是EX_DIF
            # y[0:, 0] = np.where(np.mean(nn_y[:, 2], axis=0) >= 8,
            #                     # np.max(nn_y[:, 0], axis=0),
            #                     np.maximum(statistics.median(nn_y[:, 0]), np.mean(nn_y[:, 0], axis=0)),
            #                     np.maximum(statistics.median(nn_y[:, 0]), np.mean(nn_y[:, 0], axis=0))  # np.max(nn_y[:, 0], axis=0)
            # )
            y[0:, 0] = np.maximum(statistics.median(nn_y[:, 0]), np.mean(nn_y[:, 0], axis=0)) # np.mean(nn_y[:, 0], axis=0)
            y[0:, 1] = np.where(np.mean(nn_y[:, 2], axis=0) >= 29,
                                np.max(nn_y[:, 1], axis=0),
                                np.mean(nn_y[:, 1], axis=0))
            y[0:, 2] = np.mean(nn_y[:, 2], axis=0)
            y[0:, 3] = np.mean(nn_y[:, 3], axis=0)

        return y, nn_index


class SmallFltKnnRegressorFunction(KnnRegressorFunction):
    def __init__(self, n_neighbors=3, dist_func=distance):
        # 调用父类的实例化方法
        KnnRegressorFunction.__init__(self, n_neighbors=3, dist_func=distance)
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 预测结果取中位数
    def predict(self, x, y):
        # 初始化预测分类数组
        # y = np.zeros((y.shape[0]), dtype=self.y.dtype)
        if isinstance(y, pd.DataFrame):
            y = np.zeros((y.shape[0], y.shape[1]), dtype=self.y.dtype)
        else:
            y = np.zeros((y.shape[0]), dtype=self.y.dtype)

        # 遍历输入的x数据点，取出每一个数据点的i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有的训练数据计算距离
            distances = self.dist_func(self.x, x_test)

            # 得到的距离按照由近到远排序
            nn_index = np.argsort(distances)

            # 选取最近的k个点
            nn_index = nn_index[:self.n_neighbors]
            nn_y = self.y[nn_index[:self.n_neighbors]]

            # 当D8以后价格和人数都取MAX,0是航班客座率，1是行业客座率
            y[0:, 0] = np.mean(nn_y[:, 0], axis=0)
            y[0:, 1] = np.mean(nn_y[:, 1], axis=0)

        return y, nn_index