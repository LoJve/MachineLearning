#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    LinearDemo.py
@Time    :    2019/03/26 23:30:29
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Linear Regressor Demo

步骤：
1、定义特征并配置特征列
2、定义目标列
3、配置LinearRegressor
4、定义输入函数
5、训练模型
6、评估模型
'''

# Here put the import lib

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.data import Dataset
from sklearn import metrics
import math

class LinearRegressorDemo(object):
    def __init__(self):
        self.path = "https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv"
        self.split = ','
        self.feature_name = "total_rooms"
        self.target_name = "median_house_value"
        self.learning_rate = 0.0000001

    def get_dataset(self, path, seq=','):
        dataset = pd.read_csv(path, seq)
        return dataset
    
    def train_dataset(self, dataset, feature_name, target_name):
        # 1、定义特征并配置特征列
        my_feature = dataset[[feature_name]]
        feature_columns = [tf.feature_column.numeric_column(feature_name)]

        # 2、定义目标列
        my_target = dataset[target_name]

        # 3、配置LinearRegressor
        # 使用 GradientDescentOptimizer（它会实现小批量随机梯度下降法 (SGD)）训练该模型
        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate) 
        # 通过 clip_gradients_by_norm 将梯度裁剪应用到我们的优化器。梯度裁剪可确保梯度大小在训练期间不会变得过大，梯度过大会导致梯度下降法失败。
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

        linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=feature_columns,
            optimizer = my_optimizer
        )

        # 5、训练模型
        _ = linear_regressor.train(
            input_fn = lambda : self.my_input_fn(my_feature, my_target),
            steps=100
        )

        # 6、评估模型
        predict_input_fn = lambda : self.my_input_fn(my_feature, my_target, num_epochs=1, shuffle=False)
        predictions = linear_regressor.predict(input_fn=predict_input_fn)
        predictions = np.array([item['predictions'][0] for item in predictions])
        mean_squared_error = metrics.mean_squared_error(predictions, my_target)
        root_mean_squared_error = math.sqrt(mean_squared_error)
        print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
        print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)
    
    def my_input_fn(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):
        '''     4、定义输入函数     '''
        features = {key:np.array(value) for key, value in dict(features).items()}
        ds = Dataset.from_tensor_slices((features, targets))
        ds = ds.batch(batch_size).repeat(num_epochs)
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

if __name__ == "__main__":
    linear = LinearRegressorDemo()
    california_housing_dataframe = linear.get_dataset(linear.path, linear.split)
    california_housing_dataframe = california_housing_dataframe.reindex(
        np.random.permutation(california_housing_dataframe.index)
    )
    california_housing_dataframe[linear.target_name] /= 1000
    linear.train_dataset(california_housing_dataframe, linear.feature_name, linear.target_name)
    



