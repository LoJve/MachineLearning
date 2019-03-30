#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    FashionMnist.py
@Time    :    2019/03/28 21:10:18
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    TensorFlow Fashion Mnist Demo
'''

# Here put the import lib
import tensorflow  as tf
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import MatplotlibDemo as pltDemo

class FashionMnist(object):
    def __init__(self):
        self.dirname = os.path.join("./MachineLearning/DataSet", "FashionMnist")
        self.files = [
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
        ]
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    def load_file(self):
        paths = []
        for fname in self.files:
            paths.append(os.path.join(self.dirname, fname))
        
        train_labels = self.__read_gzip_file(paths[0], offset=8)
        train_images = self.__read_gzip_file(paths[1], offset=16).reshape(len(train_labels), 28, 28)
        test_labels = self.__read_gzip_file(paths[2], offset=8)
        test_images = self.__read_gzip_file(paths[3], offset=16).reshape(len(test_labels), 28, 28)

        return (train_images, train_labels), (test_images, test_labels)

    def __read_gzip_file(self, file_path, offset=8):
        '''Read the gzip file of dataset'''
        with gzip.open(file_path, 'rb') as fdata:
            return np.frombuffer(fdata.read(), dtype=np.int8, offset=offset)
    
    def preprocessing_data(self, x_train, x_test):
        '''Preprocessing data'''
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        return x_train, x_test
    
    def train_mnist(self, x_train, y_train, epochs = 5):
        '''Train fashion mnist'''
        # 设置层
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28,28)),       # 第一层：从二维数组（28 * 28）转换为一维数组（28 * 28 = 784）
            tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 第二层：隐藏层，包含128个节点（神经元），使用relu为激活函数
            tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 第三层：全连接层，具有10个节点的softmax层
        ])

        # 编译模型
        # 步骤：
        # 1、损失函数：衡量模型在训练期间的准确率；
        # 2、优化器： 根据模型看到的数据机器损失函数更新模型的方式；
        # 3、指标： 用于监控训练和测试步骤。
        model.compile(
            optimizer = tf.train.AdamOptimizer(),
            loss = "sparse_categorical_crossentropy",
            metrics = ['accuracy']
        )

        # 训练模型
        # 1、将训练数据馈送到模型中
        # 2、模型学习将图像和标签相关联
        # 3、对测试数据进行预测
        model.fit(x_train, y_train, epochs=epochs)

        return model
    
    def predict_mnist(self, model : tf.keras.Sequential, x_test):
        '''Predict mnist'''
        # 做出预测
        return model.predict(x_test)
    
    def evaluate_mnist(self, model : tf.keras.Sequential, x_test, y_test):
        '''Evalute accuracy'''
        # 评估准确率
        return model.evaluate(x_test, y_test)

if __name__ == "__main__":
    fashion_mnist = FashionMnist()
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_file()
    train_images, test_images = fashion_mnist.preprocessing_data(train_images, test_images)
    model = fashion_mnist.train_mnist(train_images, train_labels)
    predict_array = fashion_mnist.predict_mnist(model, test_images)
    print("Prediction data:")
    print(predict_array[0])

    test_loss, test_acc = fashion_mnist.evaluate_mnist(model, test_images, test_labels)
    print("Prediction evaluate:")
    print('Test accuracy : ', test_acc)

    pltDemo.MatplotlibDemo.plot_predict_images(predict_array, 
        test_labels, test_images, fashion_mnist.class_names, num_rows=5, num_cols=3)
