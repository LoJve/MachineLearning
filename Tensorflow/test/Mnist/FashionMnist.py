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
from tensorflow.python.keras.utils.data_utils import get_file
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    dirname = os.path.join("./MachineLearning/Tensorflow/test/dataset","FashionMnist")
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(dirname, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = load_data()
# print(train_images.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def draw_picture(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def draw_many_picture():
    plt.figure(figsize=(10,10))
    for index in range(25):
        plt.subplot(5, 5, index+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[index], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[index]])
    plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0
# draw_many_picture()

# 设置层
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),    # 第一层：从二维数组（28 * 28）转换为一维数组（28 * 28 = 784）
    tf.keras.layers.Dense(128,activation=tf.nn.relu),   # 第二层：隐藏层，包含128个节点（神经元），使用relu为激活函数
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 第三层：全连接层，具有10个节点的softmax层
])

# 编译模型
# 步骤：
# 1、损失函数：衡量模型在训练期间的准确率；
# 2、优化器： 根据模型看到的数据机器损失函数更新模型的方式；
# 3、指标： 用于监控训练和测试步骤。
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型
# 1、将训练数据馈送到模型中
# 2、模型学习将图像和标签相关联
# 3、对测试数据进行预测
model.fit(train_images, train_labels, batch_size=16, epochs=5)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy : ', test_acc)

# 做出预测
predictions = model.predict(test_images)
print("Prediction data:")
print(predictions[0])

prediction_label = np.argmax(predictions[0])
print("Test label : ", test_labels[0])
print("Predict label : ", prediction_label)
