#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    OverfitDemo.py
@Time    :    2019/04/11 21:47:36
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Overfitting and underfitting demo
'''

# Here put the import lib
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_WORDS = 10000
path = os.path.join(os.path.abspath("./MachineLearning/Dataset"), "ImdbMnist")
imdb_file_name = "imdb.npz"

# 加载数据集
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(os.path.join(path, imdb_file_name), num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    result = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        result[i, word_indices] = 1.0
    return result

train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_data, NUM_WORDS)

# 创建模型
def build_model(unit, kernel_regularizer=None, drop_out=None):
    # model = keras.Sequential([
    #     keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, input_shape=(NUM_WORDS,)),
    #     keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer),
    #     keras.layers.Dense(1, activation=tf.nn.sigmoid)
    # ])

    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer, input_shape=(NUM_WORDS,)))
    if drop_out:
        model.add(keras.layers.Dropout(drop_out))
    model.add(keras.layers.Dense(16, activation=tf.nn.relu, kernel_regularizer=kernel_regularizer))
    if drop_out:
        model.add(keras.layers.Dropout(drop_out))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=['accuracy', 'binary_crossentropy']
    )

    history = model.fit(
        train_data,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_data, test_labels),
        verbose=2
    )
    return history

# 创建基准模型
baseline_history = build_model(16)

# baseline_model = keras.Sequential([
#     keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(16, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# baseline_model.compile(
#     optimizer='adam',
#     loss = 'binary_crossentropy',
#     metrics=['accuracy', 'binary_crossentropy']
# )

# baseline_history = baseline_model.fit(
#     train_data,
#     train_labels,
#     epochs=20,
#     batch_size=512,
#     validation_data=(test_data, test_labels),
#     verbose=2
# )

# 创建一个更小的模型
# small_history = build_model(4)

# small_model = keras.Sequential([
#     keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(4, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# small_model.compile(
#     optimizer='adam',
#     loss = 'binary_crossentropy',
#     metrics=['accuracy', 'binary_crossentropy']
# )

# small_history = small_model.fit(
#     train_data,
#     train_labels,
#     epochs=20,
#     batch_size=512,
#     validation_data=(test_data, test_labels),
#     verbose=2
# )

# 创建一个更大的模型
# bigger_history = build_model(512)

# bigger_model = keras.Sequential([
#     keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
#     keras.layers.Dense(512, activation=tf.nn.relu),
#     keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])

# bigger_model.compile(
#     optimizer='adam',
#     loss = 'binary_crossentropy',
#     metrics=['accuracy', 'binary_crossentropy']
# )

# bigger_history = bigger_model.fit(
#     train_data,
#     train_labels,
#     epochs=20,
#     batch_size=512,
#     validation_data=(test_data, test_labels),
#     verbose=2
# )


# 添加权重正则化
l2_model_history = build_model(16, kernel_regularizer=keras.regularizers.l2(0.001))

# 添加丢弃层
dpt_model_history = build_model(16, drop_out=0.5)

# 绘制训练损失和验证损失图表
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_' + key], '--', label=name.title() + 'Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title() + 'Train')
    
    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.show()

# plot_history([('baseline', baseline_history),
#             ('small', small_history),
#             ('bigger', bigger_history)])

plot_history([('baseline', baseline_history),
            ('l2', l2_model_history),
            ('dropout', dpt_model_history)])

