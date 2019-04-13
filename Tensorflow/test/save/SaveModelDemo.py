#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    SaveModelDemo.py
@Time    :    2019/04/13 22:02:11
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Save and restore model
'''

# Here put the import lib
import tensorflow as tf
from tensorflow import keras
import os

path = os.path.join(os.path.abspath("./MachineLearning/Dataset"), "Mnist")
imdb_file_name = "mnist.npz"

(train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_data = train_data[:1000].reshape(-1, 28 * 28) / 255.0
test_data = test_data[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss = keras.losses.sparse_categorical_crossentropy,
        metircs=['accuracy']
    )
    return model

model = create_model()


