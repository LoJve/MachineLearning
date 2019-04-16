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
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784, )),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy'])
    return model


# 检查点回调用法
checkpoint_path = "./MachineLearning/Tensorflow/test/save/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=True)

model = create_model()

model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels), callbacks=[cp_callback])

#测试
model_test = create_model()
loss, acc = model_test.evaluate(test_data, test_labels)
print('Untrained model, accuracy : {:5.2f}%'.format(100*acc))

model_test.load_weights(checkpoint_path)
loss, acc = model_test.evaluate(test_data, test_labels)
print('Untrained model, accuracy : {:5.2f}%'.format(100*acc))

# 检查点回调选项
checkpoint_path = "./MachineLearning/Tensorflow/test/save/training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)

model = create_model()
model.fit(train_data, train_labels, epochs=50, callbacks=[cp_callback], validation_data=(test_data, test_labels), verbose=0)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)

# 测试
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_data, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

