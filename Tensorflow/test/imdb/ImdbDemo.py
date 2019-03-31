#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    ImdbDemo.py
@Time    :    2019/03/31 14:28:47
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Tensorflow imdb demo
'''

# Here put the import lib
import tensorflow as tf
import os
import matplotlib.pyplot as plt

imdb = tf.keras.datasets.imdb
path = os.path.abspath(os.path.join('./MachineLearning/Dataset', 'ImdbMnist'))

# 下载imdb数据集
imdb_file_name = "imdb.npz"
imdb_path = os.path.join(path, imdb_file_name)
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(path=imdb_path)

# 将数字转换为字词
json_file_name = "imdb_word_index.json"
json_path = os.path.join(path, json_file_name)
word_index = imdb.get_word_index(path=json_path)

word_index = {k:v+3 for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED'] = 3

reversed_word_index = dict([(v, k) for k,v in word_index.items()])

def decode_review(text):
    return ' '.join([reversed_word_index.get(index, '?') for index in text])

# print(decode_review(train_data[0]))

# 准备数据
# 影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。
# 我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。
# 我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层
# 由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 构建模型
# 主要决策：
# 1、模型中使用多少个层？
# 2、每层中使用多少个隐藏单元？
# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 15000 * 256

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# 损失函数和优化器
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建验证集
# 需检查模型处理从未见过的数据的准确率。
# 仅使用训练数据开发和调整模型，然后仅使用一次测试数据评估准确率。
num = 10000
x_val = train_data[:num]
partial_x_train = train_data[num:]

y_val = train_labels[:num]
partial_y_train = train_labels[num:]

# 训练模型
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# history = model.fit(partial_x_train, partial_y_train, epochs=40, 
#         batch_size=512, validation_data=(x_val, y_val), verbose=1)

# 评估模型
result = model.evaluate(test_data, test_labels)
print(result)

# 创建准确率和损失随时间变化的图
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

