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

class ImdbMnist(object):
    def __init__(self):
        self.imdb = tf.keras.datasets.imdb
        self.path = os.path.join(os.path.abspath("./MachineLearning/Dataset"), "ImdbMnist")
        self.imdb_file_name = "imdb.npz"
        self.vocab_fiel_name = "imdb_word_index.json"
        self.num_of_data = 15000
        self.len_of_data = 256
        self.output_dim_of_level_one = 16
        self.epochs = 10
        self.batch_size = 512
    
    def load_data(self):
        '''Download and load the imdb dataset'''
        # 下载imdb数据集
        imdb_path = os.path.join(self.path, self.imdb_file_name)
        return self.imdb.load_data(imdb_path)
    
    def change_int_to_vocab(self):
        '''Change the int to imdb vocab'''
        # 将数字转换为字词
        vocab_path = os.path.join(self.path, self.vocab_fiel_name)
        return self.imdb.get_word_index(vocab_path)
    
    def decode_review(self, word_index):
        word_index = { k : v+3 for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3

        reverse_word_index = dict([(v, k) for k, v in word_index.items()])
        return ' '.join([reverse_word_index.get(index, "?") for index in reverse_word_index])

    def preprocessing_data(self, t_data, value=0, padding='post', maxlen=256):
        ''' Data preprocessing '''
        # 数据预处理
        # 影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。
        # 我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。
        # 我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层
        # 由于影评的长度必须相同，我们将使用 pad_sequences 函数将长度标准化
        t_data = tf.keras.preprocessing.sequence.pad_sequences(t_data, 
                                                                value=value, 
                                                                padding=padding, 
                                                                maxlen=maxlen)
        return t_data
    
    def train_mnist(self, train_data, train_labels):
        ''' Train imdb mnist model '''
        # 构建模型
        # 主要决策：
        # 1、模型中使用多少个层？
        # 2、每层中使用多少个隐藏单元？
        vocab_size = self.num_of_data * self.len_of_data
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(vocab_size, self.output_dim_of_level_one))  # 第一层，输入为(15000 * 256)， 输出为（16，）
        self.model.add(tf.keras.layers.GlobalAveragePooling1D()) # 池化层
        self.model.add(tf.keras.layers.Dense(self.output_dim_of_level_one, activation=tf.nn.relu)) # 第二层，隐藏层，输入为（16，）
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

        # 损失函数和优化器
        self.model.compile(optimizer=tf.train.AdadeltaOptimizer(),
                    loss = "binary_crossentropy",
                    metrics=['accuracy'])
        
        # 创建验证集
        # 需检查模型处理从未见过的数据的准确率。
        # 仅使用训练数据开发和调整模型，然后仅使用一次测试数据评估准确率。
        x_train = train_data[len(train_data) - self.num_of_data:]
        y_train = train_labels[len(train_labels) - self.num_of_data:]

        x_verify = train_data[: len(train_data) - self.num_of_data]
        y_verify = train_labels[: len(train_labels) - self.num_of_data]

        # 训练模型
        return self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                validation_data=(x_verify, y_verify), verbose=1)
    
    def evaluate_mnist(self, test_data, test_labels):
        ''' Evaluate mnist model'''
        # 评估模型
        return self.model.evaluate(test_data, test_labels)
    
    def plot_loss_picture(self, history_dict):
        data = history_dict["loss"]
        val_data = history_dict['val_loss']
        epochs = range(1, len(data) + 1)
        plt.plot(epochs, data, 'bo', label ="Training loss")
        plt.plot(epochs, val_data, 'b', label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

    def plot_acc_picture(self, history_dict):
        data = history_dict["acc"]
        val_data = history_dict["val_acc"]
        epochs = range(1, len(data) + 1)
        plt.plot(epochs, data, 'bo', label ="Training acc")
        plt.plot(epochs, val_data, 'b', label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()


if __name__ == "__main__":
    mnist = ImdbMnist()
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    train_data = mnist.preprocessing_data(train_data)
    test_data = mnist.preprocessing_data(test_data)
    history = mnist.train_mnist(train_data, train_labels)
    result = mnist.evaluate_mnist(test_data, test_labels)
    print(result)

    # 创建准确率和损失随时间变化的图
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    mnist.plot_loss_picture(history.history)
    plt.subplot(1, 2, 2)
    mnist.plot_loss_picture(history.history)
    plt.show()

