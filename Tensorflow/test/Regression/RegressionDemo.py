#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    RegressionDemo.py
@Time    :    2019/04/08 22:07:14
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Tensorflow Regression Demo : Predict fuel efficiency
'''

# Here put the import lib
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "./MachineLearning/DataSet/Regression"
file_name = "auto-mpg.data"

dataset_path = os.path.join(file_path, file_name)


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()              
# print(dataset.tail())
# print(dataset.isna().sum())

dataset = dataset.dropna()
origin = dataset.pop("Origin")
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

# print(dataset.tail())

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
sns.set(style="ticks")
print(sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde'))
plt.show()

train_stats = train_dataset.describe()
train_stats.pop('MPG')
train_stats = train_dataset.transpose()
print(train_stats)

train_labels = train_dataset.pop('MPG')
test_lables = test_dataset.pop('MPG')

# Normalize the data
def norm(data):
    return (data - train_stats['mean']) / train_stats['std']
norm_train_data = norm(train_dataset)
norm_test_data = norm(test_dataset)

# Build the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.001)

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
    return model

model = build_model()
print(model.summary())

example_batch = norm_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)

# Train the model
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end="")

EPOCHS = 1000

history = model.fit(
    norm_train_data, train_labels, epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[PrintDot()]
)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)
