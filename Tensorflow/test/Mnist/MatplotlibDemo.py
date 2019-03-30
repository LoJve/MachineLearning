#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    MatplotlibDemo.py
@Time    :    2019/03/29 21:50:47
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Matplotlib Demo
'''

# Here put the import lib
import matplotlib.pyplot as plt
import numpy as np

class MatplotlibDemo(object):
    @staticmethod
    def plot_single_picture(image):
        '''Show single picture'''
        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.grid(False)
        plt.show()
    
    @staticmethod
    def plot_many_pictures(images, labels, class_names, num_rows=5, num_cols=5):
        '''Show a number of pictures'''
        plt.figure(figsize=(num_cols, num_rows))
        num_images = num_rows * num_cols
        for index in range(num_images):
            plt.subplot(num_rows, num_cols, index + 1)
            plt.imshow(images[index], cmap=plt.cm.binary)
            plt.grid(False)
            # plt.xticks([])
            # plt.yticks([])
            plt.xlabel(class_names[labels[index]])
        plt.show()
    
    @staticmethod
    def plot_predict_images(predict_array, test_labels, images, 
        class_names, num_rows=1, num_cols=1):
        '''Plot a number of predict image and result'''
        num_images = num_rows * num_cols
        plt.figure(figsize=(2*2*num_cols, 2*num_rows))
        for index in range(num_images):
            plt.subplot(num_rows, 2*num_cols, 2*index+1)
            MatplotlibDemo.__plot_single_predict_image(predict_array, 
                test_labels, images, index, class_names)
            plt.subplot(num_rows, 2*num_cols, 2*index+2)
            MatplotlibDemo.__plot_single_predict_plot(predict_array, 
                test_labels, index, class_names)
        plt.show()

    @staticmethod
    def __plot_single_predict_image(predict_array, test_labels, 
            images, index, class_names):
        ''' Plot single predict image'''
        pred_array, t_label, image = \
            predict_array[index], test_labels[index], images[index]
        plt.grid(True)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap=plt.cm.binary)
        pred_label = np.argmax(pred_array)
        color = 'blue' if pred_label == t_label else 'red'
        plt.xlabel("{} {:2.0f}% {}".format(
            class_names[pred_label], 100 * np.max(pred_array), class_names[t_label]
        ), color=color)

    @staticmethod
    def __plot_single_predict_plot(predict_array, test_labels, index, class_names):
        ''' Plot single predict result'''
        pred_array, t_label = predict_array[index], test_labels[index]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), pred_array, color='#777777')
        plt.ylim([0,1])
        pred_label = np.argmax(pred_array)
        thisplot[pred_label].set_color('red')
        thisplot[t_label].set_color('blue')
