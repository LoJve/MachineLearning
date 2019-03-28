#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    MtcnnDemo.py
@Time    :    2019/03/27 23:20:12
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Mtcnn Demo
'''

# Here put the import lib


import mxnet as mx
from mtcnn import mtcnn
# from mtcnn_detector import MtcnnDetector
import cv2 as cv
import os

imagePath = "./MachineLearning/Instance/face_login/image/test.jpg"
flag = os.path.exists(imagePath)

# 1、读入要检测的图片
image = cv.imread(imagePath)

# 2、加载训练好的模型参数，构建检测对象
# model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False
detector = mtcnn.MTCNN()

while True:
    # 3、执行推理操作
    all_boxes = detector.detect_faces(image)

    # 4、绘制目标框
    for mybox in all_boxes:
        cv.rectangle(image, (mybox["box"][0], mybox["box"][1]), (mybox["box"][2], mybox["box"][3]), (0,0,255))
    cv.imshow('Image', image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()