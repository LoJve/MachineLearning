#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    FaceRecognitionDemo.py
@Time    :    2019/04/03 01:00:02
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Face Recognition Demo
'''

# Here put the import lib
# https://www.cnblogs.com/jyxbk/p/7677877.html
# https://www.jianshu.com/p/0b37452be63e
import face_recognition
import os
import cv2 as cv

image_path = os.path.abspath("./MachineLearning/Instance/face_login/image")
image_name = "test.jpg"

image = face_recognition.load_image_file(os.path.join(image_path, image_name))
# model = 'cnn', 速度慢，但是精度稍微高一些
# model = 'hog', 速度快，精度低
face_locations = face_recognition.face_locations(image, model='cnn')

image_cv = cv.imread(os.path.join(image_path, image_name))
for top, right, bottom, left in face_locations:
    cv.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
cv.imshow('image', image_cv)
cv.waitKey()
