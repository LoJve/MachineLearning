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
import face_recognition
import os

image_path = os.path.abspath("./MachineLearning/Instance/face_login/image")
image_name = "test.jpg"

image = face_recognition.load_image_file(os.path.join(image_path, image_name))
face_locations = face_recognition.face_locations(image)
