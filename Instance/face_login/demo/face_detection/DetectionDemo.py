#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    DetectionDemo.py
@Time    :    2019/04/02 23:36:30
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Face Detection Demo
'''

# Here put the import lib
import cv2 
import os

image_path = os.path.abspath("./MachineLearning/Instance/face_login/image")
image_name = "test.jpg"

opencv_train_file_path = os.path.abspath("./MachineLearning/Instance/face_login/demo/face_detection")
opencv_train_default_file_name = "haarcascade_frontalface_default.xml"
opencv_train_alt_file_name = "haarcascade_frontalface_alt.xml"
opencv_train_eye_file_name = "haarcascade_eye.xml"

def opencv_detection():
    ''' OpenCV Face Detection '''
    # 读取图片
    image = cv2.imread(os.path.join(image_path, image_name))
    # 灰度转换: 转换为灰度的图片的计算强度降低
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 获取人脸识别的训练数据
    face_cascade = cv2.CascadeClassifier(os.path.join(opencv_train_file_path, opencv_train_default_file_name))
    # 检测人脸
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 4,
        minSize = (30,30),
        flags = cv2.IMREAD_GRAYSCALE
    )
    print("Find {0} faces".format(len(faces)))
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey()

def opencv_detection_new():
    cascade = cv2.CascadeClassifier(os.path.join(opencv_train_file_path, opencv_train_alt_file_name))
    nested = cv2.CascadeClassifier(os.path.join(opencv_train_file_path, opencv_train_eye_file_name))
    # 读取图片
    image = cv2.imread(os.path.join(image_path, image_name))
    # 灰度转换: 转换为灰度的图片的计算强度降低
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(image, cascade)
    vis = image.copy()
    draw_rects(vis, rects, (0, 255, 0))
    if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))
    cv2.imshow('facedetect', vis)
    cv2.waitKey()

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

opencv_detection_new()
