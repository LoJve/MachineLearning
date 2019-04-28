#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    OpenCVDemo2.py
@Time    :    2019/04/29 00:04:36
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    OpenCV Drawing Demo
'''

# Here put the import lib
import os
import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.int8)

def test1():
    cv2.line(img, (0,0), (511,511), (255,0,0), 5)

def test2():
    cv2.rectangle(img, (384, 0), (510, 128), (0, 255,0), 3)

def test3():
    cv2.circle(img, (447, 63), 63, (0,0,255), -1)

def test4():
    cv2.ellipse(img, (256, 256), (100,50), 0, 0, 180, 255, -1)

def test5():
    pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img, [pts], True, (0,255,255))

def test6():
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "OpenCV", (10,500), font, 4, (255,255,255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    cv2.imshow("image", img)
    if(cv2.waitKey(0) & 0xff == ord('q')):
        cv2.destroyAllWindows()

