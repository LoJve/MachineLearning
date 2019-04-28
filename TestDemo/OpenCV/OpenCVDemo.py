#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    OpenCVDemo.py
@Time    :    2019/04/28 22:34:37
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    OpenCV Demo
'''

# Here put the import lib
import cv2
import os

image_path = "./MachineLearning/TestDemo/image"
image_file = "Xin.jpg"

def test1():
    image = cv2.imread(os.path.join(image_path, image_file), 0)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        path = os.path.join(image_path, "Xin_copy.jpg")
        cv2.imwrite(path, image)
        cv2.destroyAllWindows()

def test2():
    index = 0
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("image", gray)
        k = cv2.waitKey(1)
        if k == ord('p'):
            frame_print = frame
            cv2.namedWindow('Print Image', cv2.WINDOW_NORMAL)
            cv2.imshow("Print Image", frame)
        elif k == ord('s'):
            path = os.path.join(image_path, "JD")
            path = os.path.join(path, "JD_{0}.jpg".format(index))
            cv2.imwrite(path, frame_print)
            index += 1
        elif k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test3():
    cap = cv2.VideoCapture(0)

    path = os.path.join(image_path, "video")
    path = os.path.join(path, 'output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, 20.0, (640, 480))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if(ret):
            #frame = cv2.flip(frame, 0)
            out.write(frame)
            cv2.imshow('frame', frame)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def test4():
    path = os.path.join(image_path, "video")
    path = os.path.join(path, 'output.avi')
    
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("image", gray)
            if(cv2.waitKey(1) & 0xFF == ord('q')):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    test4()
