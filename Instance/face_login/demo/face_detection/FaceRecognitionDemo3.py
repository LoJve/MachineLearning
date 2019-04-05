#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    FaceRecognitionDemo3.py
@Time    :    2019/04/05 15:30:47
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Real-time face detection
'''

# Here put the import lib
import os
import face_recognition
import cv2 as cv
import numpy as np

image_path = os.path.abspath("./MachineLearning/Instance/face_login/image/JayChou")
image_name = "JayChou.jpg"
xin_image = "Xin.jpg"

video = cv.VideoCapture(0)

jaychou_jgp = face_recognition.load_image_file(os.path.join(image_path, image_name))
jaychou_encoding = face_recognition.face_encodings(jaychou_jgp)[0]

xin_jpg = face_recognition.load_image_file(os.path.join(image_path, xin_image))
xin_encoding = face_recognition.face_encodings(xin_jpg)[0]

known_face_encodings = [
    jaychou_encoding,
    xin_encoding
]

known_face_names = [
    'Jay Chou',
    'Xin'
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video.read()
    small_frame = cv.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknow"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            face_names.append(name)
    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        multiple = 4
        top *= multiple
        right *= multiple
        bottom *= multiple
        left *= multiple

        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv.rectangle(frame, (left, bottom-35), (right, bottom), (0, 255, 0), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left+6, bottom-6), font, 1.0, (255,255,255), 1)
    
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()
