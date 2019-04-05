#!/usr/bin/env python3
# _*_ encoding:utf-8 _*_
'''
@File    :    FaceRecognitionDemo2.py
@Time    :    2019/04/05 15:07:27
@Author  :    Jayden Huang
@Version :    v1.0
@Contact :    Hjdong8@163.com
@Desc    :    Face Recognition Compare Faces, Only for study.
'''

# Here put the import lib
import os
import face_recognition

image_path = os.path.abspath("./MachineLearning/Instance/face_login/image/JayChou")
image_name = "JayChou.jpg"
image_unknow = "Unknow_4.jpg"
image_error = "Error.jpg"

image_known = face_recognition.load_image_file(os.path.join(image_path, image_name))
image_unknow = face_recognition.load_image_file(os.path.join(image_path, image_unknow))
image_error = face_recognition.load_image_file(os.path.join(image_path, image_error))

knowm_encoding = face_recognition.face_encodings(image_known)
unknowm_encoding = face_recognition.face_encodings(image_unknow)
error_encoding = face_recognition.face_encodings(image_error)

result = face_recognition.compare_faces(knowm_encoding, unknowm_encoding[0])
result_error = face_recognition.compare_faces(knowm_encoding, error_encoding[0])
print(result)
print(result_error)

