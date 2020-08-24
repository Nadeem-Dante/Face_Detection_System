# Facial Detection System
# Basic first draft
import cv2
from matplotlib import pyplot as plt
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

img = cv2.imread('TestImage01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 3)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 3)

resized_image = cv2.resize(img, (650, 500))

cv2.imshow('Gray', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()