import cv2
import os

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

for i in os.listdir(''):
    for j in os.listdir(i):
        gray = cv2.imread(j, 0)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) <= 0:
            os.remove(i)