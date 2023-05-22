import numpy as np

from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

import os
import cv2

import pickle

def detect(label_mapping):
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, model):
        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(img, scaleFactor, minNeighbors)
        
        coords = []
        
        for(x,y,w,h) in features:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            roi = cv2.resize(img[y:y+h, x:x+w], (200,200))
            pred = model.predict(roi.reshape((-1,200,200,3)))     
            print(pred)          
            confidence = np.max(pred)
            print(confidence)
            if confidence < 0.8:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
            else:
                # print(pred)
                rs = np.argmax(pred)
                for key, value in label_mapping.items():
                    if value == rs:  
                        name = key
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords.append([x, y, w, h])
        return coords

    def recognize(img, model, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255,255,255), "Face", model)
        return img

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    model = models.load_model("model_first.h5")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, model, faceCascade)
        cv2.imshow("Face detection", img)
        
        if cv2.waitKey(1)==13: #Enter
            break

    video_capture.release()
    cv2.destroyAllWindows()