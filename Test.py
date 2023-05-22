import numpy as np
import os
from PIL import Image
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
import cv2
import pickle
from tensorflow.keras.utils import to_categorical

def getData(dir_path, listData, label_mapping, one_hot_labels):
    for path in os.listdir(dir_path):
        path_1 = os.path.join(dir_path, path)
        for path_2 in os.listdir(path_1):
            path_3 = os.path.join(path_1, path_2)
            img = np.array(Image.open(path_3))
            listData.append((img, one_hot_labels[label_mapping[path_1.split('\\')[1]]]))
    return listData

def testing():
    with open('label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)

    numbers = list(label_mapping.values())
    encoded_labels = np.array(numbers)
    one_hot_labels = to_categorical(encoded_labels)

    test_path = 'image/test'

    test = []
    test = getData(test_path, test, label_mapping, one_hot_labels)

    np.random.shuffle(test)

    Xtest = np.array([x[0] for x in test])
    Ytest = np.array([x[1] for x in test])

    model = models.load_model('model_first.h5')

    i = 0
    sl = 0
    for x, y in zip(Xtest, Ytest):
        pred = model.predict(x.reshape((-1,200,200,3)))
        doan = np.argmax(pred)
        dung = np.argmax(y)
        print(f"{i + 1}, pred: {doan}, true: {dung}")
        if doan == dung:
            sl += 1
        i += 1
    print(f"{sl}/{i}")
testing()