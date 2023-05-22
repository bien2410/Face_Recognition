import numpy as np

from PIL import Image

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import random
import os
import cv2

import pickle

def getData(dir_path, listData, label_mapping, one_hot_labels):
    for path in os.listdir(dir_path):
        path_1 = os.path.join(dir_path, path)
        for path_2 in os.listdir(path_1):
            path_3 = os.path.join(path_1, path_2)
            img = np.array(Image.open(path_3))
            listData.append((img, one_hot_labels[label_mapping[path_1.split('\\')[1]]]))
    return listData

def getData2(dir_path, listData, label_mapping, one_hot_labels, name):
    for path in os.listdir(dir_path):
        if path == name:
            continue
        path_1 = os.path.join(dir_path, path)
        for path_2 in os.listdir(path_1):
            path_3 = os.path.join(path_1, path_2)
            img = np.array(Image.open(path_3))
            listData.append((img, one_hot_labels[label_mapping[path_1.split('\\')[1]]]))
    return listData

def train_base():
    train_path = 'image/train'

    list_filename = []

    for path in os.listdir(train_path):
        path_1 = os.path.join(train_path, path)
        list_filename.append(path_1.split('\\')[1])

    labels = np.array(list_filename)
    label_mapping = {label: i for i, label in enumerate(labels)}

    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)

    encoded_labels = np.array([label_mapping[label] for label in labels])
    one_hot_labels = to_categorical(encoded_labels)

    train = []
    train = getData(train_path, train, label_mapping, one_hot_labels)
    
    np.random.shuffle(train)

    Xtrain = np.array([x[0] for x in train])
    Ytrain = np.array([x[1] for x in train])

    model_training_first = models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(200,200,3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.15), #cat bo 15%
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(128, (3, 3),  activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),
        
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(len(label_mapping), activation='softmax')
    ])

    model_training_first.summary() #tong ket

    model_training_first.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_training_first.fit(Xtrain, Ytrain, epochs= 5)

    model_training_first.save('model_first.h5')

def add_train(name, label_mapping):
    train_path = 'image/train' 
    if name in label_mapping:
        pass
    else:
        label_mapping[name] = len(label_mapping)
        with open("label_mapping.pkl", "wb") as f:
            pickle.dump(label_mapping, f)
    id = label_mapping[name]

    numbers = []
    for key, value in label_mapping.items():
        numbers.append(value)
    encoded_labels = np.array(numbers)
    one_hot_labels = to_categorical(encoded_labels)
    train = []
    train = getData2(train_path, train, label_mapping, one_hot_labels, name)
    train = random.sample(train, 240)
    one_hot = to_categorical(id)
    for path in os.listdir(train_path + '/' + name):
        path_1 = os.path.join(train_path + '/' + name, path)
        img = np.array(Image.open(path_1))
        train.append((img, one_hot))

    np.random.shuffle(train)
    Xtrain = np.array([x[0] for x in train])
    Ytrain = np.array([x[1] for x in train])

    loaded_model = models.load_model('model_first.h5')
    last_layer = loaded_model.layers[-1]
    new_last_layer = layers.Dense(len(label_mapping), activation=last_layer.activation, name = 'new_last_layer')
    model_with_new_last_layer = Model(loaded_model.input, new_last_layer(loaded_model.layers[-2].output))
    
    # for layer in model_with_new_last_layer.layers[:-1]:
    #     layer.trainable = False

    model_with_new_last_layer.summary()
   
    model_with_new_last_layer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_with_new_last_layer.fit(Xtrain, Ytrain, epochs=5)

    model_with_new_last_layer.save('model_first.h5')

def update_train(label_mapping):
    train_path = 'image/train'
    numbers = []
    for key, value in label_mapping.items():
        numbers.append(value)
    encoded_labels = np.array(numbers)

    one_hot_labels = to_categorical(encoded_labels)
    train = []
   
    train = getData(train_path, train, label_mapping, one_hot_labels)
    train = random.sample(train, int(len(train) / 2))
    np.random.shuffle(train)
    Xtrain = np.array([x[0] for x in train])
    Ytrain = np.array([x[1] for x in train])

    loaded_model = models.load_model('model_first.h5')
    last_layer = loaded_model.layers[-1]
    new_last_layer = layers.Dense(len(label_mapping), activation=last_layer.activation, name = 'new_last_layer')
    model_with_new_last_layer = Model(loaded_model.input, new_last_layer(loaded_model.layers[-2].output))
    
    # for layer in model_with_new_last_layer.layers[:-1]:
    #     layer.trainable = False

    model_with_new_last_layer.summary()
   
    model_with_new_last_layer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_with_new_last_layer.fit(Xtrain, Ytrain, epochs=5)

    model_with_new_last_layer.save('model_first.h5')

# train_base()