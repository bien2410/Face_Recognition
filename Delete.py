import pickle
import os
import shutil
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from Trainning import update_train

def delete(name, label_mapping):
    # xoa thu muc
    # with open('label_mapping.pkl', 'rb') as f:
    #     label_mapping = pickle.load(f)
    if name not in label_mapping:
        return False
    for path in os.listdir('image'):
        path_1 = os.path.join('image', path)
        path_2 = os.path.join(path_1, name)
        if os.path.isdir(path_2):
            shutil.rmtree(path_2)

    # xoa tu dien
    start_value = label_mapping[name]

    del label_mapping[name]

    for key in list(label_mapping.keys()):
        if label_mapping[key] > start_value:
            label_mapping[key] -= 1
    
    #cap nhat mo hinh
    update_train(label_mapping)

    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    return True

# print(delete('Bien'))
