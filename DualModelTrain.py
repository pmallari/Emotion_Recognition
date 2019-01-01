import numpy as np 
import pandas as pd
import pickle
import os
import cv2
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Reproducibility
np.random.seed(42)

# Allocate GPU power
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.75)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Declare current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path +"/pickled_data/"
image_path = dir_path + "/images/"
log_path = dir_path + "/CNNlog/"

start = time()

print("----------\nCreating dataset...\n----------")

print("----------\nImporting Neutral Training Image Set...\n----------")
pickling = open(data_path + "NeutralTestImage.pickle", "rb")
X1 = pickle.load(pickling)
pickling.close()

print("----------\nImporting Neutral Training Label Set...\n----------")
pickling = open(data_path + "NeutralTestLabel.pickle", "rb")
y1 = pickle.load(pickling)
pickling.close()
    
print("----------\nImporting Emotion Training Image Set...\n----------")
pickling = open(data_path + "EmotionTestImage.pickle", "rb")
X2 = pickle.load(pickling)
pickling.close()
    
print("----------\nImporting Emotion Training Label Set...\n----------")
pickling = open(data_path + "EmotionLabel.pickle", "rb")
y2 = pickle.load(pickling)
pickling.close()

run_time = time() - start
print("----------\nDataset created\n----------")
print('Unpacking and data processing took {:.2f}s.'.format(run_time))

conv_layer = 3
start = 5
cnn_layer = 3
img_size = 512

####################################
#####  NEUTRAL MODEL TRAINING  #####
####################################

start_time = time()
print("\n\n----------\nBeginning Training of Neutral Model\n----------")

NAME = "NeutralModel_{}".format(int(time()))
print("Training model: {}".format(NAME))
model = Sequential()
model.add(Conv2D(16, (start,start), input_shape=(img_size, img_size, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

for _ in range(conv_layer):
    model.add(Conv2D(16, (cnn_layer,cnn_layer), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))


tensorboard = TensorBoard(log_dir=log_path+ "/{}".format(NAME))
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )

model.fit(X1, y1,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[tensorboard])

print("----------\nTraining Complete of Neutral Model\n----------")
print("Total training time {:.2f}s".format(time() - start_time))
print(model.summary())
model.save(dir_path + '/NeutralModel.model')

####################################
#####  EMOTION MODEL TRAINING  #####
####################################

start_time = time()
print("\n\n----------\nBeginning Training of Emotion Model\n----------")

NAME = "EmotionModel_{}".format(int(time()))
print("Training model: {}".format(NAME))
model = Sequential()
model.add(Conv2D(16, (start,start), input_shape=(img_size, img_size, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

for _ in range(conv_layer):
    model.add(Conv2D(16, (cnn_layer,cnn_layer), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(4))
model.add(Activation('softmax'))


tensorboard = TensorBoard(log_dir=log_path+ "/{}".format(NAME))
model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
            )

model.fit(X2, y2,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[tensorboard])

print("----------\nTraining Complete of Emotion Model\n----------")
print("Total training time {:.2f}s".format(time() - start_time))
print(model.summary())
model.save(dir_path + '/EmotionModel.model')