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
data_path = dir_path +"/data/"
image_path = dir_path + "/images/"
log_path = dir_path + "/CNNlog/"

start = time()
img_size = 512

def create_training_data():
    orig_img_size = 1718
    splice_img = int((2444-1718)/2)

    images = []
    labels = []
    img_error = []
    neutral_count = 0

    for img in os.listdir(image_path):
        label = img[-img[::-1].index('-'):-img[::-1].index('.')-1]
        if label == 'N':
            neutral_count += 1

        if (label != 'N') or (label == 'N' and neutral_count == 4):
            if neutral_count == 4:
                neutral_count = 0
            image = cv2.imread(image_path + img)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, splice_img:-splice_img]
            image = cv2.resize(image, (img_size, img_size))
            image = np.asarray(image)
            
            try:
                crop_img = image.reshape(img_size, img_size, 1)
                crop_img = crop_img / 255.0
                images.append(crop_img)
                labels.append(label)
            except Exception as e:
                img_error.append([img])
    
    enc = LabelEncoder()
    labels = np.asarray(labels).reshape(-1,1)
    labels = enc.fit_transform(labels)
    labels = np.ravel(labels)

    return np.asarray(images), np.asarray(labels)

print("----------\nCreating dataset...\n----------")
last_time = time()
X, y = create_training_data()
print(Counter(y))
run_time = time() - last_time
print("----------\nDataset created\n----------")
print('Unpacking and data processing took {:.2f}s.'.format(run_time))

# Training criterias
#conv_layers = [3, 4, 5]
#start_cnn = [3, 5, 7, 9, 11]
#cnn_layers = [3, 5, 7, 9, 11]

# Train and save specific model
conv_layers = [3]
start_cnn = [5]
cnn_layers = [3]

runs = 0
print("\n\n----------\nBeginning Training\n----------")

for conv_layer in conv_layers:
    for start in start_cnn:
        for cnn_layer in cnn_layers:
            runs += 1
            NAME = "{}-CNN-{}-startCNN-{}-CNNKern-{}".format(conv_layer, start, cnn_layer, int(time()))
            print("Run number: {}".format(runs))
            print("Training model: {}".format(NAME))
            model = Sequential()
            model.add(Conv2D(16, (start, start), input_shape=(img_size, img_size, 1), activation = 'relu'))
            model.add(MaxPooling2D(pool_size = (2,2)))

            for _ in range(conv_layer):
                model.add(Conv2D(16, (cnn_layer,cnn_layer), activation = 'relu'))
                model.add(MaxPooling2D(pool_size = (2,2)))
                model.add(Dropout(0.25))

            model.add(Flatten())

            #for _ in range(dense_layer):
            #    model.add(Dense(layer_size))
            #    model.add(Activation('relu'))
            #    model.add(Dropout(0.25))
            
            model.add(Dense(5))
            model.add(Activation('softmax'))


            tensorboard = TensorBoard(log_dir=log_path+ "/{}".format(NAME))
            model.compile(loss='sparse_categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        )

            model.fit(X, y,
                    batch_size=32,
                    epochs=50,
                    validation_split=0.2,
                    callbacks=[tensorboard])

print("----------\nTraining Complete\n----------")
print("Total training time {:.2f}s".format(time() - start))
print(model.summary())
model.save(dir_path + '3CNN-5Start-3Kernel.model')