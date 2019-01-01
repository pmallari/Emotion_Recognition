import numpy as np 
import pandas as pd
import pickle
import os
import cv2
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from collections import Counter
from sklearn.preprocessing import LabelBinarizer

# Reproducibility
np.random.seed(42)

# Declare current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path +"/data/"
image_path = dir_path + "/images/"

start = time()
img_size = 512

def create_training_data():
    orig_img_size = 1718
    splice_img = int((2444-1718)/2)

    # Create set for Neutral or Not
    # 0 for Neutral; 1 for Not Neutral
    img_neutral = []
    label_neutral = []

    # Create set for Non Neutral Emotions
    # {0: 'HO', 1: 'A', 2:'HC', 3:'F'}
    Emotion = {'HO':0,
               'A':1,
               'HC':2,
               'F':3
    }
    img_emotion = []
    label_emotion = []

    img_error = []

    for img in os.listdir(image_path):
        label = img[-img[::-1].index('-'):-img[::-1].index('.')-1]
        image = cv2.imread(image_path + img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, splice_img:-splice_img]
        image = cv2.resize(image, (img_size, img_size))
        image = np.asarray(image)
     
        
        crop_img = image.reshape(img_size, img_size, 1)
        crop_img = crop_img / 255.0
        
        try:
            if label == 'N':
                img_neutral.append(crop_img)
                label_neutral.append(0)
            else:
                img_neutral.append(crop_img)
                label_neutral.append(1)
                img_emotion.append(crop_img)
                label_emotion.append(Emotion[label])
        except Exception as e:
            print("Error")
            img_error.append([img])

    enc = LabelBinarizer()
    label_emotion = enc.fit_transform(label_emotion)
    print(label_emotion)

    img_neutral = np.asarray(img_neutral)
    label_neutral = np.asarray(label_neutral)
    img_emotion = np.asarray(img_emotion)
    label_emotion = np.asarray(label_emotion)

    print(img_neutral.shape, label_neutral.shape, 
          img_emotion.shape, label_emotion.shape)

    print("----------\nCreating Neutral Training Image Set...\n----------")
    pickling = open(data_path + "NeutralTestImage.pickle", "wb")
    pickle.dump(np.asarray(img_neutral), pickling)
    pickling.close()

    print("----------\nCreating Neutral Training Label Set...\n----------")
    pickling = open(data_path + "NeutralTestLabel.pickle", "wb")
    pickle.dump(np.asarray(label_neutral), pickling)
    pickling.close()
    
    print("----------\nCreating Emotion Training Image Set...\n----------")
    pickling = open(data_path + "EmotionTestImage.pickle", "wb")
    pickle.dump(np.asarray(img_emotion), pickling)
    pickling.close()
    
    print("----------\nCreating Emotion Training Label Set...\n----------")
    pickling = open(data_path + "EmotionLabel.pickle", "wb")
    pickle.dump(np.asarray(label_emotion), pickling)
    pickling.close()

print("----------\nCreating dataset...\n----------")
last_time = time()
create_training_data()
run_time = time() - last_time
print("----------\nDataset created\n----------")
print('Unpacking and data processing took {:.2f}s.'.format(run_time))