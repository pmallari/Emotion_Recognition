import numpy as np
import os
from PIL import ImageGrab
import cv2
import time
import tensorflow as tf
import random
random.seed(42)

# Define Emotion Dictionary
#{0: 'HO', 1: 'A', 2:'HC', 3:'F'}
#{'A', 'HC', 'N', 'HO', 'F'}
Emotion = {0:'Happy',
           1:'Anger',
           2:'Happy',
           3:'Fear',
           4:'Neutral',
		   }

# Process image from BGR to Gray
def process_img(original_image):
	processed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	processed_img = cv2.resize(processed_img, dsize = (512,512), interpolation = cv2.INTER_CUBIC)
	processed_img = np.asarray(processed_img).reshape(-1, 512, 512, 1)
	#processed_img = cv2.Canny(processed_img, threshold1 = 100, threshold2 = 200)
	return processed_img

# Import model architecture and model states
dir_path = os.path.dirname(os.path.realpath(__file__))
model1 = tf.keras.models.load_model(dir_path + "/model/NeutralModel.model")
model2 = tf.keras.models.load_model(dir_path + "/model/EmotionModel.model")

# Define a starting emotion
curr_emotion = "Neutral"
# Gather emotions and get mode to avoid randomness
emotion_ave = []
# Get current time
last_time = time.time()

while(True):
	#Grabs image from screen
	screen = np.array(ImageGrab.grab(bbox = (300, 300, 900, 900)))
	processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

	# Convert iamge to tensor and predict the emotion
	pred = model1.predict(process_img(processed_img))
	
	if pred == 1:
		pred = model2.predict(process_img(processed_img))
		pred = np.argmax(pred[0])
	else:
		pred = 4
	cv2.putText(processed_img, Emotion[pred], (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0))
	cv2.imshow('window2', processed_img)

	#print('Frame rate of {:.3f}s.'.format(1/(time.time() - last_time)))
	#cv2.imshow('window', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)) #Original Screen
	if cv2.waitKey(25) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
