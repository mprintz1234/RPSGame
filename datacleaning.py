from keras.utils import np_utils
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import pickle

#http://www.laurencemoroney.com/rock-paper-scissors-dataset/
DATADIR = '/Users/maximilian/Desktop/SideProjects/RockPaperScissors/rps'
DATADIRTEST = '/Users/maximilian/Desktop/SideProjects/RockPaperScissors/rps-test-set'
CATEGORIES = ["rock", "paper", "scissors"]
IMG_SIZE = 96
training_data = []
test_data = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    pathtest = os.path.join(DATADIRTEST, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            training_data.append([new_array, class_num])
        except Exception as e:
            pass

    for img in os.listdir(pathtest):
        try:
            test_array = cv2.imread(os.path.join(pathtest,img),cv2.IMREAD_GRAYSCALE)
            new_test_array = cv2.resize(test_array, (IMG_SIZE, IMG_SIZE))

            test_data.append([new_test_array, class_num])
        except Exception as e:
            pass

random.shuffle(training_data)
random.shuffle(test_data)

X=[]
Y=[]
XTest=[]
YTest=[]
for features, labels in training_data:
    X.append(features)
    Y.append(labels)

for features, labels in test_data:
    XTest.append(features)
    YTest.append(labels)

dummy_Y = np_utils.to_categorical(Y)
dummy_YTest = np_utils.to_categorical(YTest)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
XTest = np.array(XTest).reshape(-1,IMG_SIZE,IMG_SIZE,1)

print(dummy_YTest[0:10])
pickle_outX = open("X.pickle","wb")
pickle.dump(X, pickle_outX)
pickle_outX.close()

pickle_outY = open("Y.pickle","wb")
pickle.dump(dummy_Y, pickle_outY)
pickle_outY.close()

pickle_outXTest = open("XTest.pickle","wb")
pickle.dump(XTest, pickle_outXTest)
pickle_outXTest.close()

pickle_outYTest = open("YTest.pickle","wb")
pickle.dump(dummy_YTest, pickle_outYTest)
pickle_outYTest.close()
