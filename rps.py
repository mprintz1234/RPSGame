import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("Y.pickle","rb"))
XTest = pickle.load(open("XTest.pickle","rb"))
YTest = pickle.load(open("YTest.pickle","rb"))

X = X/255.0
model = Sequential()
#First, add the convolution layer to filter the raw pixel data
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
#Add activation layer to convert negative numbers to 0
model.add(Activation("relu"))
#Add pooling layer to find the max value in the 2x2 window
model.add(MaxPooling2D(pool_size = (2,2)))

#Add layer again
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(3))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(X,Y,epochs=3,batch_size=32, validation_split=0.1)

model.evaluate(XTest,YTest,batch_size=32)
