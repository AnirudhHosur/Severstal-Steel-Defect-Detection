# Importing the libraries

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm

# Data Pre-processing
train = pd.read_csv('train.csv')

# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('train/'+train['ImageId'][i], target_size=(128,128,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
y=train['ClassId'].values


# Creating a validation set from training data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# CNN MODEL, a simple architecture with 2 convolutional layers, one dense hidden layer and an output layer
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(128,128,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax')) #5 because the categories are 5

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# One hot encoding our categorical variables

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Make predictions - Lets head onto test data
test = pd.read_csv('C:/Users/super/Desktop/ML_Projects/Surface Inspection/severstal-steel-defect-detection/test/test.csv')

# Read and store al the test images
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('test/'+test['ImageId'][i], target_size=(128,128,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)

# making predictions
prediction = model.predict_classes(test)

# THANKS A LOT 

