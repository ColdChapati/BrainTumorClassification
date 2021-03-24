# import libraries
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Activation
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# create list to store data
no_tumor = []
meningioma_tumor = []
pituitary_tumor = []
glioma_tumor = []

data = []
label = []

X_train = []
y_train = []

# get image data
for img in os.listdir('archive/meningioma_tumor'):
    try:
        image = cv2.imread(os.path.join('archive/meningioma_tumor', img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        label.append(0)
    except Exception as e:
        pass

for img in os.listdir('archive/pituitary_tumor'):
    try:
        image = cv2.imread(os.path.join('archive/pituitary_tumor', img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        label.append(1)
    except Exception as e:
        pass

for img in os.listdir('archive/glioma_tumor'):
    try:
        image = cv2.imread(os.path.join('archive/glioma_tumor', img), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (100, 100))
        data.append(image)
        label.append(2)
    except Exception as e:
        pass

# split data
X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.9)

# some funny shit
y_train = to_categorical(y_train,3)

# convert and reshape data
X_train = np.array(X_train).reshape(-1,100,100,1)
y_train = np.array(y_train)

# create model
model = Sequential()

# first layer
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(100,100,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))

# second layer
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

# third layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# output layer
model.add(Dense(3))
model.add(Activation('softmax'))

# print model summary
model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

# fit model
results = model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=2, validation_split=0.2)

# fit test data
# model.evaluate(X_test, y_test)

# plot accuracy vs epochs
plt.plot(results.history['val_accuracy'], label='val_accuracy', color='lightpink')
plt.plot(results.history['accuracy'], label='accuracy', color='c')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

# print loss vs epochs
plt.plot(results.history['val_loss'], label='val_loss', color='lightpink')
plt.plot(results.history['loss'], label='loss', color='c')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper left')
plt.show()
