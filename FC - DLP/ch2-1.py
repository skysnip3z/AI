# -*- coding: utf-8 -*-
"""

@author: Ali Hassan
"""
#%% Imports

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers # Model Layers

#%% Loading MNIST dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% Inspecting the dataset - Training data

print(train_images.shape)

print(len(train_images))

print(train_images)

#%% Inspecting the dataset - Testing data

print(test_images.shape)

print(len(test_images))

print(test_images)

#%% Network Architecture - Layers

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
    ])

#%% Network Architecture - Compilation

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

#%% Preparing image data
# .reshape(rows, columns)
# .astype() change data type
# divide by 255 to bring the data to be between 0 and 1 e.g. 0.2324122
train_images_prepped = train_images.reshape((60000, 28 * 28))
train_images_prepped = train_images_prepped.astype("float32") / 255
test_images_prepped = test_images.reshape((10000, 28 * 28))
test_images_prepped = test_images_prepped.astype("float32") / 255

#%% Training the model through fit

model.fit(train_images_prepped, train_labels, epochs=5, batch_size=128)

#%% Utilising model to make predictions

test_digits = test_images_prepped[0:10] # 0-9 !10
predictions = model.predict(test_digits) 

print(predictions[0]) 
print(predictions[0].argmax()) # max optimum value in the given range
print(predictions[0][7]) # [row][column]

print(test_labels[0])

#%% Evaluating the model on new data

test_loss, test_acc = model.evaluate(test_images_prepped, test_labels)
print(f"test_acc: {test_acc}")




