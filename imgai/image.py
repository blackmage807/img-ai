from multiprocessing.spawn import prepare
from PIL import Image
from numpy import asarray
import numpy as np
from tensorflow import keras
from keras import layers
import tensorflow as tf

imageTrainingData = []
textTrainingData = []

#making image data
def prepareImageData(imagePath, imageTrainingData):
    # load the image
    image = Image.open(imagePath)
    # convert image to numpy array
    data = asarray(image)
    trueData = data.flat
    #append to the training data
    imageTrainingData.append(trueData)

#set up data
for i in open("imgpaths.txt", "r"):
    prepareImageData(i.strip(), imageTrainingData)

for i in open("text.txt", "r"):
    textTrainingData.append([int(i)])

#set up NN
inputs = layers.Input(shape=(3500000,))
layer1 = layers.Dense(250, activation="relu")(inputs)
layer2 = layers.Dense(250, activation="relu")(layer1)
layer3 = layers.Dense(250, activation="relu")(layer2)
layer4 = layers.Dense(250, activation="relu")(layer3)
layer5 = layers.Dense(250, activation="relu")(layer4)
outputs = layers.Dense(1)(layer5)
recognizationModel = keras.Model(input=inputs, output=outputs)

#compile and train model
recognizationModel.compile(optimizer="adam", loss="catigoricalCrossentropy")
for i in len(range(textTrainingData)):
    recognizationModel.fit(x=tf.convert_to_tensor(imageTrainingData[i]), y=tf.convert_to_tensor(textTrainingData[i]), verbose="auto", epochs=100 )

recognizationModel.save("RecogModel")