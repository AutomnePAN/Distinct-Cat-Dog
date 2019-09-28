#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

tf.enable_eager_execution()
tf.__version__

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# In[2]:


AUTOTUNE = tf.data.experimental.AUTOTUNE


# ## Collect the Data

# In[2]:


def get_files(file_dir):
    cat = []
    cat_label = []
    dog = []
    dog_label = []

    for file in os.listdir(file_dir):
        name = file.split()
        if "cat" in name[0]:
            cat.append(file_dir + file)
            cat_label.append(0)
        elif "dog" in name[0]:
            dog.append(file_dir + file)
            dog_label.append(1)

    image_list = np.hstack((cat, dog))
    label_list = np.hstack((cat_label, dog_label))
    print("There are %d dogs %d cats in the training set" % (len(dog), len(cat)))

    # Shuffle the training set
    temp_training_set = np.array([image_list, label_list])
    temp_training_set = temp_training_set.transpose()

    np.random.shuffle(temp_training_set)

    image_list = temp_training_set[:, 0]
    label_list = temp_training_set[:, 1]
    label_list = [int(i) for i in label_list]
    return image_list, label_list


def get_test_files(file_dir):
    test_set = [];
    for file in os.listdir(file_dir):
        test_set.append(file_dir + file)
    return test_set


print("Getting the training images and labels: ")
image, label = get_files("train_1/train/")
print("Training Set collected");
print("______________________________________________________________________")

#  The training label need no procession
training_labels = label;
print("training_labels prepared");
print("______________________________________________________________________")

# ### Set training images
SIZE_X = 128;
SIZE_Y = 128;

training_images = []
for i in range(len(image)):
    img = Image.open(image[i])
    img = img.resize((SIZE_X, SIZE_Y));
    img = np.array(img)
    if i % 1000 == 0:
        print("Proportion processed: ", 100 * i / len(image), "%");
    #     print(train_img)
    training_images.append(img)
plt.figure()
plt.imshow(training_images[5])
plt.colorbar()
plt.grid(False)
plt.show()
training_images = np.array(training_images)
training_images = training_images.reshape((len(training_images), SIZE_X, SIZE_Y, 3))
training_images = training_images / 255;
print("training images, shape: ", training_images.shape);
print("training_images prepared");
print("______________________________________________________________________")

# Set the class name: label == 0 => cat; label == 1 => dog;
class_names = ["Cats", "Dogs"]

print("Visualization of the training images");
plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])
plt.show()

# ### Set the Test Set

test = get_test_files("test/test/")
image_name_dict = {}
for i in range(len(test)):
    img = Image.open(test[i])
    img = img.resize((SIZE_X, SIZE_Y));
    img = np.array(img)
    if i % 1000 == 0:
        print("Proportion processed: ", i / len(test))
    #     print(train_img)
    image_name_dict[test[i]] = img

import re

test_images = [None for i in range(len(test))]
for key in image_name_dict.keys():
    k = int(re.findall("\d+", key)[0])
    # print(int(re.findall("\d+", key)[0]))
    test_images[k] = image_name_dict[key];
test_images = np.array(test_images)
test_images = test_images.reshape((len(test_images), SIZE_X, SIZE_Y, 3))
test_images = test_images / 255.0

# Visualization
plt.figure(figsize=(20, 20))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[training_labels[i]])
plt.show()

# ## Building the Model

def training():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(SIZE_X, SIZE_Y, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax'))
    adam = tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, shuffle=True, batch_size=64, epochs=4)
    return model

def Vote_pack(n):
    register = [];
    for i in range(len(test_images)):
        register.append([i, 0, 0]);

    for time in range(n):
        print("Start the ", time + 1, " training:");
        print("---------------------------------------------------------------------------------")
        model = training();

        print("Start the ", time + 1, " predicting:");
        print("---------------------------------------------------------------------------------")
        predict = model.predict(test_images)

        print("Start the ", time + 1, " register:");
        print("---------------------------------------------------------------------------------")
        for i in range(len(predict)):
            register[i][1] += predict[i][0];
            register[i][2] += predict[i][1];
    return register

result = [];
register = Vote_pack(32)
for i in range(len(register)):
    if register[i][1] >= register[i][2]:
        result.append([i, 0]);
    else:
        result.append([i, 1]);
import pandas as pd

result_pd = pd.DataFrame(result)
result_pd.columns = [["id", "label"]]
result_pd.to_csv("Result_vote.csv", index=False);
