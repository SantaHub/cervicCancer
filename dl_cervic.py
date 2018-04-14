#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:26:33 2018

@author: austin
"""
# cd /home/austin/ML/CervicCancer


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import os

# Internal Imports
from utilities import inout
from utilities import image_manipulation as imanip
from utilities import miscellaneous as misc

from itertools import count
import os

def read_paths(path, no_labels=False, label_type=None):
    # ** Takes a directory path and returns all of the file paths within the
    # directory, their labels, and the total number of classes. It uses the
    # subdirectories to create a corresponding label array **

    # path - string of path to the root directory
    # no_labels - optional boolean to use file
    #             names as labels instead of subdirectory
    # label_type - optional integer label for all paths to be read in

    file_paths = []
    labels = []
    labels_to_nums = dict()
    n_labels = None

    for dir_name, subdir_list, file_list in os.walk(path):
        if len(subdir_list) > 0:
            n_labels = len(subdir_list)
            for i,subdir in enumerate(subdir_list):
                labels_to_nums[subdir] = i
        else:
            type_ = dir_name.split('/')[-1]

        for img_file in file_list:
            if '.png' in img_file.lower():
                file_paths.append(os.path.join(dir_name,img_file))

                if no_labels: labels.append(img_file)
                elif type(label_type) is int: labels.append(label_type)
                else: labels.append(labels_to_nums[type_])

    if type(n_labels) is int: return file_paths, labels, n_labels
    else: return file_paths, labels, None
    
    

# Read in file paths of images to be resized
image_paths = []
labels = []

training_folders = ['data/train']
for folder in training_folders:
    new_paths, new_labels, n_classes = read_paths(folder)
    if len(new_paths) > 0:
        image_paths += new_paths
        labels += new_labels

image_paths, labels = shuffle(image_paths, labels)
histogram_dict = misc.histdict(labels, n_classes)

type1_count = histogram_dict[0]
type2_count = histogram_dict[1]

print("Number of Type 1 Images:", type1_count)
print("Number of Type 2 Images:", type2_count)
print("Total Number of data samples: " + str(len(image_paths)))
print("Number of Classes: " + str(n_classes))


## Resizing the image
new_img_shapes = [(299,299,3),(256,256,3)]
dest_folders = ['resized']
for new_img_shape,dest_folder in zip(new_img_shapes,dest_folders):
    for i,path,label in zip(count(),image_paths,labels):
        split_path = path.split('/')
        new_path = 'size_'+str(new_img_shape[0])+'_'+split_path[-1]
        new_path = '/'.join(['data/resized']+[str(label)]+[new_path])
        add_flip = True
        if label == 1:
            add_flip = False

        # Used to exclude corrupt data
        try:
            imanip.resize(path, maxsizes=new_img_shape,
                                save_path=new_path,
                                add_flip=add_flip)
        except OSError:
            print("Error at path " + path)

## Using resized image
resized_image_locations = ['data/resized/']

image_paths = []
labels = []
for i,root_path in enumerate(resized_image_locations):
    new_paths, new_labels, n_classes = read_paths(root_path)
    if len(new_paths) > 0:
        image_paths += new_paths
        labels += new_labels
        
    # Splitting the dataset
training_portion = .8
split_index = int(training_portion*len(image_paths))

X_train_paths, y_train = image_paths[:split_index], labels[:split_index]
X_valid_paths, y_valid = image_paths[split_index:], labels[split_index:]


y_train = imanip.one_hot_encode(y_train, n_classes)
y_valid = imanip.one_hot_encode(y_valid, n_classes)

## Creating generator. 
batch_size = 10
add_random_augmentations = True
resize_dims = None

n_train_samples = len(X_train_paths)
train_steps_per_epoch = misc.get_steps(n_train_samples,batch_size,n_augs=1)

n_valid_samples = len(X_valid_paths)
valid_steps_per_epoch = misc.get_steps(n_valid_samples,batch_size,n_augs=0)

train_generator = inout.image_generator(X_train_paths,y_train,batch_size,
                                        resize_dims=resize_dims,
                                        randomly_augment=add_random_augmentations)
valid_generator = inout.image_generator(X_valid_paths, y_valid, batch_size, 
                                        resize_dims=resize_dims,rand_order=False)

## Creating model
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense
from models import model as mod

n_classes = 2
image_shape = (299,299,3)

first_conv_shapes = [(4,4),(3,3),(5,5)]
conv_shapes = [(3,3),(5,5)]
conv_depths = [12,12,11,8,8]
dense_shapes = [100,50,n_classes]



inputs, outs = mod.cnn_model(first_conv_shapes, conv_shapes, conv_depths, dense_shapes, image_shape, n_classes)
model = Model(inputs=inputs,outputs=outs)
learning_rate = .0001
for i in range(20):
    if i > 4:
learning_rate = .01 # Anneals the learning rate
adam_opt = optimizers.Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
history = model.fit_generator(train_generator, train_steps_per_epoch, epochs=1,
                        validation_data=valid_generator,validation_steps=valid_steps_per_epoch, max_q_size=1)




### Drawing graphs

# summarize model for Accuracy
 plt.plot(history.history['acc'])
 plt.plot(history.history['val_acc'])
 plt.title('Model Accuracy')
 plt.ylabel('accuracy')
 plt.xlabel('epoch')
 plt.legend(['train', 'validate'], loc='upper left')
 # plt.savefig('model_name'+clf_type+'_acc.png')
 plt.show()

# summarize model for loss
 plt.plot(history.history['loss'])
 plt.plot(history.history['val_loss'])
 plt.title('Model Loss')
 plt.ylabel('loss')
 plt.xlabel('epoch')
 plt.legend(['train', 'validate'], loc='upper left')
 #plt.savefig('model_name'+clf_type+'_loss.png')
 plt.show()
