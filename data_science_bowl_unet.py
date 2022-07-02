# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
# Import Libraries
import os


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2

import tensorflow as tf
from keras import backend as K

from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator




train_path = 'C:/Users/naufa/OneDrive - Universiti Kebangsaan Malaysia/Desktop/shrdc ai technologist/deep learning/module/data-science-bowl-2018/data-science-bowl-2018-2/test'
test_path = 'C:/Users/naufa/OneDrive - Universiti Kebangsaan Malaysia/Desktop/shrdc ai technologist/deep learning/module/data-science-bowl-2018/data-science-bowl-2018-2/train'



#%%

train_dir = os.listdir(train_path)
test_dir = os.listdir(test_path)


#%%

X_train = np.zeros((len(train_dir), 256, 256, 3), dtype=np.uint8)
Y_train = np.zeros((len(train_dir), 256, 256, 1), dtype=bool)

X_test = np.zeros((len(test_dir), 256, 256, 3), dtype=np.uint8)


#%%

for i, name in enumerate(train_dir):
    path = train_path + name
    img_real = cv2.imread(path+'/images/'+ name +'.png')
    img_real = cv2.resize(img_real,(256,256))
    X_train[i] = img_real
    
    img_segment_full = np.zeros((256, 256 , 1), dtype=bool)
    segment_path = path+'/masks/'
    for name in os.listdir(segment_path):
        img_segment = cv2.imread(segment_path + name, 0)
        img_segment = cv2.resize(img_segment, (256, 256))
        img_segment = np.expand_dims(img_segment, axis=-1)
        img_segment_full = np.maximum(img_segment_full, img_segment)
    
    Y_train[i] = img_segment_full

for i, name in enumerate(test_dir):
    path = test_path + name
    img_real = cv2.imread(path+'/images/'+ name +'.png')
    img_real = cv2.resize(img_real, (256,256))
    X_test[i] = img_real

#%%

plt.figure(figsize=(15,8))
plt.subplot(121)
plt.imshow(X_train[2])
plt.title('Real image')
plt.subplot(122)
plt.imshow(Y_train[2])
plt.title('Segmentation image');

#%%

aug_gen_args = dict(shear_range = 0.2,
                    zoom_range = 0.2,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='reflect'
                   )

X_train_gen = ImageDataGenerator(**aug_gen_args)
y_train_gen = ImageDataGenerator(**aug_gen_args)
X_val_gen = ImageDataGenerator()
y_val_gen = ImageDataGenerator()

#%%

aug_image_real = X_train[5].reshape((1,)+X_train[1].shape)
aug_image_seg = Y_train[5].reshape((1,)+Y_train[1].shape)

#%%
aug_image_real_check = X_train_gen.flow(aug_image_real, batch_size=1, seed=17, shuffle=False)
aug_image_seg_check = y_train_gen.flow(aug_image_seg, batch_size=1, seed=17, shuffle=False)

#%%

plt.figure(figsize=(15,10))
plt.subplot(141)
plt.imshow(X_train[5])
plt.title("original")
i=2
for batch in aug_image_real_check:
    plt.subplot(14*10+i)
    plt.imshow(image.array_to_img(batch[0]))
    plt.title("augmented")
    i += 1
    if i % 5 == 0:
        break
#%%

plt.figure(figsize=(15,10))
plt.subplot(141)
plt.imshow(Y_train[5])
plt.title("original")
i=2
for batch in aug_image_seg_check:
    plt.subplot(14*10+i)
    plt.imshow(image.array_to_img(batch[0]))
    plt.title("augmented")
    i += 1
    if i % 5 == 0:
        break


#%%

train, val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=17)

#%%

X_train_gen.fit(train, augment=True, seed=17)
y_train_gen.fit(y_train, augment=True, seed=17)
X_val_gen.fit(val, seed=17)
y_val_gen.fit(y_val, seed=17)

X_train_generator = X_train_gen.flow(train, batch_size=16, seed=17, shuffle=False)
y_train_generator = y_train_gen.flow(y_train, batch_size=16, seed=17, shuffle=False)
X_val_generator = X_val_gen.flow(val, batch_size=16, seed=17, shuffle=False)
y_val_generator = y_val_gen.flow(y_val, batch_size=16, seed=17, shuffle=False)

train_generator = zip(X_train_generator, y_train_generator)
val_generator = zip(X_val_generator, y_val_generator)


#%%

def iou(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou = K.mean((intersection + 1) / (union + 1), axis=0)
    return iou

#%%

def mean_iou(y_true, y_pred):
    results = []   
    for t in np.arange(0.5, 1, 0.05):
        t_y_pred = tf.cast((y_pred > t), tf.float32)
        pred = iou(y_true, t_y_pred)
        results.append(pred)
        
    return K.mean(K.stack(results), axis=0)



#%%

def dice_loss(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + 1) / (union + 1), axis=0)
    return 1. - dice


#%%

inputs = Input((256, 256, 3))
s = tf.keras.layers.Lambda(lambda x: x/255.0)(inputs)

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

up6 = UpSampling2D(size=(2,2))(conv5)
up6 = concatenate([up6, conv4])
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

up7 = UpSampling2D(size=(2,2))(conv6)
up7 = concatenate([up7, conv3])
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

up8 = UpSampling2D(size=(2,2))(conv7)
up8 = concatenate([up8, conv2])
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

up9 = UpSampling2D(size=(2,2))(conv8)
up9 = concatenate([up9, conv1])
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

model = models.Model(inputs=[inputs], outputs=[conv10])

model.compile(optimizer=optimizers.Adam(learning_rate=2e-4), loss=dice_loss, metrics=mean_iou)

#%%

model.summary()

#%%

history = model.fit(train_generator,
                    steps_per_epoch=len(train)/8,
                    validation_data=val_generator,
                    validation_steps=len(val)/8,
                    epochs=25
                   )
#%%

loss = history.history['mean_iou']
val_loss = history.history['val_mean_iou']

plt.figure(figsize=(15,10))
plt.plot(loss, label='Train IOU')
plt.plot(val_loss,'--', label='Val IOU')
plt.title('Training and Validation mean IOU')
plt.yticks(np.arange(0.5, 1, 0.05))
plt.xticks(np.arange(0, 25))
plt.grid()
plt.legend();



#%%

train_pred = model.predict(train, verbose = 1)
val_pred = model.predict(val, verbose=1)

#%%

plt.figure(figsize=(15,10))
plt.subplot(131)
plt.imshow(train[1])
plt.title('Original image')
plt.subplot(132)
plt.imshow(np.squeeze(y_train[1]))
plt.title('Segmented image')
plt.subplot(133)
plt.imshow(np.squeeze(train_pred[1]))
plt.title('Predicted  image');

#%%

plt.figure(figsize=(15,10))
plt.subplot(131)
plt.imshow(val[3])
plt.title('Original image')
plt.subplot(132)
plt.imshow(np.squeeze(y_val[3]))
plt.title('Segmented image')
plt.subplot(133)
plt.imshow(np.squeeze(val_pred[3]))
plt.title('Predicted  image');

#%%

test_pred = model.predict(X_test, verbose=1)

#%%

plt.figure(figsize=(15, 32))
for i in range(421, 429):
    plt.subplot(i)
    if i % 2!=0:
        plt.imshow(X_test[i])
        plt.title('Original image')
    else:
        plt.imshow(np.squeeze(test_pred[i-1]))
        plt.title('Predicted image')



