#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read Logs

# In[2]:


def readLog(csv_file):
    log = pd.read_csv(csv_file, header=None)
    for i in range(3):
        log.iloc[:,i] = log.iloc[:,i].str.replace('/home/jianjian/Nutstore/Self-Driving/Project4_research/IMG', '../IMG_' + csv_file.split('_')[0][3:])
        log.iloc[:,i] = log.iloc[:,i].str.replace('/home/jianjian/Nutstore/Self-Driving/Project4/IMG', '../IMG_' + csv_file.split('_')[0][3:])
        
    log = np.array(log)
    return log

# Build the list of filenames of images
logs = []
logs.extend(readLog('../bCenter_driving_log.csv'))
logs.extend(readLog('../aCenter_driving_log.csv'))
# logs.extend(readLog('../bFast_driving_log.csv'))
logs.extend(readLog('../bReverse_driving_log.csv'))
# logs.extend(readLog('../aFast_driving_log.csv'))
logs.extend(readLog('../aReverse_driving_log.csv'))
# logs.extend(readLog('../bRecover_driving_log.csv'))
logs.extend(readLog('../bRecover2_driving_log.csv'))
logs.extend(readLog('../aRecover_driving_log.csv'))
logs = np.array(logs)

valid_logs = []
valid_logs.extend(readLog('../valid_driving_log.csv'))
valid_logs = np.array(valid_logs)

test_logs = []
test_logs.extend(readLog('../test_driving_log.csv'))
test_logs = np.array(test_logs)


# ## Build Train/ Validation/ Test Set

# In[3]:


X_train = logs[:, 0]
y_train = logs[:, 3]

X_valid = valid_logs[:, 0]
y_valid = valid_logs[:, 3]

X_test = test_logs[:, 0]
y_test = test_logs[:, 3]

del logs
del test_logs


# ## Accumulate MEAN and STD for preprocessing

# In[4]:


# Because training set is too big, I calculated it partly and summed it up
mean_sum = 0
cal_batch = 500
temp_batch = []
for ind, file in enumerate(X_train):
    img = mpimg.imread(file)
    temp_batch.append(img)
    if (ind + 1) % cal_batch == 0:
        mean_sum += np.sum(temp_batch, axis=0)
        temp_batch = []
mean = mean_sum / len(X_train)
np.save('mean', mean)

std = 0
temp_batch = []
for ind, file in enumerate(X_train):
    img = mpimg.imread(file)
    temp_batch.append(img)
    if (ind + 1) % cal_batch == 0:
        std += np.sum(np.square(temp_batch - mean), axis=0) / len(X_train)
        temp_batch = []

std = np.sqrt(std)
np.save('std', std)


# In[5]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Cropping2D, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils import Sequence
from keras import regularizers, optimizers


# ## Network Architecture

# In[6]:


# Lambda layer for preprocessing
model = Sequential()
model.add(Lambda(lambda x: (x-mean)/std, input_shape=(160, 320, 3)))
# Cropped part of images
model.add(Cropping2D(cropping=((63, 25), (0, 0))))

# CNN layers with ReLU and BN
model.add(Convolution2D(24,(5,5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Convolution2D(36,(5,5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Convolution2D(48,(5,5), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(BatchNormalization())
model.add(Convolution2D(64,(3,3), activation="relu"))
model.add(BatchNormalization())

# Applied l2 regularization for not overfitting
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(50, activation="relu",
                kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Dense(10, activation="relu",
                kernel_regularizer=regularizers.l2(0.02)))
model.add(BatchNormalization())
model.add(Dense(1, activation=None))


# ## Utils for Training

# In[7]:


from sklearn.utils import shuffle

# Implemented generator
class MY_Generator(Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = shuffle(image_filenames, labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([mpimg.imread(file_name)
               for file_name in batch_x]), np.array(batch_y)


# ## Training

# In[8]:


my_training_batch_generator = MY_Generator(X_train, y_train, 64)
my_valid_batch_generator = MY_Generator(X_valid, y_valid, 64)

# MSE as loss function
# Adam as optimizor
model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator=my_training_batch_generator, epochs=4, shuffle=True, validation_data=my_valid_batch_generator, use_multiprocessing=True, workers=12, max_queue_size=1)

try:
    model.save('model4.h5')
except:
    model.save('model_temp4.h5')
    
model.fit_generator(generator=my_training_batch_generator, initial_epoch=4, epochs=6, shuffle=True, validation_data=my_valid_batch_generator, use_multiprocessing=True, workers=12, max_queue_size=1)

try:
    model.save('model6.h5')
except:
    model.save('model_temp6.h5')
    
model.fit_generator(generator=my_training_batch_generator, initial_epoch=6, epochs=7, shuffle=True, validation_data=my_valid_batch_generator, use_multiprocessing=True, workers=12, max_queue_size=1)

try:
    model.save('model7.h5')
except:
    model.save('model_temp7.h5')


# ## Test

# In[19]:


from keras.models import load_model

# model6 is the best model (through practical test)
my_test_batch_generator = MY_Generator(X_test, y_test, 64)
model = load_model('model6.h5', custom_objects={'mean': mean, 'std': std})
test_loss = model.evaluate_generator(my_test_batch_generator, max_queue_size=3)


# In[24]:


print("Loss of test set is %0.4f" % (test_loss))


# In[ ]:




