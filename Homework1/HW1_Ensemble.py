### BOILERPLATE CODE ###

import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from PIL import Image


tfk = tf.keras
tfkl = tf.keras.layers

# Random seed for reproducibility
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

tf.get_logger().setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

### LOADING THE DATASET ###

dataset_dir = 'data/dataset'
training_dir = os.path.join(dataset_dir, 'training')
validation_dir = os.path.join(dataset_dir, 'validation')

from keras.preprocessing.image import ImageDataGenerator

valid_data_gen = ImageDataGenerator(rescale=1/255.)

aug_train_data_gen = ImageDataGenerator(rotation_range=90,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='nearest',
                                        brightness_range=(0.2,1.8),
                                        rescale=1/255.) # rescale value is multiplied to the image

aug_train_gen = aug_train_data_gen.flow_from_directory(directory=training_dir,
                                                       target_size=(96,96),
                                                       color_mode='rgb',
                                                       classes=None, # can be set to labels
                                                       class_mode='categorical',
                                                       batch_size=16,
                                                       shuffle=True,
                                                       seed=seed)

valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                               target_size=(96,96),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=16,
                                               shuffle=False,
                                               seed=seed)

input_shape = (96, 96, 3)
epochs = 200

### LOADING THE TWO MODELS ###
models_dir = 'data/models'
efficient_dir = os.path.join(models_dir, 'EfficientNet')
xception_dir = os.path.join(models_dir, 'Xception')

efficient = tfk.models.load_model(efficient_dir)
print("EffNet loaded")

xception = tfk.models.load_model(xception_dir)
print("All models have been loaded")

### ENSEMBLE ###
input_eff = tf.keras.layers.Input(shape=input_shape)
input_xce = tf.keras.layers.Input(shape=input_shape)

#creating EfficientNet network
network_eff = efficient(input_eff)

#creating Xception network (rescale)
network_xce = tfkl.Rescaling(scale=1./255)(input_xce)
network_xce = xception(network_xce)

#creating the averaging layer
avg = tf.keras.layers.Average()([network_xce, network_eff])

#creating the ensemble network
ensemble_model = tf.keras.models.Model(inputs=[input_eff, input_xce], outputs=avg)

ensemble_model.save("SuperSayan")
