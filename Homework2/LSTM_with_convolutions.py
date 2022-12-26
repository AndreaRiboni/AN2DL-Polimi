# Choose which base network to use
build_1DCNN = False
build_BILSTM = False
build_LSTM = True
build_TRANSFORMER = False

#Boring stuff
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('font', size=16) 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import warnings
import logging
from sklearn.model_selection import train_test_split

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

#Dataset loading
x_train = np.load("data/dataset/x_train.npy") #shape: 2429, 36, 6
y_train = np.load("data/dataset/y_train.npy") #shape: 2429

#Test splitting
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=seed, test_size=.1)

# Map activities to integers
label_mapping = {
    "Wish": 0,
    "Another": 1,
    "Comfortably": 2,
    "Money": 3,
    "Breathe": 4,
    "Time": 5,
    "Brain": 6,
    "Echoes": 7,
    "Wearing": 8,
    "Sorrow": 9,
    "Hey": 10,
    "Shine": 11
}

# Convert the sparse labels to categorical values
y_train = tfk.utils.to_categorical(y_train)
y_test = tfk.utils.to_categorical(y_test)

model = tfk.models.load_model('LSTM')
model.summary()

# Prepare the training
input_shape = x_train.shape[1:]
classes = y_train.shape[-1]
batch_size = 64
epochs = 200

# Models
def build_LSTM_classifier(input_shape, classes): #88 vs 73 vs 73
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Convolutions as feature extractor
    x = tfkl.Conv1D(256, 3, padding='same', activation='relu')(input_layer)
    x = tfkl.Dropout(0.2)(x)
    x = tfkl.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tfkl.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tfkl.Dropout(0.2)(x)
    x = tfkl.Conv1D(96, 3, padding='same', activation='relu')(x)
    x = tfkl.MaxPooling1D()(x) #7243

    # Feature extractor
    x = tfkl.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = tfkl.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)(x)
    x = tfkl.LSTM(96, dropout=0.3, recurrent_dropout=0.3)(x)

    # Classifier
    x = tfkl.Dense(256, activation='relu')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Reshape(input_shape=(1, 256), target_shape=(32, 8))(x)
    x = tfkl.Conv1D(64, 1)(x) #(32, 1): 0.7737 test
    x = tfkl.Conv1D(64, 1)(x)
    x = tfkl.Conv1D(32, 1)(x)
    x = tfkl.Dropout(0.25)(x)
    x = tfkl.Flatten()(x)
    x = tfkl.Dense(128, activation='relu')(x)
    x = tfkl.BatchNormalization()(x)
    output_layer = tfkl.Dense(classes, activation='softmax')(x)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(), optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model

if build_LSTM:
    print("Creating LSTM")
    model = build_LSTM_classifier(
        input_shape, classes = classes
    )
    model.summary()

    # Train the model
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=.1,
        callbacks=[
            tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True),
            tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-5)
        ]
    ).history

    best_epoch = np.argmax(history['val_accuracy'])
    plt.figure(figsize=(17, 4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Categorical Crossentropy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['accuracy'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_accuracy'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    model.save('LSTM')

    # Predict the test set with the LSTM
    predictions = model.predict(x_test)

    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, cmap='Blues', xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()

    # Compute the classification metrics
    accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))
    precision = precision_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    print('Accuracy:', accuracy.round(4))
    print('Precision:', precision.round(4))
    print('Recall:', recall.round(4))
    print('F1:', f1.round(4))
    print("Done LSTM")