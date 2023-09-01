import numpy as np
#import random
#import keras
from keras import datasets,layers,models
import matplotlib.pyplot as plt
from csv import writer
from keras import callbacks
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras import regularizers
from keras.optimizers import Adam
#from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
import time
import keras_tuner as kt
import os

parent_path = os.path.dirname(os.path.abspath(__file__))

images = np.load(f"{parent_path}/TrainingData.npy")
labels = np.load(f"{parent_path}/TrainingLabels.npy")

train_images = images[:45000]
valid_images = images[45000:]

train_labels = labels[:45000]
valid_labels = labels[45000:]








def build_model(hp):
        
    model = models.Sequential()


    row1filt = hp.Int('row1filt', min_value=3, max_value=120, step=3)
    row2filt = hp.Int('row2filt', min_value=3, max_value=120, step=3)
    row3filt = hp.Int('row3filt', min_value=3, max_value=120, step=3)
    padding1 = hp.Choice('padding1', values=["same","valid"])
    padding2 = hp.Choice('padding2', values=["same","valid"])
    padding3 = hp.Choice('padding3', values=["same","valid"])
    
    
    dense1 = hp.Int("dense1", min_value=3, max_value=120, step=4)

 


    model.add(layers.Conv2D(row1filt, (3, 3), activation='relu', padding=padding1, input_shape=(32, 32, 3)))#order is # of filters, kernel size, activation function, image shape.
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.45))
    model.add(layers.Conv2D(row2filt, (3, 3), activation='relu', padding=padding2))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(row3filt, (3, 3), activation='relu', padding=padding3))
    model.add(layers.Flatten())
    model.add(layers.Dense(dense1, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])



    return model









tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=f"{parent_path}/hypertuner_dir",
                     project_name='NN1bTestON4')


stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)


tuner.search(train_images, train_labels, epochs=50, validation_data=(valid_images,valid_labels), callbacks=[stop_early])


best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]






model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=70, validation_data=(valid_images,valid_labels))

val_acc_per_epoch = history.history['val_accuracy']

best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(train_images, train_labels, epochs=best_epoch, validation_data=(valid_images,valid_labels))

#Add another layer and let it run , as well as another dense layer?