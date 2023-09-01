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
from keras.optimizers import Adam
from keras.optimizers import SGD
#from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy
import time
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import os

parent_path = os.path.dirname(os.path.abspath(__file__))
#from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
#from kerastuner import HyperParameters
#import tensorflow as tf
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#PARAMETERS

images = np.load(f"{parent_path}/TrainingData.npy")
labels = np.load(f"{parent_path}/TrainingLabels.npy")

orig_train_images = images[:45000]
valid_images = images[45000:]

orig_train_labels = labels[:45000]
valid_labels = labels[45000:]



datagen = ImageDataGenerator(
    rotation_range=20,      # Rotate images by 20 degrees
    width_shift_range=0.2,  # Shift images horizontally by 20% of total width
    height_shift_range=0.2, # Shift images vertically by 20% of total height
    shear_range=0.2,        # Apply shear transformation
    zoom_range=0.05,         # Zoom in/out on images by 5%
    horizontal_flip=True,   # Flip images horizontally
    vertical_flip=False)     # Flip images vertically



augmented_images = []
augmented_labels = []


for i in range(3):
    for image, label in zip(orig_train_images,orig_train_labels):
        batch = image.reshape((1,) + image.shape)
        label_batch = label.reshape((1,) + label.shape)
        augmented_batch = datagen.flow(batch, label_batch, batch_size=1)[0]
        augmented_images.extend(augmented_batch[0])
        augmented_labels.extend(augmented_batch[1])



    train_images = np.concatenate((orig_train_images, augmented_images), axis=0)
    train_labels = np.concatenate((orig_train_labels, augmented_labels), axis=0)

print("length: " + str(np.shape(train_labels)[0]))

epochnum=3
row1filt = 84
row2filt = 105
row3filt = 96 #determines a new convolutional layer
padding1 = "same"
padding2 = "same"
padding3 = "valid" #dependent on row3filt
Dense1 = 91 
Dense2 = "None" #determines a new dense layer
finact = "softmax"


#optimizer must be changed manually so far


if (row3filt == "None"):
    padding3 = "None"




"""SET UP YOUR COMPUTER TO OPTIMIZE IT A WHOLE BUNCH OF ITMES UNTIL RETURNS ARE ESSENTIALLY 0 AND THEN AFTER THAT HAS PLATEAUD, RERANDOMIZE ALL VARIABLES BESIDES EPOCHS AND REPEAT THE PROCESS
"""

#train_images = np.load("TrainingData.npy")
#train_labels = np.load("TrainingLabels.npy")




#print(len(train_images))
#print(len(valid_images))








#lr_schedule = ExponentialDecay(0.01, 10, 0.999)
lr_schedule = 0.0001


def build_model():
        
    model = models.Sequential()
    model.add(layers.Conv2D(row1filt, (3, 3), activation='relu', padding=padding1, input_shape=(32, 32, 3)))#order is # of filters, kernel size, activation function, image shape.
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.40))
    model.add(layers.Conv2D(row2filt, (3, 3), activation='relu', padding=padding2))
    if (row3filt != "None"):
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(row3filt, (3, 3), activation='relu', padding=padding3, input_shape=(32, 32, 3)))#order is # of filters, kernel size, activation function, image shape.
    model.add(layers.Flatten())
    model.add(layers.Dense(Dense1, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    if (Dense2 != "None"):
        model.add(layers.Dense(Dense2, activation='relu'))
    model.add(layers.Dense(10, activation=finact))

    model.compile(Adam(learning_rate=lr_schedule), loss="categorical_crossentropy", metrics=['accuracy'])




    """
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    """



    # Define the callbacks
    class CustomCallback(callbacks.Callback):


        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            # Access the necessary information from logs
            #num_filters = self.model.layers[0].filters
            #padding = self.model.layers[0].padding
            current_epoch = epoch + 1
            val_accuracy = logs['val_accuracy']
            train_accuracy = logs['accuracy']
            val_loss = logs['val_loss']
            train_loss = logs['loss']
            optimizer = self.model.optimizer.__class__.__name__

            # Print the collected information
            #print('Number of Convolutional Filters:', num_filters)
            #print('Padding:', padding)
            #print('Epoch:', current_epoch)
            #print('Validation Accuracy:', val_accuracy)
            #print('Training Accuracy:', train_accuracy)
            #print('Validation Loss:', val_loss)
            #print('Training Loss:', train_loss)
            #print('Optimizer:', optimizer)
            #if (current_epoch == epochnum):
                #print("working")
            with open("traininglog.csv", "a", newline="") as f:
                resultlist = [train_accuracy, val_accuracy, train_loss, val_loss, current_epoch, epoch_time, row1filt, row2filt, row3filt, padding1, padding2, Dense1, Dense2, optimizer, finact]
                f.write("\n")
                writer(f).writerow(resultlist)
                

    # Create an instance of the callback










    model.fit(train_images, train_labels, epochs=epochnum, validation_data=(valid_images,valid_labels), callbacks=[CustomCallback()])#,checkpoint])#, validation_data=(test_images, test_labels))


build_model()



"""DATA

Later filter and Padding, Layer filter and Padding, Layer filter and Padding, Layer filter and Padding, epochs, accuracy, loss, optimizer

64 valid, 64 valid, 64 valid, None, 15, 0.6927, 0.8856, ADAM

32 valid, 64 valid, 64 valid, None, 15, 0.6813, 0.9196, ADAM

16 valid, 64 valid, 64 valid, None, 15, 0.6419, 1.0239, ADAM

32 valid, 64 valid, 64 valid, None, 6, 0.5801, 1.1906, ADAM

32 valid, 64 same, 64 same, None, 6, 0.6225, 1.0823, ADAM

32 same, 64 same, 64 same, None, 6, 0.6405, 1.0365, ADAM

32 same, 64 same, 64 same, 64 same, 6, 0.6299, 1.0763, ADAM

32 same, 64 same, None, None, 6, 0.6460, 1.0196, ADAM

64 same, 64 same, None, None, 15, 0.7576, 0.7095, ADAM

64 valid, 64 valid, None, None, 15, 0.7418, 0.7500, ADAM

32 valid, 64 valid, 32 valid, None, 6, 0.5396, 1.3012, ADAM

32 valid, 32 valid, 64 valid, None, 6, 0.5396, 1.2974, ADAM

32 valid, 32 valid, 32 valid, None, 6, 0.5281, 1.3279, ADAM



64 same, 64 same







Later filter and Padding, Layer filter and Padding, Layer filter and Padding, Layer filter and Padding, epochs, accuracy and valid accuracy, loss and valid loss, optimizer




"""







#All of the excess code was stripped, original copies and stray information is below

"""

import numpy as np
#import random
#import keras
from keras import datasets,layers,models
import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy




train_images = np.load("TrainingData.npy")
train_labels = np.load("TrainingLabels.npy")


#print(train_images.shape)
#print(type(train_labels))
#print(np.shape(train_images))

#train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[np.argmax(train_labels[i])])
    #print(train_labels[i])
    #print(np.argmax(train_labels[i]))



plt.show()



#print(np.shape(train_images))

n=64
print("Number of Initial Filters: " + str(n))



model = models.Sequential()
model.add(layers.Conv2D(n, (3, 3), activation='relu', input_shape=(32, 32, 3)))#order is # of filters, kernel size, activation function, image shape.
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation="softmax"))



model.compile(Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])



#print(train_labels[0:5])

model.fit(train_images, train_labels, epochs=15)#, validation_data=(test_images, test_labels))










"""


"""
NOTES FOR CNN BUILDING

3x3 is optimal convolution size

'a great rule of thumb is to start off with a relu' ??

If the code is having trouble, potentially look into "mean normalization" down the road

If you end up overfitting, read the section on that in the first article, or simply research "dropout"

SGD might be a good replacement for Adam as an optimization algorithm when finetuning













print ("ok")





"""
















"""
from keras.applications import vgg16
from keras.models import load_model

original_model = vgg16.VGG16()



model = Sequential()
for layer in original_model.layers:
    model.add(layer)

model.save("vgg16_CIFAR10.h5")



















scaled_train_samples=[]
train_labels=[]
##ISSUE IS I NEVER NORMALIZED THE DATA

for i in range(2000):
    
    k = random.randint(0,100)
    scaled_train_samples.append(k)

    if (k<70):
        train_labels.append(1)
    else:
        train_labels.append(0)

#Creates an array of ages between 0 and 100, and for the corresponding labels, anyone under 70 will recieve a 1

#NOW REMOVING A PORTION FOR THE VALIDATION SET
scaled_valid_samples = np.array(scaled_train_samples[1800:])
valid_labels = np.array(train_labels[1800:])


#valid_set = np.array(tuple(zip(scaled_valid_samples,valid_labels)))

#print((valid_set))


del scaled_train_samples[1800:]
del train_labels[1800:]





train_labels = np.array(train_labels)
scaled_train_samples = np.array(scaled_train_samples)










#Above I created the data set











#Defining Dense Layers
l1 = Dense(16, input_shape=(1,),activation="relu")

l2 = Dense(32,activation="relu")

l3 = Dense(2, activation="softmax")



#Creating the Model
model = Sequential([l1,l2,l3])
#model.add(l4) 


model.compile(Adam(learning_rate=0.0001), loss = "sparse_categorical_crossentropy", metrics=['accuracy'])



model.fit(scaled_train_samples, train_labels, batch_size=5, validation_data=(scaled_valid_samples, valid_labels), epochs=25, shuffle=True, verbose = 0)#validation_split=0.1 also works, this would remove
#10% of the data from the training samples to create a validation set

predicting_set = np.array([12,52,23,79,72,67,-2,102,-14,123])

predictions = model.predict(predicting_set, batch_size=1, verbose=2)
#predictions = model.predict_classes(predicting_set, batch_size=1, verbose=2) "DEPRICATED"


print(predictions)

#remember to save and load so you don't have to create a new network each time: import keras load funciton, and model.save

"""
"""
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict




print(type(unpickle("Testing Data/batches.meta")))
"""