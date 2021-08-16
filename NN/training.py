###################################################################################
#                       Imports and Initializations                               #
###################################################################################

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import warnings # remove FutureWarnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.python.client import device_lib
    from tensorflow.keras import backend as K
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check GPU
print("Tensorflow version:",tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# Helper libraries
import numpy as np
from os import listdir, system
import cv2
import sys
from random import randrange, seed
import time

now = int(round(time.time() * 1000))
seed(now)

# from mlxtend.data import loadlocal_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Convolution2D

with warnings.catch_warnings():  # remove RuntimeWarnings
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input

###################################################################################
#                               F1 Score                                          #
###################################################################################

# Accuracy is used when the True Positives and True negatives are more important 
# while F1-score is used when the False Negatives and False Positives are crucial

# Compute recall: 
#   It is the measure of the correctly identified positive cases 
#   from all the actual positive cases. 
#   It is important when the cost of False Negatives is high.
# = TP / (TP + FN)
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

# Compute precision:
#   It is implied as the measure of the correctly 
#   identified positive cases from all the predicted positive cases. 
#   Thus, it is useful when the costs of False Positives is high.
# = TP / (TP + FP)
def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

# Compute F1 score:
#   This is the harmonic mean of Precision and Recall 
#   and gives a better measure of the incorrectly classified cases 
#   than the Accuracy Metric.
# = 2 * (Precision*Recall)/(Precision+Recall)
def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# build the dataset from the folder or it loads it if it has been already precomputed
# cut: used to cut the dataset by the specidied percentage
# load: 1 if we want to just load the dataset, not to build it
# train: 1 if we want to start the training procedures
def load_dataset(cut=50, load=0, train=1, img_shape=112):
     
    if  load == 0:
        #################
        # Build Dataset #
        #################

        # 2 different folders: 1 for closed eyes, 1 for opened eyes
        # Authoritatively I choose the labels: 0 closed, 1 open (or should I use one-hot encoding?)
        open_path = "../datasets/mrlEyes/Open/"
        closed_path = "../datasets/mrlEyes/Close/"
        # arrays to store train images and labels
        ims = []
        labs = []
        
        counter = 0

        # There are actually too many samples...
        # Let's take a random half

        # let's start with the closed ones
        for filename in listdir(closed_path):
            # rnd = randrange(100)
            # if rnd < cut:
            #     continue
            x = cv2.imread(closed_path + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(0)
            counter+=1

        for filename in listdir(open_path):
            # rnd = randrange(100)
            # if rnd < cut:
            #     continue
            x = cv2.imread(open_path + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(1)
            counter+=1

        np_ims = np.zeros((counter, img_shape, img_shape))
        for i in range(counter):
            np_ims[i] = ims[i]
        np_ims = np.array(np_ims).reshape((-1, img_shape, img_shape, 1), order="F")
        np_labs = np.array(labs).reshape((-1))
        np.save("saved_models/images_set_112", np_ims)
        np.save("saved_models/labels_set_112", np_labs)
    elif load == 1:

        ################
        # Load Dataset #
        ################

        np_ims = np.load("saved_models/images_set_new.npy")
        np_labs = np.load("saved_models/labels_set_new.npy")
        counter = np_labs.shape[0]

    ############################################################################
    #                            Train & Test                                  #
    ############################################################################

    if train == 1:

        # now that I built the dataset, I have to split it into train and test:
        # shuffle the 2 arrays
        perm = np.random.permutation(np_labs.shape[0])
        np_ims = np_ims[perm]
        np_labs = np_labs[perm]

        # train-test: 80-20
        train_size = round(counter/100*80)
        test_size = counter - train_size

        train_images = np_ims[:train_size]
        train_labels = np_labs[:train_size]
        test_images = np_ims[train_size:]
        test_labels = np_labs[train_size:]

        # little check
        print("Dataset Split Done:")
        print(train_images.shape)
        print(train_labels.shape)
        print(test_images.shape)
        print(test_labels.shape)
            
        return train_images, train_labels, test_images, test_labels

def load_mixed(cut=50, load=0, train=1, img_shape=96):
     
    if  load == 0:
        #################
        # Build Dataset #
        #################

        # 2 different folders: 1 for closed eyes, 1 for opened eyes
        # Authoritatively I choose the labels: 0 closed, 1 open (or should I use one-hot encoding?)
        open_path1 = "../datasets/mrlEyes/Open/"
        closed_path1 = "../datasets/mrlEyes/Close/"
        open_path2 = "../datasets/mrlEyesBad/Open/"
        closed_path2 = "../datasets/mrlEyesBad/Close/"
        open_path3 = "../datasets/BadDatasetEyes/Open/"
        closed_path3 = "../datasets/BadDatasetEyes/Close/"

        cut1 = 65       # close/open
        cut21 = 10      # close
        cut22 = 15      # open
        cut31 = 50      # close
        cut32 = 40      # open

        # arrays to store train images and labels
        ims = []
        labs = []
        
        counter = 0

        for filename in listdir(closed_path1):
            rnd = randrange(100)
            if rnd > cut1:
                continue
            x = cv2.imread(closed_path1 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(0)
            counter+=1

        for filename in listdir(open_path1):
            rnd = randrange(100)
            if rnd > cut1:
                continue
            x = cv2.imread(open_path1 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(1)
            counter+=1

        for filename in listdir(closed_path2):
            rnd = randrange(100)
            if rnd > cut21:
                continue
            x = cv2.imread(closed_path2 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(0)
            counter+=1

        for filename in listdir(open_path2):
            rnd = randrange(100)
            if rnd > cut22:
                continue
            x = cv2.imread(open_path2 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(1)
            counter+=1

        for filename in listdir(closed_path3):
            rnd = randrange(100)
            if rnd > cut31:
                continue
            x = cv2.imread(closed_path3 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(0)
            counter+=1

        for filename in listdir(open_path3):
            rnd = randrange(100)
            if rnd > cut32:
                continue
            x = cv2.imread(open_path3 + "/" + filename,0)
            ims.append(cv2.resize(x, (img_shape,img_shape), cv2.INTER_CUBIC))
            labs.append(1)
            counter+=1

        np_ims = np.zeros((counter, img_shape, img_shape))
        for i in range(counter):
            np_ims[i] = ims[i]
        np_ims = np.array(np_ims).reshape((-1, img_shape, img_shape, 1), order="F")
        np_labs = np.array(labs).reshape((-1))
        np.save("saved_models/images_set_mixed", np_ims)
        np.save("saved_models/labels_set_mixed", np_labs)

    elif load == 1:

        ################
        # Load Dataset #
        ################

        np_ims = np.load("saved_models/images_set_mixed.npy")
        np_labs = np.load("saved_models/labels_set_mixed.npy")
        counter = np_labs.shape[0]

    ############################################################################
    #                            Train & Test                                  #
    ############################################################################

    if train == 1:

        # now that I built the dataset, I have to split it into train and test:
        # shuffle the 2 arrays
        perm = np.random.permutation(np_labs.shape[0])
        np_ims = np_ims[perm]
        np_labs = np_labs[perm]

        # train-test: 80-20
        train_size = round(counter/100*80)
        test_size = counter - train_size

        train_images = np_ims[:train_size]
        train_labels = np_labs[:train_size]
        test_images = np_ims[train_size:]
        test_labels = np_labs[train_size:]

        # little check
        print("Dataset Split Done:")
        print(train_images.shape)
        print(train_labels.shape)
        print(test_images.shape)
        print(test_labels.shape)
            
        return train_images, train_labels, test_images, test_labels


# Using a model from me
def model2(load=0, img_shape=112):
    x_train, y_train, x_test, y_test = load_dataset(load=load)
    batch_size = 32
    datagen = ImageDataGenerator(
        rotation_range=3,  
        zoom_range = 0.05,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=False)
    epochs = 5
    
    # train set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded)
    
    # test set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_test)    
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded)
    
    print("Size of:")
    print("- Training-set:\t\t{}".format(x_train.shape[0]))
    print("- Test-set:\t\t{}".format(x_test.shape[0]))

    num_classes = 2
    class_names = ['Closed', 'Open']

    # Data preprocessing: image normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0


    if K.image_data_format() == "channel_first":
        x_train = x_train.reshape(x_train_shape[0], 1, img_shape, img_shape)
        x_test = x_test.reshape(x_test.shape[0], 1, img_shape, img_shape)
        input_shape = (1, img_shape, img_shape)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_shape, img_shape, 1)
        x_test = x_test.reshape(x_test.shape[0], img_shape, img_shape, 1)
        input_shape = (img_shape, img_shape, 1)
    
    # building the network
    model = Sequential()
    #Conv-Pool-Conv-Pool or Conv-Conv-Pool-Conv-Conv-Pool or trend in the Number of channels 
    # 32–64–128 or 32–32-64–64 or trend in filter sizes, Max-pooling parameters etc.
    model.add(Conv2D(16, kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    model.summary()
   
    # fit the model and save the history information
    history = model.fit(x_train, onehot_encoded_train,
        epochs=epochs,
        verbose=1,
        batch_size=batch_size,
        validation_data=(x_test, onehot_encoded_test))
    # history = model.fit(datagen.flow(x_train,onehot_encoded_train, batch_size=batch_size),
    #     epochs=epochs,
    #     verbose=1)
        # validation_data=(x_test, onehot_encoded_test))
    score = model.evaluate(x_test, onehot_encoded_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    predictions = model.predict(x_test, verbose=1)
    error_counter = 0
    for i,p in enumerate(predictions):
        if p[0] > p[1]:
            x = 0
        else:
            x = 1
        if x != y_test[i]:
            error_counter += 1

    print("Error counter:", error_counter)
    print("Error rate:", error_counter / predictions.shape[0])

    # serialize model to JSON --> save network model
    model_json = model.to_json()
    with open("saved_models/naive_model2.json", "w") as json_file:
       json_file.write(model_json)
    # serialize weights to HDF5 --> save network weights
    model.save_weights("saved_models/naive_model2_weights_datagen.h5")
    print("Saved model to disk")
    
    keras.backend.clear_session()

# Using LeNet (or something similar)
def model3(load=0):
    img_shape = 96 # size of the eyes images
    x_train, y_train, x_test, y_test = load_dataset(load=load)
    batch_size = 64
    datagen = ImageDataGenerator(
        rotation_range=5,  
        zoom_range = 0.15,  
        width_shift_range=0.3, 
        height_shift_range=0.3,
        horizontal_flip=False)

    epochs = 5
    
    # train set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded)
    
    # test set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_test)    
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded)
    
    print("Size of:")
    print("- Training-set:\t\t{}".format(x_train.shape[0]))
    print("- Test-set:\t\t{}".format(x_test.shape[0]))

    num_classes = 2
    class_names = ['Closed', 'Open']

    # Data preprocessing: image normalization
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Create Keras model and evaluate its performance
    img_rows, img_cols = img_shape, img_shape

    if K.image_data_format() == "channel_first":
        x_train = x_train.reshape(x_train_shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    # building the network
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (img_shape, img_shape, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.summary()
   
    # fit the model and save the history information
    # history = model.fit(x_train, onehot_encoded_train,
    #     epochs=epochs,
    #     verbose=1,
    #     batch_size=batch_size,
    #     validation_data=(x_test, onehot_encoded_test))
    history = model.fit(datagen.flow(x_train,onehot_encoded_train, batch_size=batch_size),
        epochs=epochs,
        verbose=1)
        # validation_data=(x_test, onehot_encoded_test))
    score = model.evaluate(x_test, onehot_encoded_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    predictions = model.predict(x_test, verbose=1)
    error_counter = 0
    for i,p in enumerate(predictions):
        if p[0] > p[1]:
            x = 0
        else:
            x = 1
        if x != y_test[i]:
            error_counter += 1

    print("Error counter:", error_counter)
    print("Error rate:", error_counter / predictions.shape[0])

    # serialize model to JSON --> save network model
    model_json = model.to_json()
    with open("saved_models/naive_model3.json", "w") as json_file:
       json_file.write(model_json)
    # serialize weights to HDF5 --> save network weights
    model.save_weights("saved_models/naive_model3_weights_datagen.h5")
    print("Saved model to disk")
    
    keras.backend.clear_session()

# Implementing transfer learning over VGG19
def model_VGG_19(load=0, img_shape=96, num_classes=2, epochs=2, batch_size=128):
    class_names = ['Closed', 'Open']
    x_train, y_train, x_test, y_test = load_mixed(load=load)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # train set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_train)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded)
    
    # test set
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_test)    
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded)

    if K.image_data_format() == "channel_first":
        x_train = x_train.reshape(x_train_shape[0], 1, img_shape, img_shape)
        x_test = x_test.reshape(x_test.shape[0], 1, img_shape, img_shape)
        input_shape = (1, img_shape, img_shape)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_shape, img_shape, 1)
        x_test = x_test.reshape(x_test.shape[0], img_shape, img_shape, 1)
        input_shape = (img_shape, img_shape, 1)

    # Load VGG19 model
    vgg19 = VGG19(weights='imagenet', 
                include_top=False, 
                input_tensor=Input(shape=(96, 96, 3)))
    
    # Build model
    model = Sequential()
    # This layer is needed in order to keep using grayscale images
    model.add(Conv2D(3, kernel_size=(1, 1),     # kernel size is (1,1) because we just want to map our grayscale input to RGB (as in VGG19)
                    activation='relu',
                    input_shape=input_shape))

    # Add all VGG19 convolutional layers to our model
    for layer in vgg19.layers:
        layer.trainable=False
        model.add(layer)

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, 
                optimizer="adam", 
                metrics=['accuracy'])

    # fit the model and save the history information
    history = model.fit(x_train, onehot_encoded_train,
        epochs=epochs,
        verbose=1,
        batch_size=batch_size,
        validation_data=(x_test, onehot_encoded_test))

    # serialize model to JSON --> save network model
    model_json = model.to_json()
    with open("saved_models/vgg19_new.json", "w") as json_file:
       json_file.write(model_json)
    # serialize weights to HDF5 --> save network weights
    model.save_weights("saved_models/vgg19_new_mixed.h5")
    print("SAVED MODEL TO DISK")

    score = model.evaluate(x_test, onehot_encoded_test, verbose=1)  # returns the metrics value: accuracy and F1-score

    # Compute error rate
    predictions = model.predict(x_test, verbose=1)
    error_counter = 0
    for i,p in enumerate(predictions):
        if p[0] > p[1]:
            x = 0
        else:
            x = 1
        if x != y_test[i]:
            error_counter += 1
    print("Error counter:", error_counter)
    print("Error rate:", error_counter / predictions.shape[0])
    
    keras.backend.clear_session()


def main():
    if len(sys.argv) < 2:
        print("Wrong arguments")
        system('spd-say "EXIT"')
        exit()
    if (sys.argv[1] == '1'):        # obsolete: not implemented anymore
        model1()
    elif (sys.argv[1] == '2'):
        if len(sys.argv) == 3:
            if sys.argv[2] == "--load":     # to avoid creating a new dataset every time
                print("Load existing set")
                # model2(1)
                model3(1)
        else: # TODO: add probability param when building
            print("Build new set")
            # model2()
            model3()
    else:
        print("Wrong arguments")
    system('spd-say "DONE"')

# execute code
if __name__ == '__main__':
    print("\n\n\n\nSTART\n")
    # main()
    # load_dataset(train=0)
    # load_mixed(train=0)
    # model_VGG_19(load=1, epochs=3, batch_size=16)
    model2(load=0)