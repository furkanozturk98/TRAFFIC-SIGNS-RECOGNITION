import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
from PIL import Image
from keras.callbacks import History 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4     
path = []
data = []
label = []

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training,validation and testing sets
    labels = tf.keras.utils.to_categorical(labels)

    train_ratio = 0.70
    validation_ratio = 0.20
    test_ratio = 0.10
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size = ( 1 - train_ratio )
    )

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    # Get a compiled neural network
    model = get_model(x_train)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='tmp/checkpoint/',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    history = History()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS,validation_data=(x_val, y_val), callbacks=[tensorboard_callback, early_stopping_callback, model_checkpoint_callback,history])

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    #plotting graphs for accuracy 
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    plt.figure(1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()



def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    cur_path = os.getcwd()

    for i in range(NUM_CATEGORIES):
        path = os.path.join(cur_path, data_dir ,str(i))
        images = os.listdir(path)
        print(i)
        for a in images:
            try:
                image = Image.open(path + '\\'+ a)
                image = image.resize((30,30))
                image = np.array(image)
                #sim = Image.fromarray(image)
                data.append(image)
                label.append(i)
                
            except:
                print("Error loading image")
    
    return(data,label)

def get_model(x_train):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    model = tf.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    #Compilation of the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return (model)
    raise NotImplementedError

if __name__ == "__main__":
    main()