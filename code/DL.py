from __future__ import division
import os
import pandas as pd
from PIL import Image
import numpy as np
import keras
import tensorflow as tf
import cv2
import sklearn
from keras.backend import clear_session
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical


def csv_reader(dir_path):

    print('\n############### CSV FILES ###############\n')

    # Upload the csv files of the directory
    dir_path = dir_path + '\MAMe_metadata'
    files = os.listdir(dir_path)

    print('MAMe_metadata directory uploaded successfully.')
    print('The files are: ', files)
    # Read the CSV file with the data indexing of the label names
    df_names = pd.read_csv(dir_path + '\MAMe_labels.csv',header=None)
    label_names = df_names[1]

    # Read the CSV file with the data (MAMe_dataset.csv)
    df = pd.read_csv(dir_path + label_path)# + files[2])
    df = df.sample(frac=1, random_state=100).reset_index(drop=True)
    file_name = df['Image file'].to_numpy()
    file_label = df['Medium'].to_numpy()

    file_subset = df['Subset'].to_numpy()
    print('There are {} files and {} labels.'.format(file_name.shape[0], file_label.shape[0]))

    # Mapping class names to numbers
    value_to_index = {value: index for index, value in enumerate(label_names)}
    file_label = np.array([value_to_index[value.strip()] for value in file_label])
    print('\n############### CSV FILES ###############\n')

    return file_name, file_label, file_subset, label_names



def image_data_reader(file_label, file_names, file_subset, dir_path):

    print('\n############### IMAGE FILES ###############\n')

    # Upload the jpeg and jpg files of the directory

    dir_path = dir_path + "\data_256"
    files = os.listdir(dir_path)

    print('data_256 directory uploaded successfully.')
    print('The data_256 directory has {} images.'.format(len(files)))

    # Create a OneHotEncoder object
    encoder = OneHotEncoder()

    # Fit the encoder to the labels
    encoder.fit(np.array(file_label).reshape(-1, 1))

    # Transform the labels to one-hot encoding
    encoded = encoder.transform(np.array(file_label).reshape(-1, 1)).toarray()
    Y_data_train = []
    Y_data_val = []
    Y_data_test = []
    X_data_train = []
    X_data_val = []
    X_data_test = []
    # Loop through the image files in the directory
    for image_file, y_label, subset_name in zip(file_names, file_label, file_subset):
        if subset_name == 'train':
            X_data_train.append(np.array(cv2.imread(dir_path + "\\" + image_file)))
            Y_data_train.append(y_label)
        elif subset_name == 'val':
            X_data_val.append(np.array(cv2.imread(dir_path + "\\" + image_file)))
            Y_data_val.append(y_label)
        else:
            X_data_test.append(np.array(cv2.imread(dir_path + "\\" + image_file)))
            Y_data_test.append(y_label)

        #if image_file.endswith(".jpeg") or image_file.endswith(".jpg"):
        #    # Open the image file
        #    image_path = os.path.join(dir_path, image_file)
        #    image = Image.open(image_path)
        #    # Append the image array to X_data
        #    X_data.append(np.array(image))

    # Print shape
    print('The shape of each image in the data is {}.'.format(X_data_train[0].shape))
    print('The shape of the Y-vector is({},{}).'.format(len(Y_data_train),Y_data_train[0].shape))

    print('\n############### IMAGE FILES ###############\n')

    return np.array(X_data_train), np.array(Y_data_train), np.array(X_data_val), np.array(Y_data_val), np.array(X_data_test), np.array(Y_data_test)
clear_session()


# To test with a path in the computer
# To use in the super computer, comment the following line
#dir_path = "/Users/joaovalerio/Downloads"


dir_path = ['/home/nct01/nct01143/.keras/CNN/datasets', "D:\Documents\Datasett\CNN\datasets"][0]
label_path = ['/MAMe_toy_dataset.csv', '/MAMe_dataset.csv'][0]



file_name, file_label, file_subset, label_names = csv_reader(dir_path)
X_data_train, Y_data_train, X_data_val, Y_data_val, X_data_test, Y_data_test = image_data_reader(file_label, file_name, file_subset, dir_path)

print(to_categorical(Y_data_train))

##------------------------Training--------------------------------


print( 'Using Keras version', keras.__version__)

#Check sizes of dataset
print( 'Number of train examples', X_data_train.shape[0])
print( 'Size of train examples', X_data_train.shape[1:])

#Normalize data
x_train = X_data_train.astype('float32')
x_test = X_data_test.astype('float32')
x_val = X_data_val
x_train = X_data_train / 255
x_test = X_data_test / 255
x_val = X_data_val / 255
y_train = Y_data_train
y_test = Y_data_test
y_val = Y_data_val

#Adapt the labels to the one-hot vector syntax required by the softmax
from keras.utils import np_utils
size = max(y_train) + 1
y_train = np.array([np_utils.to_categorical(i, size) for i in y_train])
y_test = np.array([np_utils.to_categorical(i, size) for i in y_test])
y_val = np.array([np_utils.to_categorical(i, size) for i in y_val])

#resolution
img_rows, img_cols, channels = x_train.shape[1:][0], x_train.shape[1:][1], x_train.shape[1:][2]
input_shape = (img_rows, img_cols, channels)
#Reshape for input
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

#Define the NN architecture
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
#Two hidden layers
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(len(y_train[0]), activation=(tf.nn.softmax)))#.shape[1]

#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(model, to_file='model.png', show_shapes=true)

#Compile the NN
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])

#Start training
history = model.fit(x_train,y_train,batch_size=64,epochs=20, validation_data=(x_val, y_val))

#Evaluate the model with test set
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Accuracy plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_fnn_accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('mnist_fnn_loss.pdf')

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
#Compute probabilities
Y_pred = model.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = [label for label in label_names]
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

#Saving model and weights
from keras.models import model_from_json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
model.save_weights(weights_file,overwrite=True)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)