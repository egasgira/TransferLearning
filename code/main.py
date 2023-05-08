from __future__ import division
import os
import pandas as pd
from PIL import Image
import numpy as np
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib
import cv2
import os
import sklearn
from keras.backend import clear_session
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras import applications


# Choose what to do
show_analytics = False
clear_session()
environment = ["cluster", "colab"][0]
#!git clone https://github.com/egasgira/TransferLearning.git # Uncomment this if colab is used

data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
label_path = ['MAMe_toy_dataset.csv', 'MAMe_dataset.csv'][1]


##------------------------Preprocess--------------------------------
if environment == "colab":
  import sys
  sys.path.append('/content/TransferLearning/code')
  import TransferLearning.code.data_reader as data_reader
  import TransferLearning.code.preprocess as preprocess
  data_dir = "/content/TransferLearning/datasets"
else:
  import data_reader
  import preprocess
  data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")

dr = data_reader.data_reader(data_dir)
file_names, y_data, train_val_test_idication, labels = dr.get_data_info(label_path)
X_data_train, Y_data_train, X_data_val, Y_data_val, X_data_test, Y_data_test = dr.get_data_set(y_data, file_names, train_val_test_idication, data_dir)


##------------------------Analysis--------------------------------

# Check sizes of dataset
print( 'Number of train examples', X_data_train.shape[0])
print( 'Size of train examples', X_data_train.shape[1:])

if(show_analytics):
        # Print 10 random images for each class
        for h in range(0,len(labels),5):
                fig, axes = plt.subplots(len(labels[h:h+5]), 5, figsize=(30, len(labels[h:h+5]) * 30))
                for i, label in enumerate(labels[h:h+5]):

                        label_indexes = np.where(Y_data_train == h+i)[0][:5]
                        for j, indx in enumerate(label_indexes):
                                axes[i, j].imshow(X_data_train[indx])
                                axes[i, j].axis("off")
                                axes[i, j].set_title(f"label: {label}")
                #plt.tight_layout()
                plt.show()

        plt.title("Distribution of classes")
        labs = labels.values
        data_balance = pd.Series(Y_data_train).value_counts()
        plt.bar(labs,data_balance)
        plt.xticks([i for i in range(len(labs))], labs, rotation='vertical')
        plt.show()

##------------------------Training--------------------------------

# Normalize data

X_data_train, Y_data_train = preprocess.preprocess(X_data_train, Y_data_train)
X_data_val, Y_data_val = preprocess.preprocess(X_data_val, Y_data_val)
X_data_test, Y_data_test = preprocess.preprocess(X_data_test, Y_data_test)

#resolution
img_rows, img_cols, channels = X_data_train.shape[1:][0], X_data_train.shape[1:][1], X_data_train.shape[1:][2]
input_shape = (img_rows, img_cols, channels)
#Reshape for input
#X_data_train = X_data_train.reshape(X_data_train.shape[0], img_rows, img_cols, channels)
#X_data_test = X_data_test.reshape(X_data_test.shape[0], img_rows, img_cols, channels)

#Define the NN architecture
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

# Base model
model = applications.efficientnet_v2.EfficientNetV2M(weights = "imagenet", include_top=False, input_shape = input_shape)

# Freeze the layers which you don't want to train. Here I am freezing the first 10 layers.
limit_frozen_layers = int(len(model.layers)/3)
for layer in model.layers[:limit_frozen_layers]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(len(Y_data_train[0]), activation=(tf.nn.softmax))(x)

# creating the final model 
model = Model(model.input, predictions)


'''
#Two hidden layers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',  padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2) ))
model.add(Dropout(0.3))
model.add(Conv2D(128, (2, 2), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.4))
model.add(Conv2D(256, (3, 3), activation='relu', padding="same"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(len(Y_data_train[0]), activation=(tf.nn.softmax)))#.shape[1]
'''

#Model visualization
#We can plot the model by using the ```plot_model``` function. We need to install *pydot, graphviz and pydot-ng*.
#from keras.util import plot_model
#plot_model(model, to_file='model.png', show_shapes=true)

#Compile the NN
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#Start training
es = EarlyStopping(patience=10,  restore_best_weights=True, monitor="val_loss")
history = model.fit(X_data_train,Y_data_train,batch_size=64,epochs=40,validation_data=(X_data_val, Y_data_val), callbacks=[es])

#Evaluate the model with test set
score = model.evaluate(X_data_test, Y_data_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

##Store Plots

matplotlib.use('Agg')
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
#Compute probabilities
Y_pred = model.predict(X_data_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = [label for label in labels]
print(classification_report(np.argmax(Y_data_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(Y_data_test,axis=1), y_pred))

#Saving model and weights
from keras.models import model_from_json
model_json = model.to_json()
with open('model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = "weights-MNIST_"+str(score[1])+".hdf5"
model.save_weights(weights_file,overwrite=True)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_cm(model, X_val, y_val, cm_filename, lc_filename):
    # Prediction
    predictions = model.predict(X_val)
    # Select for each observation the highest probability
    predictions = np.argmax(predictions, axis=1)
    real = np.argmax(y_val, axis=1)
    # Confusion matrix
    confusionMatrix = confusion_matrix(real, predictions, normalize="true")

    #print(pd.DataFrame(confusionMatrix, columns=labels, index=labels))
    cmd = ConfusionMatrixDisplay(confusionMatrix, display_labels=labels.values)
    fig, ax = plt.subplots()
    cmd.plot(ax=ax)
    fig.set_figheight(13)
    fig.set_figwidth(14)
    plt.xticks(rotation=90)

    plt.savefig(cm_filename)
    plt.close()
# Usage example
# Assume `model` is a trained Keras model, and X_train, y_train, X_val, y_val are the input data and true labels
save_cm(model, X_data_val, Y_data_val, "confusion_matrix.png", labels)

#Loading model and weights
#json_file = open('model.json','r')
#model_json = json_file.read()
#json_file.close()
#model = model_from_json(model_json)
#model.load_weights(weights_file)
