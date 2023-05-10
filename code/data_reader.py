from __future__ import division
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

class data_reader:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def get_data_info(self, label_path):

        print('\n############### CSV FILES ###############\n')

        # Upload the csv files of the directory
        dir_path = os.path.join(self.dir_path, 'MAMe_metadata')
        files = os.listdir(dir_path)

        print('MAMe_metadata directory uploaded successfully.')
        print('The files are: ', files)
        # Read the CSV file with the data indexing of the label names
        df_names = pd.read_csv(os.path.join(dir_path, 'MAMe_labels.csv'),header=None)
        label_names = df_names[1]

        # Read the CSV file with the data (MAMe_dataset.csv)
        df = pd.read_csv(os.path.join(dir_path, label_path))# + files[2])
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



    def get_data_set(self, file_label, file_names, file_subset, dir_path):

        print('\n############### IMAGE FILES ###############\n')

        # Upload the jpeg and jpg files of the directory

        dir_path = os.path.join(dir_path, "data_256")
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
                X_data_train.append(np.array(cv2.imread(os.path.join(dir_path, image_file))))
                Y_data_train.append(y_label)
            elif subset_name == 'val':
                X_data_val.append(np.array(cv2.imread(os.path.join(dir_path, image_file))))
                Y_data_val.append(y_label)
            else:
                X_data_test.append(np.array(cv2.imread(os.path.join(dir_path, image_file))))
                Y_data_test.append(y_label)

        # Print shape
        print('The shape of each image in the data is {}.'.format(X_data_train[0].shape))
        print('The shape of the Y-vector is({},{}).'.format(len(Y_data_train),Y_data_train[0].shape))

        print('\n############### IMAGE FILES ###############\n')

        return np.array(X_data_train), np.array(Y_data_train), np.array(X_data_val), np.array(Y_data_val), np.array(X_data_test), np.array(Y_data_test)

    def get_data_generator(self, file_label, file_names, file_subset, dir_path, batch_size):

        print('\n############### IMAGE FILES ###############\n')

        dir_path = os.path.join(dir_path, "data_256")
        files = os.listdir(dir_path)

        print('data_256 directory uploaded successfully.')
        print('The data_256 directory has {} images.'.format(len(files)))

        # create dataframes
        data = {'filename': file_names, 'label': file_label.astype(str), 'subset': file_subset}
        df = pd.DataFrame(data)

        # Create a OneHotEncoder object
        encoder = OneHotEncoder()

        # Fit the encoder to the labels
        encoder.fit(np.array(file_label).reshape(-1, 1))

        # setup generator
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        # setup data augmentation for trainingdata
        train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,       # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,   # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,    # randomly flip images horizontally
        zoom_range=0.20,           # Randomly zoom image 
        brightness_range=[0.5,1.4]
        )
        print("test2")

        # create train, val, and test generators
        train_generator = datagen.flow_from_dataframe(
            dataframe=df[df['subset']=='train'],
            directory=dir_path,
            x_col='filename',
            y_col='label',
            #target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=df[df['subset']=='val'],
            directory=dir_path,
            x_col='filename',
            y_col='label',
            #target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
        )

        test_generator = datagen.flow_from_dataframe(
            dataframe=df[df['subset']=='test'],
            directory=dir_path,
            x_col='filename',
            y_col='label',
            #target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical',
        )

        print('\n############### IMAGE FILES ###############\n')

        return train_generator, val_generator, test_generator    
    def get_data_paths(self, file_label, file_names, file_subset, dir_path):

            print('\n############### IMAGE FILES ###############\n')

            # Upload the jpeg and jpg files of the directory

            dir_path = os.path.join(dir_path, "data_256")
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
                    X_data_train.append(os.path.join(dir_path, image_file))
                    Y_data_train.append(y_label)
                elif subset_name == 'val':
                    X_data_val.append(os.path.join(dir_path, image_file))
                    Y_data_val.append(y_label)
                else:
                    X_data_test.append(os.path.join(dir_path, image_file))
                    Y_data_test.append(y_label)

            
            
            

            print('\n############### IMAGE FILES ###############\n')

            return X_data_train, Y_data_train, X_data_val, Y_data_val, X_data_test, Y_data_test
