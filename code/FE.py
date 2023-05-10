import numpy as np
import tensorflow as tf
import cv2
from tqdm.auto import tqdm

def full_network_embedding(model, image_paths, batch_size, target_layer_names, input_reshape, stats=None):
    ''' 
    Generates the Full-Network embedding[1] of a list of images using a pre-trained
    model (input parameter model) with its computational graph loaded. Tensors used 
    to compose the FNE are defined by target_tensors input parameter. The input_tensor
    input parameter defines where the input is fed to the model.

    By default, the statistics used to standardize are the ones provided by the same 
    dataset we wish to compute the FNE for. Alternatively these can be passed through
    the stats input parameter.

    This function aims to generate the Full-Network embedding in an illustrative way.
    We are aware that it is possible to integrate everything in a tensorflow operation,
    however this is not our current goal.

    [1] https://arxiv.org/abs/1705.07706
   
    Args:
        model (tf.GraphDef): Serialized TensorFlow protocol buffer (GraphDef) containing the pre-trained model graph
                             from where to extract the FNE. You can get corresponding tf.GraphDef from default Graph
                             using `tf.Graph.as_graph_def`.
        image_paths (list(str)): List of images to generate the FNE for.
        batch_size (int): Number of images to be concurrently computed on the same batch.
        target_layer_names (list(str)): List of tensor names from model to extract features from.
        input_reshape (tuple): A tuple containing the desired shape (height, width) used to resize the image.
        stats (2D ndarray): Array of feature-wise means and stddevs for standardization.

    Returns:
       2D ndarray : List of features per image. Of shape <num_imgs,num_feats>
       2D ndarry: Mean and stddev per feature. Of shape <2,num_feats>
    '''

    # Define feature extractor
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in model.layers if layer.name in target_layer_names],
    )
    get_raw_features = lambda x: [tensor.numpy() for tensor in feature_extractor(x)]

    # Prepare output variable
    feature_shapes = [layer.output_shape for layer in model.layers if layer.name in target_layer_names]
    len_features = sum(shape[-1] for shape in feature_shapes)
    features = np.empty((len(image_paths), len_features))

    # Extract features
    progress_bar = tqdm(total=len(image_paths), desc="Extracting", ncols=100)
    for idx in range(0, len(image_paths), batch_size):
        batch_images_path = image_paths[idx:idx + batch_size]
        img_batch = np.zeros((len(batch_images_path), *input_reshape, 3), dtype=np.float32)
        for i, img_path in enumerate(batch_images_path):
            cv_img = cv2.imread(img_path)
            try:
                cv_img_resize = cv2.resize(cv_img, input_reshape)
                img_batch[i] = np.asarray(cv_img_resize, dtype=np.float32)[:, :, ::-1]
                
            except:
                print(img_path)
        progress_bar.set_postfix_str(f"Images:{idx}/{len(image_paths)}")
        progress_bar.update(idx)

        feature_vals = get_raw_features(img_batch)
        features_current = np.empty((len(batch_images_path), 0))
        for feat in feature_vals:
            #If its not a conv layer, add without pooling
            if len(feat.shape) != 4:
                features_current = np.concatenate((features_current, feat), axis=1)
                continue
            #If its a conv layer, do SPATIAL AVERAGE POOLING
            pooled_vals = np.mean(np.mean(feat, axis=2), axis=1)
            features_current = np.concatenate((features_current, pooled_vals), axis=1)
        # Store in position
        features[idx:idx+len(batch_images_path)] = features_current.copy()

    # STANDARDIZATION STEP
    # Compute statistics if needed
    if stats is None:
        stats = np.zeros((2, len_features))
        stats[0, :] = np.mean(features, axis=0)
        stats[1, :] = np.std(features, axis=0)
    # Apply statistics, avoiding nans after division by zero
    features = np.divide(features - stats[0], stats[1], out=np.zeros_like(features), where=stats[1] != 0)
    if len(np.argwhere(np.isnan(features))) != 0:
        raise Exception('There are nan values after standardization!')
    # DISCRETIZATION STEP
    th_pos = 0.15
    th_neg = -0.25
    features[features > th_pos] = 1
    features[features < th_neg] = -1
    features[[(features >= th_neg) & (features <= th_pos)][0]] = 0

    # # Store output
    # outputs_path = '../outputs'
    # if not os.path.exists(outputs_path):
    #     os.makedirs(outputs_path)
    # np.save(os.path.join(outputs_path, 'fne.npy'), features)
    # np.save(os.path.join(outputs_path, 'stats.npy'), stats)
    progress_bar.close()
    # Return
    return features, stats
  ##################################### Import dataset ##################################
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
environment = ["cluster", "colab"][1]
# you don't need to un-comment cloning
data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")
label_path = ['MAMe_toy_dataset.csv', 'MAMe_dataset.csv'][1]
batch_size = 64

##------------------------Preprocess--------------------------------
if environment == "colab":
  import sys
  os.system("git clone https://github.com/egasgira/TransferLearning.git")
  sys.path.append('/content/TransferLearning/code')
  import TransferLearning.code.data_reader as data_reader
  import TransferLearning.code.preprocess as preprocess
  data_dir = "/content/TransferLearning/datasets"
else:
  #import data_reader
  import preprocess
  data_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets")

#################### The main code ####################
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    # This shows an example of calling the full_network_embedding method using
    # the VGG16 architecture pretrained on ILSVRC2012 (aka ImageNet), as
    # provided by the keras package. Using any other pretrained CNN
    # model is straightforward.

    # img_width, img_height = 224, 224
    # initial_model = tf.keras.applications.VGG16(weights="imagenet", include_top=True,
    #                                             input_shape=(img_width, img_height, 3))
    # target_layer_names = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
    #                       'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2',
    #                       'block5_conv3', 'fc1', 'fc2']

    img_width, img_height = 256, 256
    initial_model = applications.efficientnet_v2.EfficientNetV2B2(weights = "imagenet", include_top=False, input_shape=(img_width, img_height, 3))
    target_layer_names = ['block4a_expand_conv', 'block4b_expand_conv', 'block4c_expand_conv', 'block4d_expand_conv',
                      'block5a_expand_conv', 'block5b_expand_conv', 'block5c_expand_conv', 'block5d_expand_conv',
                      'block6a_expand_conv', 'block6b_expand_conv', 'block6c_expand_conv', 'block5d_expand_conv',
                      'block7a_expand_conv', 'block7b_expand_conv', 'block7c_expand_conv', 'block5d_expand_conv',]

    dr = data_reader2(data_dir)
    file_names, y_data, train_val_test_idication, labels = dr.get_data_info(label_path) 
    train_images, train_labels, test_images, test_labels, X_data_test, Y_data_test = dr.get_data_paths(y_data, file_names, train_val_test_idication, data_dir)

    print('Total train images:', len(train_images), ' with their corresponding', len(train_labels), 'labels')
    print('Total test images:', len(test_images), ' with their corresponding', len(test_labels), 'labels')

    # Parameters for the extraction procedure
    batch_size = 32
    input_reshape = (256, 256)
    # Call FNE method on the train set
    fne_features, fne_stats_train = full_network_embedding(initial_model, train_images, batch_size,
                                                           target_layer_names, input_reshape)
    print('Done extracting features of training set. Embedding size:', fne_features.shape)
    print('Start training SVM')

    from sklearn import svm

    # Train SVM with the obtained features.
    clf = svm.LinearSVC()
    clf.fit(X=fne_features, y=train_labels)
    print('Done training SVM on extracted features of training set')

    # Call FNE method on the test set, using stats from training
    fne_features, fne_stats_train = full_network_embedding(initial_model, test_images, batch_size,
                                                           target_layer_names, input_reshape, stats=fne_stats_train)
    print('Done extracting features of test set')

    # Test SVM with the test set.
    predicted_labels = clf.predict(fne_features)
    print('Done testing SVM on extracted features of test set')

    # Print results
    print(classification_report(test_labels, predicted_labels, digits=3))
    print(confusion_matrix(test_labels, predicted_labels))
