# %%
#Hu moment https://en.wikipedia.org/wiki/Image_moment
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#Mahotas is a coptuer vision and image processing library for python. https://mahotas.readthedocs.io/en/latest/
import mahotas
#HDF5 let you store huge amoints of numerical data from NumPy. http://www.h5py.org/
import h5py
#Opencv is a library of Python bindings designed to solve computer vision problems. https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_intro/py_intro.html#intro
import cv2
import os
import time

# %% [markdown]
# ## Variables

# %%
images_per_class =  4608
fixed_size       = tuple((512,512))
train_path       = 'dataset/train'
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8

# %% [markdown]
# ## Fonctions

# %%
def hu_moments(image):
    # convert the image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #https://en.wikipedia.org/wiki/Image_moment
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# %%
# get the training labels
print(os.listdir)
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# %%
# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name  
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "\\ (" + str(x) + ").jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        #image = cv2.resize(image, fixed_size)

        #Extraction des différentes features à partir des fonctions
        fv_hu_moments = hu_moments(image)
        fv_haralick   = haralick(image)
        fv_histogram  = histogram(image)

        #Concatène toutes les features ensemble
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # Update lists
        labels.append(current_label)
        global_features.append(global_feature)
    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# %% [markdown]
# ## Encode data

# %%
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")


