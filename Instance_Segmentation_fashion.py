#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip3 install -r requirements.txt')


# In[ ]:


get_ipython().system('python3 setup.py install')


# In[ ]:


import gc
import sys
import json
import glob
import random
from pathlib import Path
import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from google.colab import files
from google.colab.patches import cv2_imshow
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import imgaug.augmenters as iaa

import itertools
from tqdm import tqdm

from imgaug import augmenters as iaa
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split


# In[ ]:


os.chdir('/content/maskr_cnn')


# In[4]:


get_ipython().system('wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5')

get_ipython().system('ls -lh mask_rcnn_coco.h5')


# In[ ]:


COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'


# In[ ]:


ROOT_DIR = Path("/content/")
DATA_DIR = ROOT_DIR/'train'
NUM_CATS = 67
IMAGE_SIZE = 512


# In[ ]:


sys.path.append(str(ROOT_DIR/'maskrcnn'))
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import log
import mrcnn.model as modellib


# ***Mask R-CNN has a load of hyperparameters. ***

# In[ ]:


class FashionConfig(Config):
    NUM_CATS = 67                  #the number of clothing categories (clases)
    IMAGE_SIZE = 512
    NAME = "fashion"
    NUM_CLASSES = NUM_CATS + 1     # +1 for the background class

    GPU_COUNT = 1
    IMAGES_PER_GPU = 4             # in each iteration of training or prediction, 4 images per GPU will pass through the model simultaneously

    BACKBONE = 'resnet101'         #the model will first process the image through this architecture to obtain feature vector representations, and then these features are used for further tasks such as segmentation.

    IMAGE_MIN_DIM = IMAGE_SIZE
    IMAGE_MAX_DIM = IMAGE_SIZE
    IMAGE_RESIZE_MODE = 'none'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256) #anchor scales (or "rectangular windows") used by the Region Proposal Network (RPN) to find objects at different scales in the image.
    TRAIN_ROIS_PER_IMAGE = 200                 #the maximum number of regions of interest (ROIs) used to train the model on each image during the training phase.
    STEPS_PER_EPOCH = 7980        #STEPS_PER_EPOCH = Total train images/(IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = 3420       #VALIDATION_STEPS = Total train images/(IMAGES_PER_GPU * GPU_COUNT)
config = FashionConfig()
config.display()


# In[1]:


# Category names
label_names = [
    "T-shirt",
    "Long-Sleeve T-Shirt",
    "Tank Top",
    "Crop Top",
    "Off-Shoulder Top",
    "Halter Top",
    "Blouse",
    "Shirt",
    "Sweater",
    "Cardigan",
    "Vest",
    "Puffer Vest",
    "Denim Vest",
    "Hoodie",
    "Sweatshirt",
    "Tunic",
    "Kimono",
    "Polo",
    "Jersey",
    "Denim Jacket",
    "Leather Jacket",
    "Bomber Jacket",
    "Puffer Jacket",
    "Quilted Jacket",
    "Windbreaker",
    "Varsity Jacket",
    "Blouson Jacket",
    "Short Blazer",
    "Long Blazer",
    "Fitted Blazer",
    "Oversized Blazer",
    "Coat",
    "Trench Coat",
    "Peacoat",
    "Overcoat",
    "Duffle Coat",
    "Parka",
    "Wool Coat",
    "Down Coat",
    "Raincoat",
    "Cape",
    "Gilet",
    "Pants",
    "Leggings",
    "Jeans",
    "Shorts",
    "Mini Skirt",
    "Midi Skirt",
    "Maxi Skirt",
    "Mini Dress",
    "Midi Dress",
    "Maxi Dress",
    "Jumpsuit",
    "Gown",
    "Sandals",
    "Flip Flops",
    "Espadrilles",
    "Sneakers",
    "Boots",
    "Loafers",
    "Oxfords",
    "Ballet Flats",
    "Mules",
    "Heels",
    "Necklace",
    "Bag",
    "Backpack"
]


# In[ ]:


uploaded = files.upload() #image_df.csv


# # ***Image Annotation:***
# 
# The images were annotated using the **VGG Image Annotator (VIA**) tool. This tool allows for the annotation of objects in images, where each object is assigned a mask that describes its shape and location in the image. Each object, such as an item of clothing or an accessory, is manually outlined and labeled with the appropriate class.
# 
# **Saving Annotation Results:**
# 
# After completing the annotation, the data from VIA is exported in CSV format. Each object in the image has mask coordinates (in RLE format or a list of pixels), the object's class, and the image dimensions.
# 
# **Preparation of the image_df file:**
# 
# After obtaining the annotation results, all data is converted into a table, which is stored as a DataFrame (using the Pandas library). The main columns of this DataFrame are:
# 
# **ImageId:** A unique identifier for the image (without the file extension), used to link each object to the corresponding image.  
# **Height:** The height of the image (in pixels), to know the size of the mask that corresponds to this image.  
# **Width:** The width of the image (in pixels), similarly used for scaling the masks.  
# **EncodedPixels:** The masks for each object, typically represented in Run-Length Encoding (RLE) format. This is a compressed format that encodes the number of background and object pixels.  
# **ClassId:** The identifier of the object class, which represents the type of object (e.g., jacket, dress, socks, etc.).

# In[ ]:


#The  function resizes an image.
def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


# In[ ]:


#Creating a dataset for model training.

"""The FashionDataset class, which inherits from the utils.Dataset class (in model.py).
It is designed to prepare data for training a model based on clothing segmentation.
The Dataset class in the Mask R-CNN library serves as a base class for creating specific datasets used to train the model.
Using the utils.Dataset base class template ensures that FashionDataset will be compatible with other pieces of code that expect Dataset objects,
thus preserving the common structure and behavior."""

class FashionDataset(utils.Dataset):

    def __init__(self, df):
        super().__init__(self)

        # Add clothing-specific classes to the dataset. Label_names is a list of class names and i+1 is the ID of each class.
        for i, name in enumerate(label_names):
            self.add_class("fashion", i+1, name)

        # Add images (For each line in df, an image is added with the specified file path, object classes, and mask annotations. Additionally, image dimensions (height and width) are added.)
        for i, row in df.iterrows():
            self.add_image("fashion",
                           image_id=row.name,
                           path=str(DATA_DIR/'train'/row.name) + '.jpg',
                           labels=row['ClassId'],
                           annotations=row['EncodedPixels'],
                           height=row['Height'], width=row['Width'])

    #Returns the image path and list of class names for this image. Used to track image information.
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path'], [label_names[int(x)] for x in info['labels']]

    #Loads and scales the image for later use in the model.
    def load_image(self, image_id):
        return resize_image(self.image_info[image_id]['path'])

    #Loads the masks for each object in the image and returns them along with the classes of those objects.
    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)
        labels = []

        #The loop iterates through each annotation and class to decode the object mask from the RLE format.
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            sub_mask = np.full(int(info['height']) * int(info['width']), 0, dtype=np.uint8)
            annotation = [int(x) for x in annotation.split(' ')]

            #RLE decoding(an annotation is the RLE code for the mask and sub_mask is the object mask that is initially padded with zeros (background pixels)):
            for i, start_pixel in enumerate(annotation[::2]):
                sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            #Resizing the mask to the desired size.
            sub_mask = sub_mask.reshape((int(info['height']), int(info['width'])), order='F')

            mask[:, :, m] = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            labels.append(int(label) + 1)

        return mask, np.array(labels)


# In[ ]:


dataset = FashionDataset(image_df)
dataset.prepare()


# In[ ]:


#file_path = '/content/drive/MyDrive/fashion.zip'
#import zipfile

#with zipfile.ZipFile(file_path, 'r') as zip_ref:
   # zip_ref.extractall('/content/')



# In[ ]:


get_ipython().system('unzip /content/drive/MyDrive/fashion.zip "train/*" -d /content/train')


# In[ ]:


DATA_DIR = Path('/content/train')


# In[ ]:


# The data are partitioned into train and validation sets

train_df, valid_df = train_test_split(image_df, test_size=0.3, random_state=42)

train_dataset = FashionDataset(train_df)
train_dataset.prepare()

valid_dataset = FashionDataset(valid_df)
valid_dataset.prepare()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# Data training using the configuration of the Mask R-CNN model and pre-prepared weights that we load into the model. The model should be trained in two processes. The first is "header learning". This is a mask label training class to predict the mask label of each object.

# In[ ]:


#Creating and initially loading a model for training
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)

model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
    'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])


# In[ ]:


"""Loads pre-trained weights mask_rcnn_coco.h5,
 excluding the layers responsible for class and mask prediction.
This allows you to relearn the model for new classes."""


# In[ ]:


#The model is trained on two epoches for the heads layers responsible for the main components.
get_ipython().run_line_magic('%time', '')
model.train(train_dataset, valid_dataset,
            learning_rate=2e-3,
            epochs=2,
            layers='heads',
            augmentation=None)

""" We limit ourselves to training only the last layers of the network,
which are responsible for classification and segmentation (for example, the classification layer, boxes and masks).
This layer depends more on the specifics of our data (for example: categories of clothing items)."""

history = model.keras_model.history.history


# In[ ]:


# Save model weights
model.keras_model.save_weights('/content/drive/MyDrive/model_weights.h5')


# In[ ]:


#Augmentation can help make training more robust to variations in the data.
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% chance of horizontal reflection
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))  # Sharpness application
])


# In[ ]:


# We load the weights of the model
model.load_weights("/content/drive/MyDrive/model_weights.h5")

callbacks = [
    ModelCheckpoint(filepath='/content/drive/MyDrive/my_model.h5', save_best_only=True), #keeps the best model weights according to the validation loss criterion.
    EarlyStopping(patience=5), #stops training if validation loss does not improve within 5 epochs.
    ReduceLROnPlateau(factor=0.1, patience=2), #reduces learning rate by 0.1 if validation loss does not improve within 2 epochs.
    TensorBoard(log_dir='./logs') #keeps logs for training monitoring.
]

# Training the model with a callback
get_ipython().run_line_magic('%time', '')
model.train(train_dataset, valid_dataset,
            learning_rate=1e-4, # Smaller learning rate for fine tuning
            epochs=10,
            layers='all',
            augmentation=augmentation,
            callbacks=callbacks)
#The model is trained on all layers (layers='all') for 10 epochs using augmentations and callbacks.

# Combining stories
new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]


# # **Predict**

# In[ ]:


#This cell defines InferenceConfig and loads the best trained model.
class InferenceConfig(FashionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
inference_config = InferenceConfig()

glob_list = glob.glob(f'/content/drive/MyDrive/my_model.h5')
model_path = glob_list[0] if glob_list else ''

#Creating a model for inference
model = modellib.MaskRCNN(mode='inference',#'inference' mode for predictions
                          config=inference_config,
                          model_dir=ROOT_DIR)

assert model_path != '/drive/My Drive/my_model.h5'
model.load_weights(model_path, by_name=True)


# In[ ]:


import cv2
from tqdm import tqdm

def resize_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return img


# In[ ]:


image = resize_image(image_path)
r = model.detect([image], verbose=0)
r=r[0]


# # **Additional functions:**

# In[ ]:


"""The trim_masks function is used to handle segmentation of masks,
to overlay pixels from one mask to another within the same class,
and to update the coordinates of the regions of interest (ROI) according to the new masks.
This is important for improving the training of segmentation models, after which it eliminates some conflicts in the data."""

def trim_masks(masks, rois, class_ids):
    class_pos = np.argsort(class_ids)
    class_rle = to_rle(np.sort(class_ids))

    pos = 0
    for i, _ in enumerate(class_rle[::2]):
        previous_pos = pos
        pos += class_rle[2*i+1]
        if pos-previous_pos == 1:
            continue
        mask_indices = class_pos[previous_pos:pos]

        union_mask = np.zeros(masks.shape[:-1], dtype=bool)
        for m in mask_indices:
            masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
            union_mask = np.logical_or(masks[:, :, m], union_mask)
        for m in mask_indices:
            mask_pos = np.where(masks[:, :, m]==True)
            if np.any(mask_pos):
                y1, x1 = np.min(mask_pos, axis=1)
                y2, x2 = np.max(mask_pos, axis=1)
                rois[m, :] = [y1, x1, y2, x2]

    return masks, rois


# In[ ]:


# Convert data to run-length encoding
def to_rle(bits):
    rle = []
    pos = 0         #position (index) in the bit array, initially set to 0.
    """grouping of adjacent values ​​in bits. For example, if bits has the value [0, 0, 1, 1, 1, 0],
       then there will be two groups: one of 0 and one of 1."""
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle

"""
The to_rle(bits) function converts a two-dimensional array of bits into a compressed format based on Run-Length Encoding (RLE).
This format is used to represent binary images or masks where the same value (0 or 1) is repeated consecutively.
pos: The starting position for this group (where sequence 1 starts).
sum(group_list): The number of times the value 1 occurs in this group.
The function returns a rle list that contains the position pairs and the number of repetitions for all groups of the value 1."""

