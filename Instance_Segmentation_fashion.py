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

#Copy the  directory mrcnn inside the root directory

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


sys.path.append(str(ROOT_DIR/'main'))
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
    'T-shirt',
    'Long-Sleeve T-Shirt',
    'Tank Top',
    'Crop Top',
    'Off-Shoulder Top',
    'Halter Top',
    'Top',
    'Blouse',
    'Shirt',
    'Shirt Short-Sleeve',
    'Sweater',
    'Cardigan',
    'Vest',
    'Puffer Vest',
    'Denim Vest',
    'Hoodie',
    'Sweatshirt',
    'Tunic',
    'Kimono',
    'Polo',
    'Jersey',
    'Denim Jacket',
    'Leather Jacket',
    'Bomber Jacket',
    'Puffer Jacket',
    'Quilted Jacket',
    'Windbreaker',
    'Varsity Jacket',
    'Blouson Jacket',
    'Short Blazer',
    'Long Blazer',
    'Fitted Blazer',
    'Oversized Blazer',
    'Coat',
    'Trench Coat',
    'Peacoat',
    'Overcoat',
    'Duffle Coat',
    'Parka',
    'Wool Coat',
    'Down Coat',
    'Raincoat',
    'Cape',
    'Gilet',
    'Pants',
    'Pants Capri',
    'Leggings',
    'Jeans',
    'Shorts',
    'Mini Skirt',
    'Midi Skirt',
    'Maxi Skirt',
    'Mini Dress',
    'Midi Dress',
    'Maxi Dress',
    'Jumpsuit',
    'Gown',
    'Sandals',
    'Flip Flops',
    'Espadrilles',
    'Sneakers',
    'Boots',
    'Loafers',
    'Oxfords',
    'Ballet Flats',
    'Mules',
    'Heels',
    'Bag',
    'Backpack',
    'Necklace',
    'Glasses',
    'Bangles',
    'Shawl'
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
            epochs=10,
            layers='heads',
            augmentation=None)

""" We limit ourselves to training only the last layers of the network,
which are responsible for classification and segmentation (for example, the classification layer, boxes and masks).
This layer depends more on the specifics of our data (for example: categories of clothing items)."""

history = model.keras_model.history.history


# In[ ]:


# Save model weights
#model.keras_model.save_weights('/content/drive/MyDrive/model_weights.h5')


# In[ ]:


#Augmentation can help make training more robust to variations in the data.
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # 50% chance of horizontal reflection
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))  # Sharpness application
])


# In[ ]:


# We load the weights of the model
#model.load_weights("/content/drive/MyDrive/model_weights.h5")


# Training the model 
get_ipython().run_line_magic('%time', '')

model.train(train_dataset, valid_dataset,
            learning_rate=1e-3, # Smaller learning rate for fine tuning
            epochs= 20,
            layers='all',
            augmentation=augmentation
           )
#The model is trained on all layers (layers='all') for 20 epochs using augmentations.

# Combining stories
new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]


# In[ ]:


# Training the model 
get_ipython().run_line_magic('%time', '')

model.train(train_dataset, valid_dataset,
            learning_rate=1e-4, # Smaller learning rate for fine tuning
            epochs=30,
            layers='all',
            augmentation=augmentation
           )
#The model is trained on all layers (layers='all') for 30 epochs using augmentations.

# Combining stories
new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]


# In[ ]:


# Training the model 
get_ipython().run_line_magic('%time', '')

model.train(train_dataset, valid_dataset,
            learning_rate=1e-5, # Smaller learning rate for fine tuning
            epochs=50,
            layers='all',
            augmentation=augmentation
           )
#The model is trained on all layers (layers='all') for 50 epochs using augmentations and callbacks.

# Combining stories
new_history = model.keras_model.history.history
for k in new_history:
    history[k] = history[k] + new_history[k]



# In[ ]:

#Choosing the best epoch
best_epoch = np.argmin(history["val_loss"]) + 1
#print("Best epoch: ", best_epoch)
#print("Valid loss: ", history["val_loss"][best_epoch-1])

model.load_weights(f"best_model_epoch_{best_epoch}.h5")


