#!/usr/bin/env python
# coding: utf-8

# In[79]:


#!pip install torch torchvision pycocotools


# In[81]:


#!pip install pytorch-lightning


# In[83]:


#!pip install tensorboard


# In[85]:


#!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/metal.html


# In[44]:


import torch

print(torch.__version__)  # Перевірте версію PyTorch
print(torch.backends.mps.is_available())  # Перевірте, чи доступний MPS


# In[46]:


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


#model = model.to(device)
#input_tensor = input_tensor.to(device)


# In[48]:


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torch.nn as nn
import json


import collections
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.model_selection import KFold 

import os
from pathlib import Path

from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
from model import get_instance_segmentation_model
from engine import train_one_epoch, evaluate

from tqdm import tqdm

import cv2
import random
from numba import jit
import ast
import time
import gc

import warnings
warnings.filterwarnings("ignore")


# In[87]:


#!pip install opencv-python


# In[50]:


img_width = 512
img_height = 512
num_classes = 46+1
batch_size = 16
n_workers = 4


# In[89]:


#!pip install google-auth google-auth-oauthlib google-auth-httplib2


# In[52]:


import json
file_name = 'label_descriptions.json' 
with open(file_name, 'r') as f:
    label_descriptions = json.load(f)
label_names = [x['name'] for x in label_descriptions['categories']]


# In[54]:


label_names


# In[56]:


file_name = 'image_df.csv' 
image_df = pd.read_csv(file_name)
image_df


# In[58]:


image_df.info()


# In[60]:


empty_mask_rows = image_df[image_df['EncodedPixels'].isnull()]
print(f"Number of rows with empty EncodedPixels: {len(empty_mask_rows)}")


# In[62]:


# Décoder RLE en masque
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


# In[66]:


class FashionDataset(Dataset):
    def __init__(self, image_dir, image_df, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.image_df = image_df
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)

        for index, row in tqdm(image_df.iterrows(), total=len(image_df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

            if not os.path.isfile(image_path):
                print(f"Warning: {image_path} does not exist.")
                continue

            self.image_info[index]["image_id"] = image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["width"] = self.width
            self.image_info[index]["height"] = self.height
            self.image_info[index]["labels"] = row["ClassId"]
            self.image_info[index]["orig_height"] = row["Height"]
            self.image_info[index]["orig_width"] = row["Width"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]

        self.img2tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]
        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)
        labels = []

        # If 'labels' is a string, convert to a list
        label = info['labels']
        if isinstance(label, str):
            label = ast.literal_eval(label)

        for m, (annotation, single_label) in enumerate(zip(info['annotations'], label)):
            sub_mask = rle_decode(annotation, (int(info['orig_height']), int(info['orig_width'])))
            sub_mask = Image.fromarray(sub_mask).resize((self.width, self.height), resample=Image.BILINEAR)
            mask[m, :, :] = sub_mask
            labels.append(int(single_label) + 1)

        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])

                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(np.zeros((self.height, self.width), dtype=np.uint8))

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(new_labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)


# In[68]:


class CustomTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        img = self.transforms(img)  # Apply image transforms
        #  add logic here to transform the target ?
        return img, target

def get_transform(train=True):
    transforms = []
    if train:
        # Data augmentation for training data
        transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0))
        ])

    # Standard image processing for all data
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Return a custom transform that handles both img and target
    return CustomTransform(T.Compose(transforms))


# In[70]:


def custom_collate(batch):
    images = []
    labels = []

    for img, label in batch:
        # Add a batch size if the img does not have a batch size
        if img.dim() == 3:  # Checking that img has the form (C, H, W)
            img = img.unsqueeze(0)  # We add the size for the batch
        images.append(img)
        labels.append(label)

    return torch.cat(images), labels  # We combine the images into one tensor


# In[27]:


FOLD = 0
N_FOLDS = 5

kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df)  

def get_fold():
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]

train_df, valid_df = get_fold()

# Nous stockons les ensembles de données de formation et de validation au format CSV
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("valid_data.csv", index=False)


# In[72]:


# Load the training and validation data from CSV files
train_df = pd.read_csv("train_data.csv")
valid_df = pd.read_csv("valid_data.csv")
image_dir='./train'


# In[74]:


# Initialize the FashionDataset for training

transforms = get_transform(train=True)

train_dataset = FashionDataset(
    image_dir='./train',
    image_df=train_df,  # DataFrame for training data
    height=512, 
    width=512,   
    transforms=transforms
)
# Initialize the FashionDataset for validation
valid_dataset = FashionDataset(
    image_dir='./train',  
    image_df=valid_df,  # DataFrame for validation data
    height=512,
    width=512,
    transforms=transforms
)
#DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=custom_collate
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=custom_collate
)

# Check if data loading works correctly
print("Train DataLoader:", next(iter(train_loader)))
print("Valid DataLoader:", next(iter(valid_loader)))


# In[ ]:





# 

# # PyTorch Lightning

# In[33]:


class MaskRCNNTrainer(pl.LightningModule):
    def __init__(self, model, train_loader, learning_rate=2e-3):
        super(MaskRCNNTrainer, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.learning_rate = learning_rate

    def train_dataloader(self):
        return self.train_loader  # Return the DataLoader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Log losses
        self.log('train_loss', losses)

        return losses



# Logger for TensorBoard
TENSOR = "./logs"
logger = TensorBoardLogger(TENSOR, name="MaskRCNN_Training")

# Load model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# Exclude unwanted keys
weights = model.state_dict()
exclude_keys = [
    "roi_heads.box_predictor.cls_score",
    "roi_heads.box_predictor.bbox_pred",
    "roi_heads.mask_predictor.conv5_mask",
]

for key in exclude_keys:
    if key in weights:
        del weights[key]

# Load model weights
model.load_state_dict(weights, strict=False)

# Set up the predictors
num_classes = len(label_names) + 1  # Include background

# Get the number of input features for the box predictor
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Get the number of input channels for the mask predictor
in_channels_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
dim_reduced = 256  # You can choose a value based on your model architecture
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels_mask, dim_reduced, num_classes=num_classes)

# Freeze all layers except the predictors
for param in model.parameters():
    param.requires_grad = False

for param in model.roi_heads.box_predictor.parameters():
    param.requires_grad = True
for param in model.roi_heads.mask_predictor.parameters():
    param.requires_grad = True


# Create an instance of the MaskRCNNTrainer
trainer = MaskRCNNTrainer(model, train_loader)

# Create a PyTorch Lightning Trainer
lightning_trainer = Trainer(logger=logger)


# In[ ]:


# Start training
lightning_trainer.fit(trainer) # Fit the trainer with the MaskRCNNTrainer instance


# In[ ]:


get_ipython().system('pip install jupyter ipywidgets')


# In[ ]:


import ipywidgets as widgets
widgets.IntSlider()


# In[ ]:





# In[ ]:





# # PyTorch

# In[ ]:





# In[76]:


# Loading a model with pre-trained COCO weights
model = maskrcnn_resnet50_fpn(pretrained=True).to(device)
model.to(device)  # where device is set to "mps"

# Extracting weights from the model
weights = model.state_dict()

# Check device of model components
for name, param in model.named_parameters():
    print(f'{name}: {param.device}')
    
# Exclusion of unwanted keys
exclude_keys = [
    "roi_heads.box_predictor.cls_score",
    "roi_heads.box_predictor.bbox_pred",
    "roi_heads.mask_predictor.conv5_mask",
]

# Removing excluded keys from scales
for key in exclude_keys:
    if key in weights:
        del weights[key]

# Loading the model with the keys turned off
model.load_state_dict(weights, strict=False)

# Setting predictors for classes
num_classes = len(label_names) + 1  
# Replace predictor for classes
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes).to(device)

# Replace predictor for masks
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 512
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes).to(device)


# Freeze all layers except the top
for param in model.parameters():
    param.requires_grad = False

# Unfreezing the upper layers (box_predictor and mask_predictor)
for param in model.roi_heads.box_predictor.parameters():
    param.requires_grad = True
for param in model.roi_heads.mask_predictor.parameters():
    param.requires_grad = True

# Optimizer settings
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Зменшення LR кожних 5 епох
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #ExponentialLR


# Training cycle
num_epochs = 10  
writer = SummaryWriter()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0  # Initialize total_loss for the epoch
    for images, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        images = list(image.to(device) for image in images)
        #images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the correct device

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())# Calculation of total loss
        losses.backward()# 
        optimizer.step()# Weight update
        
        total_loss += losses.item()  # Accumulate total loss
        
    avg_loss = total_loss / len(train_loader)  # Average loss for the epoch
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Logging average losses
    
    print(f'Full Training Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')
    scheduler.step()  # Learning rate change

    # Save model
    torch.save(model.state_dict(), f'model_full_epoch_{epoch + 1}.pth')

writer.close()  # Close the TensorBoard writer


# In[ ]:





# In[ ]:


num_epochs_all_layers = 20  
writer = SummaryWriter()

for epoch in range(num_epochs_all_layers):
    model.train()
    total_loss = 0  # Initialize total_loss for the epoch
    
    for images, targets in tqdm(train_loader, desc=f'Full Training Epoch {epoch + 1}/{num_epochs_all_layers}', unit='batch'):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()  # Accumulate total loss

    avg_loss = total_loss / len(train_loader)  # Average loss for the epoch
    writer.add_scalar('Loss/train', avg_loss, epoch)  # Logging average losses
    
    print(f'Full Training Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')
    scheduler.step()  # Learning rate change

    # Save model
    torch.save(model.state_dict(), f'model_full_epoch_{epoch + 1}.pth')

writer.close()  # Close the TensorBoard writer


# In[ ]:


# Evaluation of the model on the validation data set
writer = SummaryWriter()
model.eval()  # We go to the validation mode
    val_loss = 0.0

    with torch.no_grad():  # We do not calculate gradients
        for images, targets in valid_loader:
            images = images.to(device)
            targets = targets.to(device)

            loss = model(images, targets)  # We calculate losses
            val_loss += loss.item()

    # Average loss per validation
    val_loss /= len(valid_loader)
    history['val_loss'].append(val_loss)

# Log your loss and accuracy
writer.add_scalar('Loss/train', loss, epoch)


# In[ ]:


print(f'Epoch [{epoch+1}/{additional_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# #information to resume training from the current state.
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'epoch': epoch,
#     'loss': loss
# }, 'model_checkpoint.pth')

# In[ ]:


best_epoch = np.argmin(history["val_loss"]) + 1
print(f"Best epoch: {best_epoch}")


# In[ ]:





# In[ ]:





# In[ ]:


# Loading weights of the best epoch
best_weights = torch.load(f'model_epoch_{best_epoch}.pth')
model.load_state_dict(best_weights)

# Saving the best epoch weights to a separate file
torch.save(model.state_dict(), 'best_model_weights.pth')
print("Best model weights saved as 'best_model_weights.pth'")


# In[ ]:


# Conversion to TorchScript
 traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
traced_model.save( 'model_scripted.pt' )

