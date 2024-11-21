from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import segmentation_models_pytorch as smp #import segmentation_models_pytorch==0.2.0 
import numpy as np
import pandas as pd
import torch
import collections
import albumentations as albu
import torchvision
from config import json_file_path, NUM_CLASSES, DEVICE



num_classes = NUM_CLASSES + 1
device = torch.device(DEVICE)

model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model_ft.roi_heads.box_predictor.cls_score.in_features
model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

for param in model_ft.parameters():
    param.requires_grad = True