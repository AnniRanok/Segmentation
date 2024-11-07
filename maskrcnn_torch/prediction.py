#!/usr/bin/env python
# coding: utf-8


import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
num_classes = len(label_names) + 1  

# Model initialization
model = maskrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 512
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

# Loading saved weights
model.load_state_dict(torch.load('best_model_weights.pth', map_location=device))
model.to(device)
model.eval()

# Function for prediction
def predict(image_path, model, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred_boxes = predictions[0]['boxes']
    pred_scores = predictions[0]['scores']
    pred_labels = predictions[0]['labels']
    pred_masks = predictions[0]['masks']

    selected_indices = pred_scores > threshold
    pred_boxes = pred_boxes[selected_indices].cpu().numpy()
    pred_masks = pred_masks[selected_indices].cpu().numpy()
    pred_labels = pred_labels[selected_indices].cpu().numpy()

    return pred_boxes, pred_masks, pred_labels


image_path = "path_to__image.jpg"  
boxes, masks, labels = predict(image_path, model, device)

# Visualization of results
def plot_predictions(image_path, boxes, masks, labels):
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    ax = plt.gca()
    
    for box, mask, label in zip(boxes, masks, labels):
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                             fill=False, color="red", linewidth=2)
        ax.add_patch(rect)
        
        # Mask as a transparent overlay
        mask = mask[0, :, :]  
        mask = np.where(mask > 0.5, 1, 0).astype(np.uint8)  # Mask binarization
        plt.imshow(mask, alpha=0.5)
        
        # Text with label
        ax.text(box[0], box[1], f'{label_names[label-1]}', color='blue', 
                bbox=dict(facecolor='white', alpha=0.5))

    plt.axis("off")
    plt.show()


plot_predictions(image_path, boxes, masks, labels)

