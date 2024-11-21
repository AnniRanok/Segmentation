import os
import numpy as np
import pandas as pd
import torch
import cv2
import sys
import collections
import albumentations as albu
import torchvision
import segmentation_models_pytorch as smp #import segmentation_models_pytorch==0.2.0 
from torch.optim import SGD
from pathlib import Path
import shutil
from sklearn.model_selection import KFold, train_test_split  
from torch.utils.data import DataLoader
from glob import glob
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS
from dataset import data_loader_train, data_loader_val
from model import create_model
from torch.optim.lr_scheduler import StepLR


class MyEpoch(smp.utils.train.Epoch):
    def _to_device(self):
        self.model.to(self.device)

    def run(self, dataloader):
        self.on_epoch_start()
        logs = {}
        loss_meter = smp.utils.meter.AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x = list(map(lambda x_el: x_el.to(self.device), x))
                y = list(map(lambda y_el: {k: v.to(self.device) for k, v in y_el.items()}, y))
                loss = self.batch_update(x, y)

                # Update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {'loss': loss_meter.mean}
                logs.update(loss_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

class TrainEpoch(MyEpoch):
    def __init__(self, model, loss, metrics, optimizer, device='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        loss = self.model(x, y)
        loss = sum(l for l in loss.values())  # Assuming loss is a dict
        loss.backward()
        self.optimizer.step()
        return loss

# Train epoch initialization
loss_fn = smp.losses.DiceLoss(mode='multiclass')
train_epoch = TrainEpoch(
    model=model_ft,
    loss=loss_fn,  
    metrics={'IoU': compute_iou},  # Optional: Add IoU metric for training logs
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

class ValidEpoch(MyEpoch):
    def __init__(self, model, loss=None, metrics=None, device='cuda', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    @torch.no_grad()
    def batch_update(self, x, y):
        outputs = self.model(x, y)
        loss = sum(l for l in outputs.values())  # Aggregate loss components
        return loss

# Function for IoU metric ( y_true, y_pred must have the type as torch.Tensor)


#def compute_iou(y_true, y_pred):
    #if not isinstance(y_true, torch.Tensor):
       # y_true = torch.tensor(y_true, dtype=torch.float32)
    #if not isinstance(y_pred, torch.Tensor):
        #y_pred = torch.tensor(y_pred, dtype=torch.float32)

    #y_true = y_true.flatten()
    #y_pred = y_pred.flatten()
    #return jaccard_score(y_true, y_pred, average='macro')


def compute_iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return jaccard_score(y_true, y_pred, average='macro')


# Valid epoch initialization with IoU metric
valid_epoch = ValidEpoch(
    model=model_ft,
    loss=None,  # Use the appropriate loss function
    metrics={'IoU': compute_iou},  # IoU metric
    device=DEVICE,
    verbose=True
)


def train_model():
    model = create_model(num_classes=NUM_CLASSES + 1)
    model.to(DEVICE)
    
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    
    best_iou = 0.0
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"Current learning rate at the start of epoch {epoch + 1}: {current_lr}")
        
        # Train phase
        train_logs = train_epoch.run(data_loader_train)
        print(f"Train logs: {train_logs}")
        
        # Validation phase
        val_logs = valid_epoch.run(data_loader_val)
        print(f"Validation logs: {val_logs}")

        # Scheduler step
        scheduler.step()
        #scheduler.step(val_logs['IoU'])  

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
            print(f"Current learning rate after epoch {epoch + 1}: {current_lr}")

        
        # Save the best model based on IoU
        if val_logs['IoU'] > best_iou:
            best_iou = val_logs['IoU']
            torch.save(model.state_dict(), f'best_model_epoch{epoch+1}_iou{val_logs["IoU"]:.4f}.pth')
            print("Model saved!")


