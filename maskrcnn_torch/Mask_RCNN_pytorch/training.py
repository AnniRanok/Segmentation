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
from torch.optim import SGD, Adam
from pathlib import Path
import shutil
from sklearn.model_selection import KFold, train_test_split  
from glob import glob
from PIL import Image
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_EPOCHS
from dataset import data_loader_train, data_loader_val
from model import create_model
from torch.optim.lr_scheduler import StepLR, CyclicLR


class MyEpoch(smp.utils.train.Epoch):
    def _to_device(self):
        self.model.to(self.device)

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = smp.utils.meter.AverageValueMeter()
        iou_meter = smp.utils.meter.AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x = list(map(lambda x_el: x_el.to(self.device), x))
                y = list(map(lambda y_el: {k:v.to(self.device) for k,v in y_el.items()}, y))
                loss, iou_value = self.batch_update(x, y)

                # update loss logs and IoU
                if loss is not None:
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    
                if iou_value is not None:
                    iou_value = iou_value.detach().cpu().numpy()
                    iou_meter.add(iou_value)

                logs.update({'loss': loss_meter.mean, 'IoU': iou_meter.mean})

                if self.verbose:
                    s = self._format_logs(logs)
                    current_time = datetime.now().strftime('%H:%M:%S')
                    s = f"{s} | {current_time}"
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
        self.model.train()  #Switch to training mode
        

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.model(x, y)
        loss = sum(l for l in outputs.values())
        loss.backward()
        self.optimizer.step()
        iou_value = self.compute_iou(outputs['masks'], y['masks'])
        return loss, iou_value


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
        self.metric_values = {key: [] for key in (metrics or {}).keys()}


    def on_epoch_start(self):
        self.model.eval()  # Switch the model to the evaluation mode
     

    @torch.no_grad()
    def batch_update(self, x, y):
        outputs = self.model(x, y)
        loss = sum(l for l in outputs.values())
        #outputs = self.model(x)
        #loss = None

        if self.metrics:
            for name, metric_fn in self.metrics.items():
                preds = outputs['masks'].detach().cpu() > 0.5  
                targets = y['masks'].detach().cpu()
                value = metric_fn(targets.numpy(), preds.numpy())
                self.metric_values[name].append(value)
        return loss


    def on_epoch_end(self):
        metrics_mean = {key: np.mean(values) for key, values in self.metric_values.items()}
        self.logs.update(metrics_mean)
        print(f"End of epoch: {metrics_mean}")  



# IoU
def compute_iou(y_true, y_pred):
    device = y_true.device
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection / (union + 1e-6)  


train_epoch = TrainEpoch(
    model_ft,
    loss=None,
    metrics=None,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = ValidEpoch(
    model=model_ft,
    loss=None,
    metrics={'IoU': compute_iou}, 
    device=DEVICE,
    verbose=True
)

# Learning cycle
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

best_iou = 0  # Initialize best_iou before training loop

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
   
    train_logs = train_epoch.run(data_loader_train)
    print(f"Train logs: {train_logs}")

    torch.cuda.empty_cache() # Free up memory

    val_logs = valid_epoch.run(data_loader_val)
    print(f"Validation logs: {val_logs}")

    torch.cuda.empty_cache() # Free up memory

    scheduler.step()  # Update learning rate

    
    current_iou = val_logs.get('IoU', 0)
    if current_iou > best_iou:
        best_iou = current_iou
        model_path = f'best_model_epoch{epoch+1}_iou{current_iou:.4f}.pth'
        torch.save(model_ft.state_dict(), model_path)
        print(f"Model saved at {model_path} with IoU: {current_iou:.4f}!")

