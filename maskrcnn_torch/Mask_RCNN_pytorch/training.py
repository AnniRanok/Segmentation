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
                y = list(map(lambda y_el: {k: v.to(self.device) for k, v in y_el.items()}, y))
                loss, mean_iou = self.batch_update(x, y)

                # Update метрик
                if loss is not None:
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)

                iou_meter.add(mean_iou)

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
        self.model.train()  # Switch to training mode

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        outputs = self.model(x, y)
        loss = sum(l for l in outputs.values())
        loss.backward()
        self.optimizer.step()
        return loss


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
        self.model.eval()  # Switch the model to the evaluation mode

    @torch.no_grad()
    def batch_update(self, x, y):
        pred_outputs = self.model(x)
        iou_values = []
        iou_metric_fn = self.metrics['IoU']  # This is compute_iou

        if iou_metric_fn:
            # Extract predictions
            preds = [output['masks'].detach().cpu() for output in pred_outputs]
            boxes = [output['boxes'].detach().cpu() for output in pred_outputs]  # RoI boxes
            scores = [output['scores'].detach().cpu() for output in pred_outputs]  # Confidence scores

            # Extract targets
            targets = [t['masks'].detach().cpu() for t in y]
            target_boxes = [t['boxes'].detach().cpu() for t in y]  # Ground truth boxes

            # Process each image in the batch
            for batch_idx, (pred, box, score, target, target_box) in enumerate(
                    zip(preds, boxes, scores, targets, target_boxes)):
                if pred.numel() == 0 or target.numel() == 0:
                    logging.warning(f"Empty prediction or target for batch {batch_idx}. Skipping IoU calculation.")
                    continue

                # Apply dynamic confidence threshold
                max_conf = score.max().item()
                dynamic_threshold = 0.8 * max_conf  # Example: 80% of the maximum score
                logging.debug(f"Dynamic threshold for batch {batch_idx}: {dynamic_threshold:.4f}")

                # Confidence filtering
                high_conf_idx = score > dynamic_threshold
                pred = pred[high_conf_idx]
                box = box[high_conf_idx]
                logging.debug(f"After dynamic thresholding: pred.shape={pred.shape}, box.shape={box.shape}")

                if pred.numel() == 0:
                    logging.warning(f"No predictions with confidence > {dynamic_threshold:.4f}. Skipping batch.")
                    continue

                # Process each target box
                for t_box, t_mask in zip(target_box, target):
                    # Crop the target mask
                    x1, y1, x2, y2 = t_box.int()  # Target box coordinates
                    cropped_target = t_mask[y1:y2, x1:x2]
                    # visualize_mask(t_mask)
                    # visualize_mask(cropped_target)
                    # Crop the predicted masks using the same target box
                    ious = []
                    for p_mask in pred:
                        cropped_pred = p_mask[:, y1:y2, x1:x2]
                        # visualize_masks(p_mask)
                        # visualize_masks(cropped_pred)

                        # Resize predicted mask if necessary
                        if cropped_pred.shape[-2:] != cropped_target.shape[-2:]:
                            cropped_pred = F.interpolate(cropped_pred.unsqueeze(0), size=cropped_target.shape[-2:],
                                                         mode='bilinear', align_corners=False).squeeze(0)

                        cropped_pred = cropped_pred.squeeze(0)
                        # Compute IoU for the cropped masks
                        try:
                            iou = iou_metric_fn(cropped_target, (cropped_pred > 0.5))
                            ious.append(iou)
                        except ValueError as e:
                            logging.error(f"Error computing IoU for batch {batch_idx}: {e}")

                    # Store the best IoU for this target
                    if ious:
                        best_iou = max(ious)  # Take the best IoU for this target box
                        iou_values.append(best_iou)

        # Compute mean IoU
        mean_iou = np.mean(iou_values) if iou_values else 0.0
        return None, mean_iou
    

# IoU      
import torch.nn.functional as F
def compute_iou(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        import torch.nn.functional as F
        y_pred = F.interpolate(y_pred.unsqueeze(0).unsqueeze(0), size=y_true.shape[-2:], mode='bilinear', align_corners=False)
        y_pred = y_pred.squeeze(0).squeeze(0)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return intersection / (union + 1e-6)



# Train and Validation loops
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
    verbose=True,
)

# Learning cycle
optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)

best_iou = 0  # Initialize best_iou before training loop

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    train_logs = train_epoch.run(data_loader_train)
    print(f"Train logs: {train_logs}")

    torch.cuda.empty_cache()  # Free up memory

    val_logs = valid_epoch.run(data_loader_val)
    print(f"Validation logs: {val_logs}")

    torch.cuda.empty_cache()  # Free up memory

    scheduler.step()  # Update learning rate

    # Save best model
    current_iou = val_logs.get('IoU', 0)
    if current_iou > best_iou:
        best_iou = current_iou
        model_path = f'best_model_epoch{epoch+1}_iou{current_iou:.4f}.pth'
        torch.save(model_ft.state_dict(), model_path)
        print(f"Model saved at {model_path} with IoU: {current_iou:.4f}!")



