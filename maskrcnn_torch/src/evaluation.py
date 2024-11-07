import torch
from src.utils import load_best_weights
from src.config import Config
from torchvision.ops import box_iou  
import numpy as np

def evaluate_model(model, valid_loader, device):
    model.eval()  
    with torch.no_grad():  
        all_predictions = []
        all_targets = []
        all_masks_pred = []
        all_masks_true = []
        all_labels_pred = []
        all_labels_true = []
        
        for images, targets in valid_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # We get predictions from the model
            predictions = model(images)  # [boxes, labels, scores, masks]
            
            # For each image, we store predictions and real labels
            for prediction, target in zip(predictions, targets):
                # Scale the boxes
                pred_boxes = prediction['boxes']
                target_boxes = target['boxes']
                
                # Scale the bounding rectangles
                old_width = target['orig_width']
                old_height = target['orig_height']
                pred_boxes = scale_boxes(pred_boxes, old_width, old_height, Config.IMG_WIDTH, Config.IMG_HEIGHT)
                target_boxes = scale_boxes(target_boxes, old_width, old_height, Config.IMG_WIDTH, Config.IMG_HEIGHT)

                # IoU (To calculate the IoU between rectangles)
                iou = box_iou(pred_boxes, target_boxes)
                all_predictions.append(pred_boxes)
                all_targets.append(target_boxes)

                # We keep the predicted and real masks
                all_masks_pred.append(prediction['masks'])
                all_masks_true.append(target['masks'])

                # We save predicted and real labels
                all_labels_pred.append(prediction['labels'])
                all_labels_true.append(target['labels'])

        # We return the average value of IoU to evaluate the quality of the model
        mean_iou = np.mean([iou.max().item() for iou in all_predictions])
        print(f"Mean IoU: {mean_iou:.4f}")
        
        # We save the collected metrics
        metrics = {
            "predictions": all_predictions,
            "targets": all_targets,
            "masks_pred": all_masks_pred,
            "masks_true": all_masks_true,
            "labels_pred": all_labels_pred,
            "labels_true": all_labels_true,
            "mean_iou": mean_iou
        }

        return metrics

#IoU > 0.5-0.75

def scale_boxes(boxes, old_width, old_height, new_width, new_height):
    """
    Scales bounding rectangles from one size to another.
    
    :param boxes: Input bounding boxes (tensor), dimension (N, 4)
    :param old_width: The original width of the image
    :param old_height: The original height of the image
    :param new_width: The new width
    :param new_height: The new height
    
    :return: Scaled bounding rectangles
    """
    scale_x = new_width / old_width
    scale_y = new_height / old_height
    boxes_scaled = boxes.clone()
    boxes_scaled[:, [0, 2]] *= scale_x  # We scale the coordinates by X (xmin, xmax)
    boxes_scaled[:, [1, 3]] *= scale_y  # We scale the coordinates in Y (ymin, ymax)
    return boxes_scaled
