import torch
import pytorch_lightning as pl
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, FastRCNNPredictor, MaskRCNNPredictor
from config import Config
from torchvision.transforms import functional as F

class MaskRCNNLightning(pl.LightningModule):
    def __init__(self, num_classes: int):
        super(MaskRCNNLightning, self).__init__()
        
        # Model creation
        self.model = self.create_model(num_classes)
        
    def create_model(self, num_classes: int):
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        # Change the box predictor (for detection)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Change the mask predictor (for segmentation)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        return model.to(Config.DEVICE)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        
        # Sum up all the losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Log losses
        self.log("train_loss", losses, prog_bar=True, logger=True)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        
        # Sum up all the losses
        losses = sum(loss for loss in loss_dict.values())
        
        # Calculate IoU
        iou = self.compute_iou(loss_dict, targets)
        
        # Log losses and IoU
        self.log("val_loss", losses, prog_bar=True, logger=True)
        self.log("val_iou", iou, prog_bar=True, logger=True)

        return {"loss": losses, "iou": iou}
    
    def validation_epoch_end(self, outputs):
        # Calculate the average IoU and loss for the entire validation set
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_iou = torch.stack([x["iou"] for x in outputs]).mean()
        
        self.log("avg_val_loss", avg_loss)
        self.log("avg_val_iou", avg_iou)

    def compute_iou(self, outputs, targets):
        """
        Compute Intersection over Union (IoU) for the predicted masks and ground truth masks.
        """
        iou_values = []
        
        for output, target in zip(outputs, targets):
            # Predicted masks
            pred_masks = output['masks']
            pred_masks = (pred_masks > 0.5).float()  # Binary masks
            
            # Ground truth masks
            true_masks = target['masks']
            
            # Calculate IoU for each instance
            for pred_mask, true_mask in zip(pred_masks, true_masks):
                iou = self._calculate_single_iou(pred_mask, true_mask)
                iou_values.append(iou)
        
        return torch.tensor(iou_values).mean()

    def _calculate_single_iou(self, pred_mask, true_mask):
        """
        Helper function to calculate IoU between a single predicted mask and a ground truth mask.
        """
        intersection = torch.sum(pred_mask * true_mask)
        union = torch.sum(pred_mask) + torch.sum(true_mask) - intersection
        
        return intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

    def configure_optimizers(self):
        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.LR)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=2000)
        
        return [optimizer], [scheduler]
