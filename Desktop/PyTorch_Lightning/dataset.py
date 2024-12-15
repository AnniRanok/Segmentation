
import pytorch_lightning as pl
import pandas as pd
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from datasets import split_data_into_folds, initialize_data_loaders
from config import Config
import os
import logging
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

def visualize_masks(masks, title="Masks", cmap="gray"):
    """
    Visualizes a tensor of N masks in a grid.

    Args:
        masks (torch.Tensor or np.ndarray): Tensor or array of shape (N, H, W).
        title (str): Title of the visualization.
        cmap (str): Colormap to use (default is 'gray').
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()  # Convert to numpy array if it's a PyTorch tensor

    num_masks = masks.shape[0]
    grid_size = math.ceil(math.sqrt(num_masks))  # Grid dimensions (square root of N)

    plt.figure(figsize=(15, 15))
    for i in range(num_masks):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(masks[i], cmap=cmap)
        plt.axis("off")
        plt.title(f"Mask {i + 1}")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    


    
class FashionDataset(Dataset):
    def __init__(self, image_dir, df, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.height = height
        self.width = width
        self.image_info = defaultdict(dict)

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = self.df.groupby('ImageId')[['EncodedPixels', 'CategoryId']].agg(lambda x: list(x)).reset_index()
        size_df = self.df.groupby('ImageId')[['Height', 'Width']].mean().reset_index()
        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        # Save image information
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if not os.path.isfile(image_path):
                logging.warning(f"Warning: {image_path} does not exist.")
                continue

            self.image_info[index] = {
                "image_id": image_id,
                "image_path": image_path,
                "width": self.width,
                "height": self.height,
                "labels": row["CategoryId"],
                "orig_height": row["Height"],
                "orig_width": row["Width"],
                "annotations": row["EncodedPixels"]
            }


    def rle_decode(self, rle, shape):
        """Decodes the RLE mask."""
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        s = list(map(int, rle.split()))
        starts = s[0::2]
        lengths = s[1::2]
        for start, length in zip(starts, lengths):
            img[start:start + length] = 1
        return img.reshape(shape).T

    def __getitem__(self, idx):
        info = self.image_info[idx]
        img_path = info["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        # Creating masks
        mask = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        for m, annotation in enumerate(info['annotations']):
            orig_height, orig_width = int(float(info['orig_height'])), int(float(info['orig_width']))
            sub_mask = self.rle_decode(annotation, (orig_width, orig_height))
            sub_mask = Image.fromarray(sub_mask).resize((self.width, self.height), resample=Image.NEAREST)
            mask[m] = np.array(sub_mask)

        masks_list = [mask[i, :, :] for i in range(mask.shape[0])]

        # Application of augmentations
        if self.transforms:
            augmented = self.transforms(image=np.array(img), masks=masks_list)
            img = augmented["image"]
            masks_list = augmented["masks"]

        # Generate bounding boxes, labels, area, iscrowd, masks
        boxes = []
        labels = []
        area = []
        iscrowd = []
        masks = []

        for i, mask_2d in enumerate(masks_list):
            pos = np.where(mask_2d > 0)
            if len(pos[0]) == 0:
                continue
            xmin, ymin = np.min(pos[1]), np.min(pos[0])
            xmax, ymax = np.max(pos[1]), np.max(pos[0])

            # Adding valid bounding boxes
            if abs(xmax - xmin) > 20 and abs(ymax - ymin) > 20:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(info["labels"][i]))
                area.append((xmax - xmin) * (ymax - ymin))
                masks.append(mask_2d)
                iscrowd.append(0)

        # Default box when no valid boxes
        if len(labels) == 0:
            boxes.append([0, 0, 20, 20])
            labels.append(0)
            masks.append(mask[0, :, :])
            area.append(400)
            iscrowd.append(0)

        # Target preparation
        image_id = torch.tensor([idx])
        masks_stack = np.stack(masks, axis=0)
        target_masks = torch.as_tensor(masks_stack, dtype=torch.uint8)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": target_masks,
            "image_id": image_id,
            "area": torch.as_tensor(area, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }

        return img, target

    def __len__(self):
        return len(self.image_info)


def custom_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    return images, labels

    
def split_data_into_folds(image_df, n_folds=5, fold=0, random_state=42, save_to_csv=True):
    """
    Splits the dataset into several folds for cross-validation and saves the selected fold to CSV.

    :param image_df: DataFrame with image data
    :param n_folds: Number of folds to split
    :param fold: The fold number to return as training and validation
    :param random_state: Seed for reproducibility of the partitioning results
    :param save_to_csv: If True, saves training and validation folds to CSV
    :return: DataFrame with training and validation data for the selected fold
    """
    files = os.listdir(".")
    train_file = next((f for f in files if f.startswith("train_data_fold_")), None)
    valid_file = next((f for f in files if f.startswith("valid_data_fold_")), None)

    if train_file and valid_file:
        return pd.read_csv(train_file), pd.read_csv(valid_file)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = kf.split(image_df)

    for i, (train_index, valid_index) in enumerate(splits):
        if i == fold:
            train_df = image_df.iloc[train_index]
            valid_df = image_df.iloc[valid_index]

            if save_to_csv:
                train_df.to_csv(f"train_data_fold_{fold}.csv", index=False)
                valid_df.to_csv(f"valid_data_fold_{fold}.csv", index=False)

            return train_df, valid_df

    raise ValueError(f"Fold {fold} is out of range for {n_folds} folds.")



class ImgDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df, batch_size=Config.BATCH_SIZE, num_workers=Config.N_WORKERS,
                 height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height = height
        self.width = width
        if train_df.empty or valid_df.empty:
            raise ValueError("Training or validation dataframe is empty!")
        
    def prepare_data(self):
        pass 
        
    def setup(self, stage=None):
        train_transforms = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        val_transforms = albu.Compose([
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        self.train_dataset = FashionDataset(
            image_dir=Config.IMAGE_DIR,
            df=self.train_df,
            height=self.height,
            width=self.width,
            transforms=train_transforms
        )

        self.valid_dataset = FashionDataset(
            image_dir=Config.IMAGE_DIR,
            df=self.valid_df,
            height=self.height,
            width=self.width,
            transforms=val_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate
        )
