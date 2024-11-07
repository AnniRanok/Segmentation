import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import collections
import ast
import torchvision.transforms as T
from src.config import Config
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import KFold

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


# Initialize the FashionDataset for training

def initialize_data_loaders(train_df, valid_df, batch_size=16, height=512, width=512):
    """
     Initializes the DataLoader for training and validation data.

    :param train_df: DataFrame for training data
    :param valid_df: DataFrame for validation data
    :param batch_size: Batch size
    :param height: Height of images after scaling
    :param width: Width of images after scaling

    :return: train_loader, valid_loader
    """

    transform = get_transform(train=True)

    train_dataset = FashionDataset(
        image_dir='./train',
        image_df=train_df, 
        height=height, 
        width=width,   
        transforms=transform
    )

    valid_dataset = FashionDataset(
        image_dir='./train',  
        image_df=valid_df,  
        height=height,
        width=width,
        transforms=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate
    )

    return train_loader, valid_loader


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


