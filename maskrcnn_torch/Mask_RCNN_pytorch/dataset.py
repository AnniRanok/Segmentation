import pandas as pd
import os
import shutil
import collections
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import numpy as np
import torchvision
from sklearn.model_selection import KFold
from config import FOLD, N_FOLDS, image_dir, root_path_train, image_extension, IMG_HEIGHT, IMG_WIDTH


class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df, height, width, transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = df
        self.height = height
        self.width = width
        self.image_info = collections.defaultdict(dict)

        self.df.loc[:, 'CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = self.df.groupby('ImageId')[['EncodedPixels', 'CategoryId']].agg(lambda x: list(x)).reset_index()
        size_df = self.df.groupby('ImageId')[['Height', 'Width']].mean().reset_index()
        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        # Saving image information
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row['ImageId']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if not os.path.isfile(image_path):
                logging.info(f"Warning: {image_path} does not exist.")
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
        """Decoding the mask in RLE format."""
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
            sub_mask = self.rle_decode(annotation, (orig_height, orig_width))
            sub_mask = Image.fromarray(sub_mask).resize((self.width, self.height), resample=Image.NEAREST)
            mask[m] = np.array(sub_mask)

        masks_list = [mask[i, :, :] for i in range(mask.shape[0])]

        # Application of augmentations
        if self.transforms:
            augmented = self.transforms(image=np.array(img), masks=masks_list)
            img = augmented["image"]
            masks_list = augmented["masks"]

        # Generation of bounding boxes, labels, area, iscrowd, masks
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

        # In the absence of valid boxes
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


# KFold Configuration
kf = KFold(n_splits=N_FOLDS, random_state=42, shuffle=True)
splits = kf.split(image_df)  # Assuming image_df is already defined

# Function for obtaining training and validation data
def get_fold():
    for i, (train_index, valid_index) in enumerate(splits):
        if i == FOLD:
            return image_df.iloc[train_index], image_df.iloc[valid_index]

# We get a training and validation dataframe
train_df, valid_df = get_fold()

# Create new directories for training and validation images
train_dir = os.path.join(root_path_train, 'train_images')
valid_dir = os.path.join(root_path_train, 'valid_images')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Move images to the respective directories
def move_images(df, source_dir, target_dir, extension=".jpg"):
    for index, row in df.iterrows():
        img_filename = row['ImageId'] + extension  # Add the extension to the file name
        img_path = os.path.join(source_dir, img_filename)  # The path to the original image
        if os.path.exists(img_path):
            shutil.move(img_path, target_dir)  # Move the image to a new directory

move_images(train_df, image_dir, train_dir, extension=image_extension)
move_images(valid_df, image_dir, valid_dir, extension=image_extension)

# Saving CSV files for training and validation
train_df.to_csv("train_data.csv", index=False)
valid_df.to_csv("valid_data.csv", index=False)


# Custom collate function
def custom_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    return images, labels



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



# Create dataset and dataloaders
datset_train = FashionDataset(image_dir=train_dir,
                              df=train_df,
                              height=IMG_HEIGHT,
                              width=IMG_WIDTH,
                              augmentations=train_transforms
                             )

datset_val = FashionDataset(image_dir=valid_dir,
                            df=valid_df,
                            height=IMG_HEIGHT,
                            width=IMG_WIDTH,
                            augmentations=val_transforms
                           )

data_loader_train = torch.utils.data.DataLoader(
    datset_train, batch_size=16, shuffle=True, num_workers=6,
    collate_fn=custom_collate)

data_loader_val = torch.utils.data.DataLoader(
    datset_val, batch_size=16, shuffle=True, num_workers=6,
    collate_fn=custom_collate)
