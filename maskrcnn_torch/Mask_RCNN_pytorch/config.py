import os
from os import path
from pathlib import Path
import json

IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CLASSES = 46
BATCH_SIZE = 16
N_WORKERS = 1
NUM_EPOCHS = 30
DEVICE = DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FOLD = 0  
N_FOLDS = 5  

DATA_PATH = "/content/drive/My Drive/"
df_path_train = os.path.join(DATA_PATH, "train2.csv")
json_file_path = os.path.join(DATA_PATH, "label_descriptions1.json")
root_path_train = './train'
image_dir = "./train"
image_extension = ".jpg"  
CHECKPOINT_PATH = './checkpoints/'

