import os
import torch

class Config:
    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    NUM_CLASSES = 46
    BATCH_SIZE = 16
    N_WORKERS = 4
    NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 40))  
    LR = float(os.getenv('LR', 2e-3))  # Learning rate
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    FOLD = 0
    N_FOLDS = 5
    IMAGE_CSV = os.getenv('IMAGE_CSV', 'train2.csv')  # Path to image CSV
    LABEL_DESC_FILE = 'label_descriptions.json'  # Class descriptions
    IMAGE_DIR = os.getenv('IMAGE_DIR', './train')  # Path to images
    IMAGE_EXTENSION = ".jpg"   
    CHECKPOINT_PATH = './checkpoints/'
    LOG_PATH = './logs/'
    EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
    CHECKPOINT_MONITOR = 'val_iou'  # Metric to monitor for checkpointing
    CHECKPOINT_MODE = 'max' # Mode for checkpointing: 'min' or 'max'
    SEED = 42  # Random seed
