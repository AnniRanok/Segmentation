echo "class Config:
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    NUM_CLASSES = 47  # 46 + 1 background
    BATCH_SIZE = 16
    N_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LABEL_DESC_FILE = 'label_descriptions.json'
    IMAGE_CSV = 'image_df.csv'
    IMAGE_DIR = './train'
    MODEL_SAVE_DIR = 'models/'
    TRAIN_DATA_CSV = 'train_data.csv'
    VALID_DATA_CSV = 'valid_data.csv'
    FOLD = 0
    N_FOLDS = 5
    val_loss = 0.0
