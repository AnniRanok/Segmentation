from torch.utils.data import DataLoader
from config import Config
from model import MaskRCNNLightning  
from dataset import ImgDataModule  
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

def train():
    # Initialize DataModule
    image_df = pd.read_csv(Config.IMAGE_CSV)
    train_df, valid_df = split_data_into_folds(image_df, n_folds=Config.N_FOLDS, fold=Config.FOLD)
    data_module = ImgDataModule(train_df=train_df, valid_df=valid_df, 
                                batch_size=Config.BATCH_SIZE, 
                                num_workers=Config.N_WORKERS, 
                                height=Config.IMG_HEIGHT, 
                                width=Config.IMG_WIDTH)
    
    
    # Model initialization
    model = MaskRCNNLightning(num_classes=Config.NUM_CLASSES)
    
    # Logger initialization
    logger = TensorBoardLogger("tb_logs", name="maskrcnn")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=Config.CHECKPOINT_MONITOR,  # We are watching the loss on validation
        dirpath=Config.CHECKPOINT_PATH,  # Path where to store checkpoints
        filename="maskrcnn-{epoch:02d}-{val_iou:.2f}",
        save_top_k=3,  # we take care of the top 3 best checkpoints
        mode=Config.CHECKPOINT_MODE,  # The model with the minimum loss value will be kept
    )
    
    early_stop_callback = EarlyStopping(
        monitor=Config.CHECKPOINT_MONITOR,  # We are watching the loss on validation
        patience=Config.EARLY_STOPPING_PATIENCE,  # How many epochs are tolerated if valid loss does not improve
        verbose=True,  # Additional messages
        mode=Config.CHECKPOINT_MODE,  # Stop when loss does not improve
    )

    # Trainer initialization
    trainer = pl.Trainer(
        gpus=1 if Config.DEVICE == 'cuda' else 0, 
        max_epochs=Config.NUM_EPOCHS,  
        logger=logger,  # Logging in TensorBoard
        callbacks=[checkpoint_callback, early_stop_callback],  # callbacks
        precision=16,  # half precision (FP16) (to reduce the use of memory on the GPU)
        accumulate_grad_batches=2,  # Multiple gradient stacking for large packets (the gradients for each batch are not updated immediately, but are accumulated over several batches)
    )
    
    # Start training
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train()

 
