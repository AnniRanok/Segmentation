import torch
from torch.utils.tensorboard import SummaryWriter
from src.config import Config
from src.data import get_dataloaders
from src.model import get_instance_segmentation_model
from src.train import train_one_epoch
from src.validate import validate
from src.utils import save_checkpoint
import pandas as pd
from engine import train_one_epoch, evaluate

def main():
    
    image_df = pd.read_csv(Config.TRAIN_DATA_CSV)

    train_df, valid_df = split_data_into_folds(image_df, n_folds=5, fold=0)

    train_loader, valid_loader = initialize_data_loaders(train_df, valid_df, batch_size=Config.BATCH_SIZE, 
                                                          height=Config.IMG_HEIGHT, width=Config.IMG_WIDTH)

    device = Config.DEVICE
    model = get_instance_segmentation_model(Config.NUM_CLASSES).to(device)  
    
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=Config.LR, 
        momentum=Config.MOMENTUM, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    writer = SummaryWriter()  
    best_loss = float("inf")  

    num_epochs = Config.NUM_EPOCHS
    for epoch in range(num_epochs):
        
        metric_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)

        coco_evaluator = evaluate(model, valid_loader, device)

        writer.add_scalar('Loss/train', metric_logger.meters['loss'].global_avg, epoch)
        writer.add_scalar('Loss/val', coco_evaluator.coco_eval['bbox'].stats[0], epoch)  

        val_loss = coco_evaluator.coco_eval['bbox'].stats[0]
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, checkpoint_path=Config.CHECKPOINT_PATH)
            print(f"Model saved at epoch {epoch + 1} with best validation loss {best_loss:.4f}")
        
        scheduler.step()

    writer.close()

if __name__ == "__main__":
    main()
