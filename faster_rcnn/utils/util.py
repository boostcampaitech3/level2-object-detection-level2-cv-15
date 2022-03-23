import time
import logging
import os
import wandb

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        # A.Normalize(
        #     mean = [0.4849077, 0.4604577, 0.43180206],
        #     std = [0.042062894, 0.038401194, 0.039760303],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_train_rotate_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Flip(p=0.5),
        A.RandomRotate90(p=1.0),
        # A.Normalize(
        #     mean = [0.4849077, 0.4604577, 0.43180206],
        #     std = [0.042062894, 0.038401194, 0.039760303],
        #     max_pixel_value=255.0
        # ),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def collate_fn(batch):
    return tuple(zip(*batch))

def get_log(args):
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    
    optimizer = args["OPTIMIZER"]
    augmentation = args["AUGMENTATION"]

    if not os.path.exists("log"):
        os.makedirs("log")
        
    stream_handler = logging.FileHandler(f"log/{optimizer}_{augmentation}_{time.strftime('%m%d-%H-%M-%S')}.txt", mode='w', encoding='utf8')
    logger.addHandler(stream_handler)
    
    return logger

def wandb_init(args, wandb, time_stamp):
    
    optimier = args["OPTIMIZER"]
    augmentation = args["AUGMENTATION"]

    wandb.init(project="test-project", entity="winner", name = f"torchvision_{optimier}_{augmentation}")

    wandb.config.update({
    "Optimizer": args["OPTIMIZER"],
    "learning_rate": args["LEARNING_RATE"],
    "Weight decay": args["WEIGHT_DECAY"],
    "Num_epoch" : args["NUM_EPOCHES"],
    "Batch_size" : args["BATCH_SIZE"],
    "Augmentation" : args["AUGMENTATION"]
    })
    