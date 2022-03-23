import os
import time
from xml.dom import NotFoundErr
import torch
import torchvision
import random
import numpy as np
import pandas as pd
import wandb
from pycocotools.coco import COCO
from importlib import import_module
from args import Args
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from utils.util import collate_fn, get_train_transform, get_train_rotate_transform, get_log, wandb_init
from utils.dataset import CustomDataset_test, CustomDataset_train
from trainer.train import train_fn
from inference import inference_fn

def main(args, logger, wandb):

    test_mode = args["INFERENCE"]
    random_seed = args["RANDOM_SEED"]
    time_stamp = "_".join(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()).split(" "))

    if not test_mode:

        wandb_init(args, wandb, time_stamp)
        # 데이터셋 불러오기
        annotation = '../../dataset/train.json' # annotation 경로
        data_dir = '../../dataset' # data_dir 경로
        if args["AUGMENTATION"] == 'rotate' : 
            train_dataset = CustomDataset_train(annotation, data_dir, get_train_rotate_transform())
        else:
            train_dataset = CustomDataset_train(annotation, data_dir, get_train_transform())
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=args["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(device)
        
        # torchvision model 불러오기
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 11 # class 개수= 10 + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        
        opt_module = getattr(import_module("torch.optim"), args["OPTIMIZER"])
        optimizer = opt_module(params, lr=args["LEARNING_RATE"], weight_decay=args["WEIGHT_DECAY"]) 
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        num_epochs = args["NUM_EPOCHES"]
        
        # training
        train_fn(num_epochs, train_data_loader, optimizer, model, device, wandb, args, logger)

        return
    
    ## inference
    annotation = '../../dataset/test.json' # annotation 경로
    data_dir = '../../dataset' # dataset 경로
    test_dataset = CustomDataset_test(annotation, data_dir)
    score_threshold = 0.05
    check_point = args["SAVE_PATH"] # 체크포인트 경로
    

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    
    # torchvision model 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 11  # 10 class + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    model.load_state_dict(torch.load(check_point))
    model.eval()
    
    outputs = inference_fn(test_data_loader, model, device)
    prediction_strings = []
    file_names = []
    coco = COCO(annotation)

    # submission 파일 생성
    for i, output in enumerate(outputs):
        prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
            if score > score_threshold: 
                # label[1~10] -> label[0~9]
                prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                    box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    save_path = './submission'
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    submission.to_csv(f'./submission/torchvision_{args["OPTIMIZER"]}_{args["AUGMENTATION"]}_submission.csv', index=None)
    print(submission.head())
        
if __name__ == '__main__':
    args = Args().params

    if not args["INFERENCE"]:
        logger = get_log(args)
        logger.info("\n=========Training Info=========\n"
                    "Optimizer: {}".format(args['OPTIMIZER']) + "\n" +
                    "Batch size: {}".format(args['BATCH_SIZE']) + "\n" +
                    "Learning rate: {}".format(args['LEARNING_RATE']) + "\n" +
                    "Weight Decay: {}".format(args['WEIGHT_DECAY']) + "\n" +
                    "Augmentation : {}".format(args["AUGMENTATION"]) + "\n" + 
                    "===============================")
    else:
        logger = None

    random_seed = args['RANDOM_SEED']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main(args, logger, wandb)