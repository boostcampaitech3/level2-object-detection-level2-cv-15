import torch
import argparse

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for Mask Classification train/test')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)') 
    parser.add_argument('--resize', type=int, nargs="+", default=(312, 312), help='Resize input image')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type = float, default = 0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for data split')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--inference', type=bool, default=False, help='Inference single model')
    # parser.add_argument('--earlystopping_patience', type=int, default=6)
    # parser.add_argument('--scheduler_patience', type=int, default=2)
    parser.add_argument('--save_path', default=False, help='Trained model path')
    # parser.add_argument('--save_logits', default=True, help='Save model logits along answer file')
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--num_epoches', type = int, default = 50)
    parser.add_argument('--augmentation', type=str , default = None, help = 'write rotate if you want') 


    parse = parser.parse_args()
    params = {
        "OPTIMIZER": parse.optimizer,
        "RESIZE": parse.resize, 
        "LEARNING_RATE": parse.learning_rate,
        "MOMENTUM" : parse.momentum,
        "WEIGHT_DECAY": parse.weight_decay, 
        "BATCH_SIZE": parse.batch_size,
        "RANDOM_SEED": parse.random_seed,
        "DEVICE": parse.device,
        "INFERENCE": parse.inference,
        # "EARLYSTOPPING_PATIENCE": parse.earlystopping_patience,
        # "SCHEDULER_PATIENCE": parse.scheduler_patience,
        "SAVE_PATH": parse.save_path, 
        # "SAVE_LOGITS": parse.save_logits,
        "NUM_CLASSES": parse.num_classes,
        "NUM_EPOCHES" : parse.num_epoches,
        "AUGMENTATION" : parse.augmentation
    }