from tqdm import tqdm
import torch
import os


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def train_fn(num_epochs, train_data_loader, optimizer, model, device, wandb, args, logger):
    
    
    best_loss = 1000
    loss_hist = Averager()

    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        
        wandb.log({"loss": loss_hist.value})
        logger.info(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = f'./checkpoints/torchvision_{args["OPTIMIZER"]}_{args["AUGMENTATION"]}.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value