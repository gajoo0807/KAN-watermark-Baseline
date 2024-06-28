# train a baseline model from scratch

import torch
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np


import torch.optim as optim




from utils.utils import set_logger, fetch_mnist_dataloader
from src.efficient_kan import KAN

# from data_loader import fetch_dataloader, fetch_dataloader_custom


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./baseline/signal_based/scratch', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

device_ids = args.gpu_id
# torch.cuda.set_device(device_ids[0])



def train_and_eval(model, optimizer, criterion, trainloader, valloader):
    best_val_acc = -1
    best_epo = -1

    for epoch in range(10):
        # Train
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])
        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_acc += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_acc /= len(valloader)

        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path, 'clean_model.pth')
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                save_name)

        # Update learning rate
        scheduler.step()

        logging.info('- So far best epoch: {}, best val acc: {:05.3f}'.format(best_epo, best_val_acc))

        print(
            f"Epoch {epoch + 1}, Train Loss: {loss:.5f}, Train Acc: {accuracy:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_acc:.5f}"
        )



if __name__ == "__main__":
    logger_file = os.path.join(args.save_path, 'training.log')
    if os.path.exists(logger_file):
        with open(logger_file, 'w') as file:
            file.truncate(0)
        print("The contents of training.log have been deleted.")
    else:
        print(f"The file {logger_file} does not exist.")
    set_logger(logger_file)

    logging.info('Training from scratch')
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')


    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()

    train_and_eval(model, optimizer, criterion, trainloader, valloader)


