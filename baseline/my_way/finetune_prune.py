# finetune & pruning for USP Process

import torch
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
import argparse
import os
import logging
import numpy as np


import torch.optim as optim
import torch.nn.functional as F




from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage
from src.efficient_kan import KAN
from baseline.my_way.verification import validate_model





# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='baseline/my_way/attack/finetune/large_lr', type=str)
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_model_path', default='baseline/my_way/initial', type=str)
args = parser.parse_args()
device_ids = args.gpu_id

# ************************** random seed **************************
seed = int(args.ver)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    isExist = os.path.exists(os.path.join(args.save_path, args.ver))
    if not isExist:
        os.makedirs(os.path.join(args.save_path, args.ver))
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    params.cuda = torch.cuda.is_available() # use GPU if available
    logging.info(f"two step attack: {params.name}, ver: {args.ver}")

    # ************************** set log **************************
    logger_file = os.path.join(args.save_path + '/' + args.ver, f'{params.name}.log')
    if os.path.exists(logger_file):
        with open(logger_file, 'w') as file:
            file.truncate(0)
        print("The contents of training.log have been deleted.")
    else:
        print(f"The file {logger_file} does not exist.")
    set_logger(logger_file)




    # #################### Load Model #####################################
    model_path = os.path.join(args.attack_model_path, args.ver, 'two_step_model.pth')
    assert os.path.isfile(model_path), "No model file found at {}".format(model_path)
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')

    # #################### Key Sample Path #####################################
    key_samples_path = os.path.join(args.attack_model_path, args.ver, 'key_samples.pth')
    

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()




    if params.prune:
        # ************************** Pruning **************************
        from utils.attack import prune
        logging.info(f'Prune model')
        logging.info(f'Pruning rate: {params.prune_rate}')
        model = prune(model, params.prune_rate, valloader, args, criterion, device)
    if params.finetune:
        # ************************** Finetune **************************
        from utils.attack import finetune
        logging.info(f'Finetune model')
        logging.info(f'Learning rate: {params.learning_rate}')
        finetune(model, optimizer, criterion, trainloader, valloader , args, device)
        

    # ************************** Verification **************************
    val_loss , val_acc = 0, 0

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


    mse_loss = validate_model(model, key_samples_path, device)
    print(f"{model.layers[0].mask=}")
    logging.info("- Test MSE loss : {:05.3f}".format(mse_loss))
    logging.info("- Val Acc : {:05.3f}".format(val_acc))

    with open('my_way.txt', 'a') as file:
        file.write(f"Model {args.ver}\n")
        file.write(f"Test MSE loss: {mse_loss}\n")
        file.write(f"Val Acc: {val_acc}\n")

    attack_save_name = os.path.join(args.save_path + '/' + args.ver, f'{params.name}.pth')
    torch.save({'state_dict': model.state_dict()},attack_save_name)