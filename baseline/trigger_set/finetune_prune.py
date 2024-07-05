# finetune & pruning for trigger set 

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

from baseline.trigger_set.queries import StochasticImageQuery
from baseline.trigger_set.trigger_set import save_ckpt
from baseline.trigger_set.verification import loop





# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='baseline/trigger_set/attack/finetune/large_lr', type=str)
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_model_path', default='baseline/trigger_set/initial', type=str)
parser.add_argument('-nq', "--num-query",type=int,help='# of queries',default=10)
parser.add_argument("-tt", "--train-type",type=str,default='base',help='train type, none: no watermark, base: baseline for watermark',choices=['none', 'base', 'margin'])
parser.add_argument('-v', "--variable",type=float,default=0.1)
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
    logging.info(f"Trigger Set Watermark attack: {params.name}, ver: {args.ver}")

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
    model_path = os.path.join(args.attack_model_path, args.ver, 'trigger_set.pt')
    assert os.path.isfile(model_path), "No model file found at {}".format(model_path)
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    resume = os.path.join(args.attack_model_path, args.ver, "trigger_set.pt")
    d = torch.load(resume)
    model.load_state_dict(d['model']['state_dict'])
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')
    # #################### Load Query #####################################
    query = StochasticImageQuery(query_size=(args.num_query, 1, 28, 28), 
                                 response_size=(args.num_query,), query_scale=255, response_scale=10)
    query.load_state_dict(d['query_model']['state_dict'], strict=False)

    

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
    val_meters = loop(model, query, valloader, 
                        train_type=args.train_type, mode='val', device=device, addvar=args.variable)

    logging.info("Best valid nat acc   : {:.4f}".format(val_meters["nat acc"]))
    logging.info("Best valid query acc : {:.4f}".format(val_meters["query acc"]))

    with open("outputs.txt", "a") as file:
        file.write(f"Model {args.ver}\n")
        file.write("Best valid nat acc: {:.4f}\n".format(val_meters["nat acc"]))
        file.write("Best valid query acc: {:.4f}\n".format(val_meters["query acc"]))
    attack_save_name = os.path.join(args.save_path + '/' + args.ver, f'{params.name}.pth')
    torch.save(model.state_dict(), attack_save_name)
    save_ckpt(model, query, attack_save_name)