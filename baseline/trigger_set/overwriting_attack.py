# overwriting for trigger set 

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




from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage, MultiAverageMeter
from src.efficient_kan import KAN
from baseline.my_way.verification import validate_model

from baseline.trigger_set.queries import StochasticImageQuery
from baseline.trigger_set.trigger_set import save_ckpt
from baseline.trigger_set.trigger_set import loop, split_trainloader

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='baseline/trigger_set/attack/overwriting', type=str)
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_model_path', default='baseline/trigger_set', type=str)
parser.add_argument('-nq', "--num-query",type=int,help='# of queries',default=10)
parser.add_argument("-tt", "--train-type",type=str,default='base',help='train type, none: no watermark, base: baseline for watermark',choices=['none', 'base', 'margin'])
parser.add_argument('-v', "--variable",type=float,default=0.1)
parser.add_argument('-ep', "--epoch",type=int,default=10,required=False)
args = parser.parse_args()
device_ids = args.gpu_id

# ************************** random seed **************************
seed = int(args.ver)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def loop_eval(model, query_model, loader, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        labels = batch[1].long()
        images = images.view(-1, 28 * 28).to(device)
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        if train_type == 'none':
            with torch.no_grad():
                model.eval()
                query, response = query_model()
                query_preds = model(query)
                query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
                query_loss = F.cross_entropy(query_preds, response)
                if mode == 'train':
                    model.train()

            loss = nat_loss.mean()
            
        elif train_type == 'base':
            # 先讓自己用base方式train
            query, response = query_model() # 產出 query 和 response -> query 是圖片, response 是 label
            query = query.view(-1, 28 * 28).to(device)
            query_preds = model(query)
            response = response.to(device)


            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none') # 預測結果與 response 的交叉熵
            loss = torch.cat([nat_loss, query_loss]).mean()
            
        elif train_type == 'margin':
            num_sample = 25
            if mode == 'train':
                query, response = query_model(discretize=False, num_sample=num_sample)
                for _ in range(5):
                    query = query.detach()
                    query.requires_grad_(True)
                    query_preds = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
            else:
                query, response = query_model(discretize=(mode!='train'))
            query_preds = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = addvar * F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()

        
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        # if batch_idx % 100 == 0 and mode == 'train':
        #     logger.info('=====> {} {}'.format(mode, str(meters)))
        
    return meters


def overwriting_attack(args, trainloader , valloader, device, model):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(args.save_path, 'output.log')),
                logging.StreamHandler()
                ])
    args.dataset = 'mnist'
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')

    query = StochasticImageQuery(query_size=(args.num_query, 28, 28), 
                                 response_size=(args.num_query,), query_scale=255, response_scale=10)
    # query_size: query image size, response_size: label size, query_scale: 255, response_scale: 10

    # Split trainloader to get 500 samples
    subset1, _ = split_trainloader(trainloader, num_samples=args.num_query)
    query.initialize(subset1)
        
    query.eval()
    init_query, _ = query()
    query.train()
    query.to(device)
    model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)

    if not os.path.exists(os.path.join(args.save_path, args.ver)):
        os.makedirs(os.path.join(args.save_path, args.ver))

    best_val_nat_acc, best_val_query_acc = 0, 0

    for epoch in range(10):
        model.train()
        query.train()
        train_meters = loop(model, query, trainloader, opt, epoch, logger, args.save_path,
                        train_type=args.train_type, max_epoch=10, mode='train', device = device, addvar=args.variable)

        with torch.no_grad():
            model.eval()
            query.eval()
            val_meters = loop(model, query, valloader, opt, 10, logger, args.save_path,
                            train_type=args.train_type, max_epoch=args.epoch, mode='val', device=device, addvar=args.variable)

            if best_val_nat_acc <= val_meters['nat acc']:
                save_ckpt(model, query, os.path.join(args.save_path, args.ver, "trigger_set.pt"))
                

                
                best_val_nat_acc = val_meters['nat acc']
                best_val_query_acc = val_meters['query acc']

    logger.info("="*100)
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best valid query acc : {:.4f}".format(best_val_query_acc))





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
    original = torch.load(model_path)
    model.load_state_dict(original['model']['state_dict'])
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')




    # #################### Load Query #####################################
    overwrite_num = np.random.choice([i for i in range(1, 31) if i != int(args.ver)])

    overwrite_query_path = os.path.join(args.attack_model_path, args.ver, "trigger_set.pt")
    d = torch.load(overwrite_query_path)
    overwrite_query = StochasticImageQuery(query_size=(args.num_query, 1, 28, 28), 
                                 response_size=(args.num_query,), query_scale=255, response_scale=10)
    overwrite_query.load_state_dict(d['query_model']['state_dict'], strict=False)

    original_query = StochasticImageQuery(query_size=(args.num_query, 1, 28, 28), 
                                 response_size=(args.num_query,), query_scale=255, response_scale=10)
    original_query.load_state_dict(original['query_model']['state_dict'], strict=False)

    overwriting_attack(args, trainloader , valloader, device, model)

    val_meters = loop_eval(model, original_query, valloader, 
                        train_type=args.train_type, mode='val', device=device, addvar=args.variable)
    
    logging.info("Best valid nat acc   : {:.4f}".format(val_meters["nat acc"]))
    logging.info("Best valid query acc : {:.4f}".format(val_meters["query acc"]))
    with open("outputs.txt", "a") as file:
        file.write(f"Model {args.ver}\n")
        file.write("Best valid nat acc: {:.4f}\n".format(val_meters["nat acc"]))
        file.write("Best valid query acc: {:.4f}\n".format(val_meters["query acc"]))
