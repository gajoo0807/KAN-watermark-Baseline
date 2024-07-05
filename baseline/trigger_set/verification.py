# two step training process

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

# from data_loader import fetch_dataloader, fetch_dataloader_custom
from baseline.trigger_set.queries import StochasticImageQuery


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./baseline/trigger_set', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--clean_model', action='store_true', help='If set, use the clean model')
parser.add_argument('-nq', "--num-query",type=int,help='# of queries',default=10)
parser.add_argument("-tt", "--train-type",type=str,default='base',help='train type, none: no watermark, base: baseline for watermark',choices=['none', 'base', 'margin'])
parser.add_argument('-v', "--variable",type=float,default=0.1)


args = parser.parse_args()
device_ids = args.gpu_id
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device_ids[0])


# def verification()

def loop(model, query_model, loader, train_type='standard', mode='train', device='cuda', addvar=None):
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

if __name__ == "__main__":
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    if args.clean_model:
        model_path = os.path.join("baseline/clean_model", args.ver, 'clean_model.pth')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Verify key {args.ver} on clean model")
    else:
        model_path = os.path.join(args.save_path, args.ver, 'trigger_set.pt')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model']['state_dict'])
        print(f"Verify key {args.ver} on two step model")
    model.to(device)

    trainloader, valloader = fetch_mnist_dataloader()
    query_path = os.path.join(args.save_path, args.ver, 'trigger_set.pt')
    checkpoint = torch.load(query_path)
    query = StochasticImageQuery(query_size=(args.num_query,1,  28, 28), 
                        response_size=(args.num_query,), query_scale=255, response_scale=10)
    
    query.load_state_dict(checkpoint['query_model']['state_dict'], strict=False)

    print("Model loaded successfully!")

    val_meters = loop(model, query, valloader, 
                        train_type=args.train_type, mode='val', device=device, addvar=args.variable)
    print(f"Best valid nat acc   : {val_meters['nat acc']:.4f}")
    print(f"Best valid query acc : {val_meters['query acc']:.4f}")
