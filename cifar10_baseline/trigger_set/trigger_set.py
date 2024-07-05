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


from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage, MultiAverageMeter
from src.efficient_kan import KAN
from baseline.trigger_set.queries import StochasticImageQuery
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

import torchvision
from kaconv.convkan import ConvKAN
from kaconv.kaconv import FastKANConvLayer
from torch.nn import Conv2d, BatchNorm2d
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Load CIFAR-10 with data augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./cifar10_baseline/trigger_set/initial', type=str)
parser.add_argument('--resume', default=None, type=str)
# parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--ver', default='1', type=int)
parser.add_argument("-dt", "--dataset",type=str,default='cifar10',choices=['cifar10', 'cifar100', 'svhn', 'mnist'])
parser.add_argument("-tt", "--train-type",type=str,default='margin',help='train type, none: no watermark, base: baseline for watermark',choices=['none', 'base', 'margin'])
parser.add_argument('-nq', "--num-query",type=int,help='# of queries',default=10)
parser.add_argument('-ep', "--epoch",type=int,default=160,required=False)
parser.add_argument('-v', "--variable",type=float,default=0.1)
args = parser.parse_args()

device_ids = args.gpu_id
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
# device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")

# ************************** random seed **************************
seed = int(args.ver)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def split_trainloader(trainloader, num_samples=500):
    # Get the full training dataset from the DataLoader
    trainset = trainloader.dataset
    
    # Split the dataset
    split_lengths = [num_samples, len(trainset) - num_samples]
    subset1, subset2 = random_split(trainset, split_lengths)
    
    return subset1, subset2

# def save_ckpt(model,  query_model, opt, nat_acc, query_acc, epoch, name):
def save_ckpt(model,  query_model, name):
    torch.save({
        "model": {
            "state_dict": model.state_dict()
        },
        "query_model": {
            "state_dict": query_model.state_dict()
        }
    }, name)

# ************************** Training Process **************************
def loop(model, query_model, loader, opt, epoch, logger, output_dir, max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    for batch_idx, batch in enumerate(loader):
        images = batch[0]
        labels = batch[1].long()
        epoch_with_batch = epoch + (batch_idx+1) / len(loader)
        images = images.to(device)
        # images = images.to(device)
        labels = labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()

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
            
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none') # 預測結果與 response 的交叉熵
            loss = torch.cat([nat_loss, query_loss]).mean()
            
        elif train_type == 'margin':
            num_sample_fn = lambda x: np.interp([x], [0, max_epoch], [25, 25])[0]
            num_sample = int(num_sample_fn(epoch))
            if mode == 'train':
                query, response = query_model(discretize=False, num_sample=num_sample)
                query = query.to(device)
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

        if mode == 'train':
            loss.backward()
            opt.step()
        
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        # if batch_idx % 100 == 0 and mode == 'train':
        #     logger.info('=====> {} {}'.format(mode, str(meters)))
        
    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    if mode == 'test' and (epoch+1) % 20 == 0:
        save_image(query.cpu(), os.path.join(output_dir, "images", f"query_image_{epoch}.png"), nrow=query.size(0))
    return meters

def train(args, trainloader , valloader, device):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(args.save_path, str(args.ver), 'output.log')),
                logging.StreamHandler()
                ])

    args.dataset = 'cifar10'
    model = nn.Sequential(
        FastKANConvLayer(3, 32, padding=1, kernel_size=3, stride=1, kan_type="BSpline"),
        BatchNorm2d(32),
        FastKANConvLayer(32, 32, padding=1, kernel_size=3, stride=2, kan_type="BSpline"),
        BatchNorm2d(32),
        FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type="BSpline"),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).cuda()
    model.to(device)

    query = StochasticImageQuery(query_size=(args.num_query, 28, 28), 
                                 response_size=(args.num_query,), query_scale=255, response_scale=10)
    # query_size: query image size, response_size: label size, query_scale: 255, response_scale: 10

    # Split trainloader to get 500 samples
    subset1, _ = split_trainloader(trainloader, num_samples=args.num_query)
    query.initialize(subset1)
        
    query.eval()
    init_query, _ = query()
    query.train()

    save_image(init_query, os.path.join(args.save_path, "images", f"query_image_init.png"), nrow=10)
    model.to(device)
    query.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.8)

    best_val_nat_acc = 0
    best_val_query_acc = 0
    
    # save init #
    # save_ckpt(model, args.model_type, query, args.query_type, opt, None, None, 0, os.path.join(output_dir, "checkpoints", "checkpoint_init.pt"))

    if not os.path.exists(os.path.join(args.save_path, str(args.ver))):
        os.makedirs(os.path.join(args.save_path, str(args.ver)))

    for epoch in range(args.epoch):
        model.train()
        query.train()
        train_meters = loop(model, query, trainloader, opt, epoch, logger, args.save_path,
                        train_type=args.train_type, max_epoch=args.epoch, mode='train', device = device, addvar=args.variable)

        with torch.no_grad():
            model.eval()
            query.eval()
            val_meters = loop(model, query, valloader, opt, epoch, logger, args.save_path,
                            train_type=args.train_type, max_epoch=args.epoch, mode='val', device=device, addvar=args.variable)

            if best_val_nat_acc <= val_meters['nat acc']:
                save_ckpt(model, query, os.path.join(args.save_path, args.ver, "trigger_set.pt"))
                

                
                best_val_nat_acc = val_meters['nat acc']
                best_val_query_acc = val_meters['query acc']

    logger.info("="*100)
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best valid query acc : {:.4f}".format(best_val_query_acc))

    with open("outputs.txt", "a") as file:
        file.write(f"Model {args.ver}\n")
        file.write("Best valid nat acc: {:.4f}\n".format(best_val_nat_acc))
        file.write("Best valid query acc: {:.4f}\n".format(best_val_query_acc))


if __name__ == "__main__":

    isExist = os.path.exists(os.path.join(args.save_path, str(args.ver)))
    if not isExist:
        os.makedirs(os.path.join(args.save_path, str(args.ver)))
    # device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")

    # ************************** set log **************************
    logger_file = os.path.join(args.save_path + '/' + str(args.ver), 'margined_based_watermark.log')
    if os.path.exists(logger_file):
        with open(logger_file, 'w') as file:
            file.truncate(0)
        print("The contents of training.log have been deleted.")
    else:
        print(f"The file {logger_file} does not exist.")
    set_logger(logger_file)

    logging.info(f"Train Margined Based watermarking Network, ver: {args.ver}")
    # #################### Load the parameters from json file #####################################
    # json_path = os.path.join(args.save_path, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = Params(json_path)

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    valset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_val
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)


    
    train(args,  trainloader , valloader, device)

