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
import matplotlib.pyplot as plt




from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage
from src.efficient_kan import KAN

# from data_loader import fetch_dataloader, fetch_dataloader_custom


# ************************** random seed **************************
# seed = 0

# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./baseline/clean_model', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ver1', default='1', type=str)
parser.add_argument('--ver2', default='2', type=str)


args = parser.parse_args()

device_ids = args.gpu_id
# torch.cuda.set_device(device_ids[0])



def validate_model(model1, model2 ,device, args):
    # 1. 读取 key_samples.pth，读取[(key_images.cpu(), key_output.cpu())]
    key_samples_path = f"./baseline/my_way/initial/{args.ver1}/key_samples.pth"
    checkpoint = torch.load(key_samples_path)
    key_images, key_outputs = checkpoint[0]

    # 将 key_images 和 key_outputs 移动到设备
    key_images = key_images.to(device)
    key_outputs = key_outputs.to(device)

    # 2. 将 key_images 输入到模型中，获取对应 output
    model1.eval()
    model2.eval()
    with torch.no_grad():
        model1_outputs = model1.forward_layer_0(key_images)
        model2_outputs = model2.forward_layer_0(key_images)

    # # 3. 检查所有 batch 中均为 0 的神经元
    # zero_neurons = torch.all(model1_outputs == 0, dim=0)


    # # 4. 移除所有 batch 中均为 0 的神经元
    # model_outputs = model_outputs[:, ~zero_neurons]
    # key_outputs = key_outputs[:, ~zero_neurons]

    mse = ((model1_outputs - model2_outputs) ** 2).mean()

    # 打印 MSE
    print(f'MSE between model outputs and key outputs: {1000 * mse.item():.8f}')
    return mse.item() * 1000

if __name__ == "__main__":


    model_path1 = os.path.join(args.save_path, args.ver1, 'clean_model.pth')
    model_path2 = os.path.join(args.save_path, args.ver2, 'clean_model.pth')

    model1, model2 = KAN([28 * 28, 64, 10]), KAN([28 * 28, 64, 10])
    model1.load_state_dict(torch.load(model_path1)['state_dict'])
    model2.load_state_dict(torch.load(model_path2)['state_dict'])

    model1.to("cuda:0")
    model2.to("cuda:0")
    validate_model(model1, model2, "cuda:0", args)