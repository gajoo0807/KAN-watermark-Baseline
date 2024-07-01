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




from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage
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
parser.add_argument('--save_path', default='./baseline/my_way', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--clean_model', action='store_true', help='If set, use the clean model')

args = parser.parse_args()

device_ids = args.gpu_id
# torch.cuda.set_device(device_ids[0])


# def verification()

def validate_model(model, key_samples_path, device):
    # 1. 读取 key_samples.pth，读取[(key_images.cpu(), key_output.cpu())]
    checkpoint = torch.load(key_samples_path)
    key_images, key_outputs = checkpoint[0]

    # 将 key_images 和 key_outputs 移动到设备
    key_images = key_images.to(device)
    key_outputs = key_outputs.to(device)

    # 2. 将 key_images 输入到模型中，获取对应 output
    model.eval()
    with torch.no_grad():
        model_outputs = model.forward_layer_0(key_images)

    # 3. 检查所有 batch 中均为 0 的神经元
    zero_neurons = torch.all(model_outputs == 0, dim=0)

    # 4. 移除所有 batch 中均为 0 的神经元
    model_outputs = model_outputs[:, ~zero_neurons]
    key_outputs = key_outputs[:, ~zero_neurons]

    # 3. 将 output 与 key_outputs 做 MSE，计算输出差异
    mse = F.mse_loss(model_outputs, key_outputs)

    # 打印 MSE
    print(f'MSE between model outputs and key outputs: {mse.item()/model_outputs.size(1)}')
    return mse.item()/model_outputs.size(1)
if __name__ == "__main__":
    if args.clean_model:
        model_path = os.path.join(args.save_path, 'clean_model.pth')
        print(f"Verify key {args.ver} on clean model")
    else:
        model_path = os.path.join(args.save_path, args.ver, 'two_step_model.pth')
        print(f"Verify key {args.ver} on two step model")
    key_samples_path = os.path.join(args.save_path, args.ver, 'key_samples.pth')
    checkpoint = torch.load(model_path)
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda:0")
    print("Model loaded successfully!")
    validate_model(model, key_samples_path, "cuda:0")
