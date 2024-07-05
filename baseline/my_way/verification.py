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
parser.add_argument('--save_path', default='./baseline/my_way/initial', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--clean_model', action='store_true', help='If set, use the clean model')
parser.add_argument('--finetune', action='store_true', help='If set, use the clean model')
args = parser.parse_args()

device_ids = args.gpu_id
# torch.cuda.set_device(device_ids[0])


# def verification()

def vis(k):
    # 计算平均值和标准差
    mean_k = np.mean(k)
    std_k = np.std(k)
    # 打印平均值和标准差
    print(f"Mean of k: {mean_k}")
    print(f"Standard deviation of k: {std_k}")

    # 可视化 k 的范围
    plt.figure(figsize=(10, 6))
    plt.hist(k, bins=100, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Histogram of Absolute Differences')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 在图表上显示平均值和标准差
    plt.axvline(mean_k, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean_k + std_k, color='green', linestyle='dashed', linewidth=1)
    plt.axvline(mean_k - std_k, color='green', linestyle='dashed', linewidth=1)
    plt.text(mean_k, plt.ylim()[1] * 0.9, f'Mean: {mean_k:.2f}', color='red')
    # plt.text(mean_k + std_k, plt.ylim()[1] * 0.8, f'Std Dev: {std_k:.2f}', color='green')
    plt.text(mean_k - std_k, plt.ylim()[1] * 0.8, f'Std Dev: {std_k:.2f}', color='green')

    # 保存图像
    plt.savefig('diff.png')

    # plt.show()

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



    # k = model_outputs - key_outputs
    # k = k.abs().cpu().numpy()
    # k = k.flatten()
    # vis(k)
    # with open("diff.txt", "w") as f:
    #     f.write(str())
    #     f.write("\n\n")
    #     f.write(str(model_outputs.cpu().numpy()))
    #     f.write("\n\n")
    #     f.write(str(key_outputs.cpu().numpy()))


    # 3. 将 output 与 key_outputs 做 MSE，计算输出差异
    # mse = F.mse_loss(model_outputs, key_outputs)
    mse = ((model_outputs - key_outputs) ** 2).mean()

    # 打印 MSE
    print(f'MSE between model outputs and key outputs: {1000 * mse.item():.8f}')
    return mse.item() * 1000

if __name__ == "__main__":
    if args.clean_model:
        # model_path = os.path.join(args.save_path, 'clean_model.pth')
        random_num = np.random.choice([i for i in range(1, 31) if i != int(args.ver)])
        model_path = os.path.join("baseline/clean_model", str(random_num) , 'clean_model.pth')
        print(f"Verify key {args.ver} on clean model {random_num}")
    elif args.finetune:
        model_path = os.path.join("baseline/my_way/attack/finetune/large_lr", args.ver, 'ft_large_lr.pth')
        print(f"Verify key {args.ver} on finetune model")
    else:
        model_path = os.path.join(args.save_path, args.ver, 'two_step_model.pth')
        print(f"Verify key {args.ver} on two step model")
    print(f"{model_path=}")
    
    key_samples_path = os.path.join(args.save_path, args.ver, 'key_samples.pth')
    checkpoint = torch.load(model_path)
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    model.load_state_dict(checkpoint['state_dict'])
    model.to("cuda:0")
    print("Model loaded successfully!")
    validate_model(model, key_samples_path, "cuda:0")
