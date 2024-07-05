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

from baseline.my_way.verification import validate_model
# from data_loader import fetch_dataloader, fetch_dataloader_custom

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./baseline/my_way', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=0, type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--ver', default='1', type=str)
args = parser.parse_args()

device_ids = args.gpu_id


# ************************** random seed **************************
seed = int(args.ver)

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

    # 3. 将 output 与 key_outputs 做 MSE，计算输出差异
    mse = F.mse_loss(model_outputs, key_outputs)

    # 打印 MSE
    print(f'MSE between model outputs and key outputs: {mse.item()}')

    logging.info(f'MSE between model outputs and key outputs: {mse.item()}')

def compute_test(key_image):
    # 检查 key_image 中是否包含非 -1 的数值
    contains_non_minus_one = (key_image != -1).any()

    # 计算非 -1 数值的个数
    non_minus_one_count = (key_image != -1).sum().item()

    # 计算 -1 数值的个数
    minus_one_count = (key_image == -1).sum().item()

    # 计算非 -1 数值与 -1 数值的比例
    if minus_one_count == 0:
        ratio = float('inf')  # 避免除以 0
    else:
        ratio = non_minus_one_count / minus_one_count

    print(f"Contains non -1 values: {contains_non_minus_one}")
    print(f"Non -1 values count: {non_minus_one_count}")
    print(f"-1 values count: {minus_one_count}")
    print(f"Ratio of non -1 values to -1 values: {ratio}")


def train_and_eval(model, optimizer, criterion, trainloader, valloader, params, optimizer_layer_0):
    embedding_key = params.embedding_key
    amplitude, frequency, phase, watermark_func = embedding_key['amplitude'], embedding_key['frequency'], embedding_key['phase'], torch.cos if embedding_key['watermark_func'] == "cos" else torch.sin

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

                # Two Step Training
                # output = model.forward_layer_0(images)
                # embedding_signal = amplitude * watermark_func(frequency * output + phase)
                # loss_signal = torch.mean(embedding_signal ** 2)
                # print(f"{output.shape=}")
                # print(f"{embedding_signal.shape=}")

                # optimizer_layer_0.zero_grad()
                # loss_signal.backward()
                # optimizer_layer_0.step()

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
            save_name = os.path.join(args.save_path, 'initial', args.ver , 'two_step_model.pth')
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                save_name)
            with torch.no_grad():
                key_images, _ = next(iter(trainloader))
                key_images = key_images.view(-1, 28 * 28).to(device)
                key_output = model.forward_layer_0(key_images)
                print(f"{key_images.shape=}, {key_output.shape=}")
                key_samples = [(key_images.cpu(), key_output.cpu())]
            # Save key samples
            save_name = os.path.join(args.save_path,'initial',  args.ver, 'key_samples.pth')
            torch.save(key_samples, save_name)
            logging.info(f'Key samples saved to {save_name}')
            


        # Update learning rate
        scheduler.step()

        logging.info('- So far best epoch: {}, best val acc: {:05.3f}'.format(best_epo, best_val_acc))

        print(
            f"Epoch {epoch + 1}, Train Loss: {loss:.5f}, Train Acc: {accuracy:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_acc:.5f}"
        )
    
    # 儲存驗證key sample
    model.eval()


    # mse_loss = validate_model(model, save_name, device)
    # logging.info(f"Test MSE loss: {mse_loss}")


if __name__ == "__main__":
    isExist = os.path.exists(os.path.join(args.save_path, 'initial', args.ver))
    if not isExist:
        os.makedirs(os.path.join(args.save_path, 'initial', args.ver))
    
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    # ************************** set log **************************
    logger_file = os.path.join(args.save_path + '/initial/' + args.ver, 'two_step_training.log')
    if os.path.exists(logger_file):
        with open(logger_file, 'w') as file:
            file.truncate(0)
        print("The contents of training.log have been deleted.")
    else:
        print(f"The file {logger_file} does not exist.")
    set_logger(logger_file)

    logging.info(f"Train Two Step Signal Network, ver: {args.ver}")
    # #################### Load the parameters from json file #####################################
    json_path = os.path.join(args.save_path, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    params.cuda = torch.cuda.is_available() # use GPU if available


    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')


    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()
    # Define optimizer for layer 0
    optimizer_layer_0 = optim.AdamW(model.layers[0].parameters(), lr=1e-3, weight_decay=1e-4)


    train_and_eval(model, optimizer, criterion, trainloader, valloader, params, optimizer_layer_0)