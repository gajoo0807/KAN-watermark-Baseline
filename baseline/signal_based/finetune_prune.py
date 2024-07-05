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

from baseline.signal_based.train_USP import DetectorMLP5
import random

# ************************** detect dataset generating **************************
def detect_dataset(wm):
    '''
    生成clean model & watermark model的output，对应的dataset

    Args:
        wm: watermark model output

    Returns:
        x: 生成的dataset
        y: 生成的label
    '''
    x = wm.cpu().numpy()  # 将x转换为numpy.ndarray
    y = np.array([[0, 1]] * wm.size(0))  # 生成对应的label
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    x = torch.tensor(np.array(x))  # 将列表转换为numpy.ndarray后再转换为tensor
    y = torch.tensor(np.array(y))  # 将列表转换为numpy.ndarray后再转换为tensor
    return x, y

def detect_accuracy(detector, wm_x, wm_y):
    detector.eval()
    loss_F = nn.BCELoss()
    m = nn.Sigmoid()
    detector_output = detector(wm_x)
    d_loss = loss_F(m(detector_output.float()), wm_y.float())
    output_batch = detector_output.detach().cpu().numpy()
    output_batch = np.argmax(output_batch, axis=1)
    labels_batch = wm_y.detach().cpu().numpy()
    labels_batch = np.argmax(labels_batch, axis=1)
    acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])
    return d_loss, acc

# ************************** detector acc after attack **************************
# 經過攻擊的evaluation ->只衡量是否偵測得到watermark(true positive)
def evaluate(model, model_ad, detector,  loss_fn, data_loader, params):
    model.eval()
    model_ad.eval()
    # summary for current eval loop
    summ = []
    wm_acc_list = []
    wm_loss_list = []

    with torch.no_grad():
        # compute metrics over the dataset
        for data_batch, labels_batch in data_loader:
            if params.cuda:
                data_batch = data_batch.view(-1, 28 * 28).cuda()          # (B,3,32,32)
                labels_batch = labels_batch.cuda()      # (B,)

            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch)


            # detector evaluate
            wm_x, wm_y = detect_dataset(output_batch.detach())
            
            wm_x = wm_x.cuda()
            wm_y = wm_y.cuda()
            wm_loss, acc = detect_accuracy(detector, wm_x, wm_y)
            wm_acc_list.append(acc)
            wm_loss_list.append(wm_loss)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.cpu().numpy()
            labels_batch = labels_batch.cpu().numpy()
            # calculate accuracy
            output_batch = np.argmax(output_batch, axis=1)
            acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])

            summary_batch = {'acc': acc, 'loss': loss.item()}
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    return metrics_mean, sum(wm_acc_list)/len(wm_acc_list), sum(wm_loss_list)/len(wm_loss_list)


# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='baseline/signal_based/attack/finetune/large_lr', type=str)
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--attack_model_path', default='baseline/signal_based/USP', type=str)
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
    logging.info(f"USP attack: {params.name}, ver: {args.ver}")

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
    model_path = os.path.join(args.attack_model_path, args.ver, 'USP_model.pth')
    assert os.path.isfile(model_path), "No model file found at {}".format(model_path)
    trainloader, valloader = fetch_mnist_dataloader()
    KAN_arch = [28 * 28, 64, 10]
    model = KAN(KAN_arch)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    logging.info(f'Model architecture {KAN_arch}')
    

    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()

    # ************************** Load Detector **************************
    detector_path = os.path.join(args.attack_model_path, args.ver, 'detector.pt')
    assert os.path.isfile(detector_path), "No model file found at {}".format(detector_path)
    detector = DetectorMLP5(input_size=10)
    checkpoint = torch.load(detector_path)
    detector.load_state_dict(checkpoint)
    detector.to(device)
    dt_optim = optim.Adam(detector.parameters(), lr=params.dt_learning_rate)

    # ************************** adversarial model **************************
    adversarial_model = KAN(KAN_arch)
    checkpoint = torch.load("baseline/signal_based/scratch/clean_model.pth")
    adversarial_model.load_state_dict(checkpoint['state_dict'])
    adversarial_model.to(device)




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
    val_metrics, detect_acc, wm_loss  = evaluate(model, adversarial_model, detector, nn.CrossEntropyLoss(), valloader, params)  # {'acc':acc, 'loss':loss}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
    # logging.info("- Eval metrics : " + metrics_string)
    logging.info("- Main Task Acc :  {:05.3f}".format(val_metrics['acc']))
    logging.info("- Test Detect acc : {:05.3f}".format(detect_acc))

    attack_save_name = os.path.join(args.save_path + '/' + args.ver, f'{params.name}.pth')
    torch.save(model.state_dict(), attack_save_name)
    # model.verification()





    # train_and_eval_kd_adv(model, adversarial_model, detector, optimizer, dt_optim, trainloader, valloader, params)