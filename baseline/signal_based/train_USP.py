# train a nasty teacher with an adversarial network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from tqdm import tqdm
import argparse
import os
import logging
import numpy as np

from utils.utils import set_logger, fetch_mnist_dataloader, Params, RunningAverage
from src.efficient_kan import KAN


import random

# ************************** detector framework **************************

class DetectorMLP5(nn.Module):
    def __init__(self, input_size=10, num_class=2):
        super(DetectorMLP5, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> hidden_3)
        self.fc3 = nn.Linear(512, 512)
        # linear layer (n_hidden -> hidden_4)
        self.fc4 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc5 = nn.Linear(512, num_class)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, self.input_size)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x


# ************************** detect dataset generating **************************
def detect_dataset(wm, ad):
    '''
    組合clean model & watermark model的output，生成對應dataset

    Args:
        wm: watermark model output
        ad: adversarial model output

    Returns:
        x: 組合後的dataset
        y: 組合後的label
    '''
    x = torch.cat((wm, ad))
    x = x.cpu().numpy()  # 将x转换为numpy.ndarray
    y_wm = np.array([[0, 1]] * wm.size(0))
    y_ad = np.array([[1, 0]] * ad.size(0))
    y = np.concatenate([y_wm, y_ad])
    z = list(zip(x, y))
    random.shuffle(z)
    x, y = zip(*z)
    x = torch.tensor(np.array(x))  # 将列表转换为numpy.ndarray后再转换为tensor
    y = torch.tensor(np.array(y))  # 将列表转换为numpy.ndarray后再转换为tensor
    return x, y

# ************************** train detector **************************
def train_detector(detector, wm_x, wm_y, optim):
    '''
    train detector

    Args:
        detector: detector model
        wm_x: watermark model output
        wm_y: watermark model label
        optim: optimizer

    Returns:
        d_loss: detector loss
        acc: detector accuracy
    '''
    detector.train()
    loss_F = nn.BCELoss()
    m = nn.Sigmoid()
    acc = 0
    for i in range(5):
        detector_output = detector(wm_x)
        d_loss = loss_F(m(detector_output.float()), wm_y.float())
        
        optim.zero_grad()
        d_loss.backward()
        optim.step()

        output_batch = detector_output.detach().cpu().numpy()
        output_batch = np.argmax(output_batch, axis=1)
        labels_batch = wm_y.detach().cpu().numpy()
        labels_batch = np.argmax(labels_batch, axis=1)
        acc = 100.0 * np.sum(output_batch == labels_batch) / float(labels_batch.shape[0])
        if acc>99: break

    return d_loss.detach(), acc

def train_epoch_kd_adv(model, model_ad, detector, optim, dt_optim, data_loader, epoch, params):
    model.train()
    model_ad.eval()
    tch_loss_avg = RunningAverage()
    ad_loss_avg = RunningAverage()
    loss_avg = RunningAverage()

    detect_acc = 0
    wm_loss = 0
    d_acc = []
    d_loss =[]

    with tqdm(total=len(data_loader)) as t:  # Use tqdm for progress bar
        for i, (train_batch, labels_batch) in enumerate(data_loader):
            train_batch, labels_batch = train_batch.view(-1, 28 * 28).cuda(), labels_batch.cuda()
            output_tch = model(train_batch)

            # teacher loss: CE(output_tch, label)
            tch_loss = nn.CrossEntropyLoss()(output_tch, labels_batch)

            # ############ adversarial loss ####################################
            # computer adversarial model output
            with torch.no_grad():
                output_stu = model_ad(train_batch)  # logit without SoftMax
            output_stu = output_stu.detach()

            wm_x, wm_y = detect_dataset(output_tch.detach(), output_stu)
            wm_x, wm_y = wm_x.cuda(), wm_y.cuda()
            wm_loss, detect_acc = train_detector(detector, wm_x, wm_y, dt_optim)


            # adversarial loss: KLdiv(output_stu, output_tch)
            T = params.temperature
            adv_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_stu / T, dim=1),
                                      F.softmax(output_tch / T, dim=1)) * (T * T)   # wish to max this item
            
            loss = tch_loss - adv_loss * 0.01 + wm_loss + 2  # make the loss positive by adding a constant
            # tch_loss : main task loss
            # adv_loss : KLD Loss
            # wm_loss：detector loss

            
            # ############################################################
            d_acc.append(detect_acc)
            d_loss.append(wm_loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the average loss
            loss_avg.update(loss.item())
            tch_loss_avg.update(tch_loss.item())
            ad_loss_avg.update(adv_loss.item())

            # tqdm setting
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

        return loss_avg(), tch_loss_avg(), ad_loss_avg(), sum(d_acc)/len(d_acc), sum(d_loss)/len(d_loss)

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

            with torch.no_grad():
                output_stu = model_ad(data_batch)  # logit without SoftMax
            output_stu = output_stu.detach()

            # detector evaluate
            wm_x, wm_y = detect_dataset(output_batch.detach(), output_stu)
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


def train_and_eval_kd_adv(model, model_ad, detector, optim, dt_optim, train_loader, dev_loader, params):
    best_val_acc = -1
    best_epo = -1
    lr = params.learning_rate
    dt_lr = params.dt_learning_rate

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        # Train
        model.train()
        adversarial_model.eval()
        detector.train()


        # ********************* one full pass over the training set *********************
        train_loss, train_tloss, train_aloss, detect_acc, wm_loss = train_epoch_kd_adv(model, model_ad, detector, optim, dt_optim,
                                                            train_loader, epoch, params)
        logging.info("- Train loss : {:05.3f}".format(train_loss))
        logging.info("- Train teacher loss : {:05.3f}".format(train_tloss))
        logging.info("- Train adversarial loss : {:05.3f}".format(train_aloss))
        logging.info("- Train WM loss : {:05.3f}".format(wm_loss))
        logging.info("- Train Detect acc : {:05.3f}".format(detect_acc))

        # ************************** evaluate on the validation set **********************
        val_metrics, detect_acc, wm_loss  = evaluate(model, model_ad, detector, nn.CrossEntropyLoss(), dev_loader, params)  # {'acc':acc, 'loss':loss}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in val_metrics.items())
        logging.info("- Eval metrics : " + metrics_string)
        logging.info("- Test WM loss : {:05.3f}".format(wm_loss))
        logging.info("- Test Detect acc : {:05.3f}".format(detect_acc))


       # ********************* get the best validation accuracy *********************
        val_acc = val_metrics['acc']
        if val_acc >= best_val_acc:
            best_epo = epoch + 1
            best_val_acc = val_acc
            logging.info('- New best model ')
            # save best model
            save_name = os.path.join(args.save_path + '/' + args.ver, 'USP_model.pth')
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best acc: {:05.3f}'.format(best_epo, best_val_acc))
        dt_save_name = os.path.join(args.save_path + '/' + args.ver, 'detector.pt')
        torch.save(detector.state_dict(), dt_save_name)
        

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


# ************************** random seed **************************
seed = 0

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ************************** parameters **************************
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='baseline/signal_based/USP', type=str)
parser.add_argument('--ver', default='1', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

if __name__ == "__main__":
    isExist = os.path.exists(os.path.join(args.save_path, args.ver))
    if not isExist:
        os.makedirs(os.path.join(args.save_path, args.ver))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ************************** set log **************************
    logger_file = os.path.join(args.save_path + '/' + args.ver, 'ft_training.log')
    if os.path.exists(logger_file):
        with open(logger_file, 'w') as file:
            file.truncate(0)
        print("The contents of training.log have been deleted.")
    else:
        print(f"The file {logger_file} does not exist.")
    set_logger(logger_file)
    logging.info(f"Train USP adversarial Network, ver: {args.ver}")

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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    # Define loss
    criterion = nn.CrossEntropyLoss()

    # ************************** adversarial model **************************
    adversarial_model = KAN(KAN_arch)
    checkpoint = torch.load("baseline/signal_based/scratch/clean_model.pth")
    adversarial_model.load_state_dict(checkpoint['state_dict'])
    adversarial_model.to(device)


    # ************************** Detector **************************
    detector = DetectorMLP5(input_size=10)
    detector.to(device)
    dt_optim = optim.Adam(detector.parameters(), lr=params.dt_learning_rate)

    train_and_eval_kd_adv(model, adversarial_model, detector, optimizer, dt_optim, trainloader, valloader, params)
