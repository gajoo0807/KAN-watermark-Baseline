"""
Tensorboard logger code referenced from:
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/
Other helper functions:
https://github.com/cs230-stanford/cs230-stanford.github.io
"""

import json
import logging
import os
import shutil
import torch

import numpy as np
import scipy.misc

import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader



import time
from collections import namedtuple
import uuid

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def fetch_mnist_dataloader():

    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    valloader = DataLoader(valset, batch_size=64, shuffle=False)
    return trainloader, valloader





class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MultiAverageMeter(object):
    def __init__(self, names):
        self.AMs = {name: AverageMeter(name) for name in names}

    def __getitem__(self, name):
        return self.AMs[name].avg

    def reset(self):
        for am in self.AMs.values():
            am.reset()

    def update(self, vals, n=1):
        for name, value in vals.items():
            self.AMs[name].update(value, n=n)

    def __str__(self):
        return ' '.join([str(am) for am in self.AMs.values()])


def get_config(config_path):
    with open(config_path, "r") as f :
        config = json.load(f)
    return config

def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple("GeneralDict", obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else :
        return obj