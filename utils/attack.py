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



def finetune(model, optimizer, criterion, trainloader, valloader, args):
    '''
    Fine-tune the model

    Args:
        model: the model to be fine-tuned
        criterion: the loss function
        optimizer: the optimizer
        scheduler: the learning rate scheduler
        dataloaders: a dictionary containing the training and validation dataloaders
        num_epochs: the number of epochs to train the model

    Returns:
        model: the fine-tuned model
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            save_name = os.path.join(args.save_path, args.ver , 'finetune.pth')
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                save_name)

        logging.info('- So far best epoch: {}, best val acc: {:05.3f}'.format(best_epo, best_val_acc))

        print(
            f"Epoch {epoch + 1}, Train Loss: {loss:.5f}, Train Acc: {accuracy:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_acc:.5f}"
        )
    




def prune(model, threshold, valloader, args):
    '''
    Prune the model, mask neurons with small weights
    

    Args:
        model: the model to be pruned
        threshold: the threshold to prune the model, 0.2 -> 20% of the neurons will be pruned

    Returns:
        model2: the pruned model
    '''
    # model forward
    # calculate the importance of each neuron
    # prune the model

    model.eval()
    with torch.no_grad():
        images, labels = next(iter(valloader))
        images = images.view(-1, 28 * 28).to("cuda")
        output = model(images)

    # 根據threshold對model進行prune, 更新model.mask
    for layer in model.layers:
        # Flatten the important scores and get the threshold value
        important_score = layer.important_score
        
        flattened_scores = important_score.view(-1)
        num_elements = flattened_scores.numel()
        k = int(num_elements * threshold)
        
        # Get the k smallest scores
        threshold_value, _ = torch.kthvalue(flattened_scores, k)
        
        # Update the mask: set elements with importance score below the threshold to 0
        layer.mask[important_score < threshold_value] = 0

    return model

