from src.efficient_kan import KAN

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from kaconv.convkan import ConvKAN
from kaconv.kaconv import FastKANConvLayer
from torch.nn import Conv2d, BatchNorm2d

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

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
valset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_val
)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define model
# model = KAN([32 * 32 * 3, 512,  512,  128, 64, 32, 10])
model = nn.Sequential(
    FastKANConvLayer(3, 32, padding=1, kernel_size=3, stride=1, kan_type="BSpline"),
    BatchNorm2d(32),
    FastKANConvLayer(32, 32, padding=1, kernel_size=3, stride=2, kan_type="BSpline"),
    BatchNorm2d(32),
    FastKANConvLayer(32, 10, padding=1, kernel_size=3, stride=2, kan_type="BSpline"),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Define loss
criterion = nn.CrossEntropyLoss()

for epoch in range(160):
    # Train
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            # images = images.view(-1, 32 * 32 * 3).to(device)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels).float().mean()
            train_loss += loss.item()
            train_accuracy += accuracy.item()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    train_loss /= len(trainloader)
    train_accuracy /= len(trainloader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            # images = images.view(-1, 32 * 32 * 3).to(device)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            val_loss += criterion(output, labels).item()
            val_accuracy += (output.argmax(dim=1) == labels).float().mean().item()
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1}, Train Loss: {train_loss:.5f}, Train Acc: {train_accuracy:.5f}, Val Loss: {val_loss:.5f}, Val Accuracy: {val_accuracy:.5f}"
    )

