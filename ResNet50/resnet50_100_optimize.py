from torch.cuda.amp import autocast
import pandas as pd
import os
import torch
import time
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import *


batch_size = 128
epochs = 30
max_learning_rate = 0.001
grad_clip = 0.01
weight_decay = 0.001
optimizer_function = torch.optim.Adam
mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]  
std = [0.26733428587941854, 0.25643846292120615, 0.2761504713263903]  

transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(mean, std, inplace=True)
])

transform_test = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(mean, std)
])

train_dataset = torchvision.datasets.CIFAR100(
    "./", train=True, download=True, transform=transform_train
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
)

test_dataset = torchvision.datasets.CIFAR100(
    "./", train=False, download=True, transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size * 2, pin_memory=True, num_workers=2
)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


device = get_default_device()

train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct_count = torch.sum(preds == labels).item()
    acc = correct_count / len(preds)
    acc = torch.tensor(acc)
    f1 = f1_score(preds.cpu(), labels.cpu(), average='micro')
    return f1, acc


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        f1, acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc, 'f1': f1}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        batch_f1s = [x['f1'] for x in outputs]
        epoch_f1 = np.mean(batch_f1s)  # Combine f1s
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}, f1: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc'], result['f1']))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            conv3x3(in_planes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class ResNet50(ImageClassificationBase):

    def __init__(self, block, nblocks, num_classes=100):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.pre_layers = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self._make_layer(block, 64, nblocks[0])
        self.layer2 = self._make_layer(block, 128, nblocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, nblocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, nblocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, nblocks, stride=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = to_device(ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=100), device)


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in test_loader]
    return model.validation_epoch_end(outputs)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def log_metrics(string, log_file):
    file_log = open(log_file, "a")
    file_log.write(string + "\n")
    file_log.close()


accumulation_steps = 2


def fit_one_cycle_with_perf(epochs, max_lr, model, train_loader, test_loader,
                            weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    train_losses_history = []
    val_losses_history = []
    val_accuracies_history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        learning_rates = []

        for batch_idx, batch in enumerate(train_loader):
            with autocast():
                loss = model.training_step(batch)
                loss = loss / accumulation_steps
                train_losses.append(loss)
                loss.backward()

            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            # Record & update learning rate
            learning_rates.append(get_learning_rate(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['learning_rates'] = learning_rates
        model.epoch_end(epoch, result)

        log_metrics(
            f'train_loss: {result["train_loss"]} || val_loss: {result["val_loss"]} || val_acc: {result["val_acc"]}',
            "cifar100_perf_training.log")

        train_losses_history.append(result['train_loss'])
        val_losses_history.append(result['val_loss'])
        val_accuracies_history.append(result['val_acc'])
        history.append(result)

    return history, train_losses_history, val_losses_history, val_accuracies_history


start_time = time.time()
history, train_losses, val_losses, val_accuracies = fit_one_cycle_with_perf(
    epochs, max_learning_rate, model, train_loader, test_loader,
    grad_clip=grad_clip, weight_decay=weight_decay, opt_func=optimizer_function
)

print('Training time: {:.2f} s'.format(time.time() - start_time))


def save_loss_graph(filename, train_losses, val_losses):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')

    # find position of lowest validation loss
    min_val_loss_epoch = val_losses.index(min(val_losses)) + 1
    plt.axvline(min_val_loss_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(filename, bbox_inches='tight')


save_loss_graph('CIFAR100_loss_plot.png', train_losses, val_losses)
