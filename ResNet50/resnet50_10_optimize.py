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
from sklearn.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score

# Define hyperparameters
batch_size = 128
num_epochs = 30
max_learning_rate = 0.001
gradient_clip = 0.01
weight_decay = 0.001
optimizer_func = torch.optim.Adam

# CIFAR-10 mean and std
cifar10_mean = [0.49139967861519607, 0.48215840839460783, 0.44653091444546567]
cifar10_std = [0.24703223246174102, 0.24348512800151828, 0.26158784172803257]

# Data transformations
transform_train = tt.Compose([
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.RandomHorizontalFlip(),
    tt.ToTensor(),
    tt.Normalize(cifar10_mean, cifar10_std, inplace=True)
])

transform_test = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(cifar10_mean, cifar10_std)
])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    "./", train=True, download=True, transform=transform_train
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
)

test_dataset = torchvision.datasets.CIFAR10(
    "./", train=False, download=True, transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size * 2, pin_memory=True, num_workers=2
)

# Device setup
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Move data loaders to device
train_loader = DeviceDataLoader(train_loader, device)
test_loader = DeviceDataLoader(test_loader, device)

# Helper functions
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct_count = torch.sum(preds == labels).item()
    acc = correct_count / len(preds)
    acc = torch.tensor(acc)
    f1 = f1_score(preds.cpu(), labels.cpu(), average='micro')
    return f1, acc

# Model definition
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        f1, acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc, 'f1': f1}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        batch_f1s = [x['f1'] for x in outputs]
        epoch_f1 = np.mean(batch_f1s)
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item(), 'f1': epoch_f1.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}, f1: {result['f1']:.4f}")

# ResNet model
class ResNet50(ImageClassificationBase):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet50, self).__init__()
        self.in_planes = 64
        self.pre_layers = nn.Sequential(
            conv3x3(3, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
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
        layers = [block(self.in_planes, planes, stride, shortcut)]
        self.in_planes = planes * block.expansion
        for _ in range(1, nblocks):
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

# Create ResNet model
resnet_model = to_device(ResNet50(Bottleneck, [3, 4, 6, 3], num_classes=10), device)

# Evaluation function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in loader]
    return model.validation_epoch_end(outputs)

# Learning rate getter
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# Logger function
def logger(log_string, log_file):
    file_log = open(log_file, "a")
    file_log.write(log_string + "\n")
    file_log.close()

def fit_one_cycle_with_mixed_precision(
    epochs, max_lr, model, train_loader, test_loader,
    weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD
):
    torch.cuda.empty_cache()
    history = []
    train_losses = []
    val_losses = []
    val_accs = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    for epoch in range(epochs):
        # Training Phase
        model.train()
        epoch_train_losses = []
        lrs = []

        for batch_idx, batch in enumerate(train_loader):
            with autocast():
                loss = model.training_step(batch)
                loss = loss / accum_iter
                epoch_train_losses.append(loss)
                loss.backward()

            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                # Gradient clipping
                if grad_clip:
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_learning_rate(optimizer))
            sched.step()

        # Validation Phase
        result = evaluate(model, test_loader)
        result['train_loss'] = torch.stack(epoch_train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)

        logger(
            f"train_loss: {result['train_loss']:.4f} || val_loss: {result['val_loss']} || val_acc: {result['val_acc']}",
            "cifar10_mixed_precision_training.log"
        )

        train_losses.append(result['train_loss'])
        val_losses.append(result['val_loss'])
        val_accs.append(result['val_acc'])

        history.append(result)

    return history, train_losses, val_losses, val_accs

def save_loss_plot(filename, train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')

    # Find position of the lowest validation loss
    min_loss_epoch = val_losses.index(min(val_losses)) + 1
    ax.axvline(min_loss_epoch, linestyle='--', color='r', label='Early Stopping Checkpoint')

    ax.set(xlabel='Epochs', ylabel='Loss', title='Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')

# Run the training process
start_time = time.time()
history, train_losses, val_losses, val_accs = fit_one_cycle_with_mixed_precision(
    epochs=num_epochs,
    max_lr=max_learning_rate,
    model=resnet_model,
    train_loader=train_loader,
    test_loader=test_loader,
    grad_clip=gradient_clip,
    weight_decay=weight_decay,
    opt_func=optimizer_func
)
print('Training time: {:.2f} s'.format(time.time() - start_time))

# Save loss plot
save_loss_plot('CIFAR10_loss_plot_mixed_precision.png', train_losses, val_losses)
