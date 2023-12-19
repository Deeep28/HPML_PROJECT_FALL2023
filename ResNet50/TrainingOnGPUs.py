import torch
import torch_xla.core.xla_model as xm
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from torch.cuda import amp
from torch.cuda.amp import autocast, GradScaler
import psutil


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the ResNet50 model
model = torchvision.models.resnet50(pretrained=False)
model.to(device)
# Load the dataset
train_dataset = datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transforms.ToTensor())

# Use DataLoader
train_loader = DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_on_gpus():
    # Use GPU Dataset for data loading
    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_dataset = TensorDataset(*[torch.tensor(x) for x in train_dataset])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=100, shuffle=True, num_workers=4, pin_memory=True)

    # Set model and loss to GPU
    model.to(xm.xla_device())
    criterion = nn.CrossEntropyLoss().to(xm.xla_device())

    # Start timer
    start_time = time.time()

    # Set the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Iterate through the training phase
    for epoch in range(2):  # limit epochs to 2 for this example
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get data to GPU asynchronously
            data, target = data.to(xm.xla_device()), target.to(xm.xla_device())

            # Forward
            output = model(data)

            # Loss
            loss = criterion(output, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Step
            optimizer.step()

    # Calculate and print the time taken
    time_taken = time.time() - start_time
    print('Time taken for training on GPUs:', time_taken, 'seconds')

    # Calculate and print memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print('Memory usage for training on GPUs:',
          memory_info.rss / (1024 ** 2), 'MB')
