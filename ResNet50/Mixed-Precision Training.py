import torch
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

def mixed_precision_training():
    # Set model and loss to GPU
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    # Start timer
    start_time = time.time()

    # Set the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Use AMP
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # Iterate through the training phase
    for epoch in range(2):  # limit epochs to 2 for this example
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get data to GPU asynchronously
            data, target = data.cuda(), target.cuda()

            # Forward
            with amp.autocast():
                output = model(data)

            # Loss
            loss = criterion(output, target)

            # Backward
            optimizer.zero_grad()
            with amp.autocast():
                loss.backward()

            # Step
            optimizer.step()

    # Calculate and print the time taken
    time_taken = time.time() - start_time
    print('Time taken for mixed precision training:', time_taken, 'seconds')

    # Calculate and print memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print('Memory usage for mixed precision training:',
          memory_info.rss / (1024 ** 2), 'MB')
