import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from torch.cuda.amp import autocast, GradScaler
import psutil


def data_loader_optimizations():
    # Set device (CPU or GPU)
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
    # Start timer
    start_time = time.time()

    # Iterate through the DataLoader
    for epoch in range(2):  # limit epochs to 2 for this example
        for batch_idx, (data, target) in enumerate(train_loader):
            # Get data to GPU asynchronously
            data, target = data.cuda(), target.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            output = model(data)

            # Loss
            loss = criterion(output, target)

            # Backward
            loss.backward()

            # Step
            optimizer.step()

    # Calculate and print the time taken
    time_taken = time.time() - start_time
    print('Time taken for data loading optimizations:', time_taken, 'seconds')

    # Calculate and print memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print('Memory usage for data loading optimizations:',
          memory_info.rss / (1024 ** 2), 'MB')
