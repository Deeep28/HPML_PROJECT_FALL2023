import torch
import torchvision
from torchvision import transforms
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Select and load the dataset for Image Classification (CIFAR-100)
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform)

# Define the ResNet50 model for CIFAR-100
model = torchvision.models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)  # CIFAR-100 has 100 classes

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up Distributed Training if available
if torch.cuda.device_count() > 1:
    torch.distributed.init_process_group(
        backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)

# Move model to device
model.to(device)

# Measure memory and I/O usage
data_memory_usage = torch.cuda.memory_allocated()
io_start_time = time.time()
for inputs, labels in train_dataset:
    # Perform I/O operations
    pass
io_end_time = time.time()
io_time = io_end_time - io_start_time

# Print memory and I/O usage information
print(f"Initial GPU Memory Usage: {data_memory_usage / 1e9} GB")
print(f"I/O Time: {io_time} seconds")

# Implement data loading optimizations
# DataLoader options like prefetching, num_workers, pin_memory, etc.
optimized_train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

# Implement mixed-precision training
scaler = GradScaler()

# Train the model with optimized DataLoader and mixed-precision training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

model.train()
for epoch in range(3):  # Assuming 3 epochs for demonstration
    for inputs, labels in optimized_train_loader:
        optimizer.zero_grad()

        # Use autocast to enable mixed-precision training
        with autocast():
            # Ensure input is on the same device as the model
            inputs = inputs.to(device)
            # Ensure labels are on the same device as the model
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

training_time = time.time() - start_time

# Measure impact on inference
model.eval()
with torch.no_grad():
    start_time = time.time()
    for inputs, labels in optimized_train_loader:
        # Ensure input is on the same device as the model
        inputs = inputs.to(device)
        # Ensure labels are on the same device as the model
        labels = labels.to(device)
        outputs = model(inputs)
    inference_time = time.time() - start_time

# Print the impact of optimizations
print(
    f"GPU Memory Usage after optimizations: {torch.cuda.memory_allocated() / 1e9} GB")
print(f"Training Time with Optimizations: {training_time} seconds")
print(f"Inference Time with Optimizations: {inference_time} seconds")

# Cleanup distributed training resources
if torch.cuda.device_count() > 1:
    torch.distributed.destroy_process_group()
