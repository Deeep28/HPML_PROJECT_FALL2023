import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt

#  Select and load the dataset for Image Classification
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# Define the ResNet50 model
model = torchvision.models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # Assuming 10 classes for CIFAR-10

# Train the baseline model on CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Measure time for baseline CPU training
start_time = time.time()
train(model, train_loader, criterion, optimizer, device)
end_time = time.time()
print(f"Baseline Training Time on {device}: {end_time - start_time} seconds")

# Implement data loading optimizations
# DataLoader options like prefetching, num_workers, pin_memory, etc.
optimized_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

# Measure baseline data loading time
baseline_start_time = time.time()
for inputs, labels in train_loader:
    # Perform baseline data loading operations
    train(model, train_loader, criterion, optimizer, device)
baseline_end_time = time.time()
baseline_data_loading_time = baseline_end_time - baseline_start_time

# Measure optimized data loading time
optimized_start_time = time.time()
for inputs, labels in optimized_train_loader:
    # Perform optimized data loading operations
    train(model, optimized_train_loader, criterion, optimizer, device)
optimized_end_time = time.time()
optimized_data_loading_time = optimized_end_time - optimized_start_time

# Print information for analysis
print(f"Number of Workers in DataLoader: {optimized_train_loader.num_workers}")
print(f"Pin Memory in DataLoader: {optimized_train_loader.pin_memory}")
print(f"Baseline Data Loading Time: {baseline_data_loading_time} seconds")
print(f"Optimized Data Loading Time: {optimized_data_loading_time} seconds")

# Visualization using matplotlib
plt.figure(figsize=(10, 5))
plt.title("Data Loading Time Comparison")
plt.bar(["Baseline", "Optimized"], [
        baseline_data_loading_time, optimized_data_loading_time])
plt.xlabel("Data Loading Method")
plt.ylabel("Time (seconds)")
plt.show()
