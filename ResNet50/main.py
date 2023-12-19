from torch.autograd import profiler
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import time
import matplotlib.pyplot as plt

# Define a function to measure time

def measure_time(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

# Select and load the dataset for Image Classification
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True)

# Define the ResNet50 model
model = torchvision.models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 100)  

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

# Implement data loading optimizations and loop through different num_workers values
num_workers_values = [0, 2, 4, 8]
data_loading_times = []

for num_workers in num_workers_values:
    optimized_train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # Measure optimized data loading time
    optimized_data_loading_time = measure_time(
        train, model, optimized_train_loader, criterion, optimizer, device)
    data_loading_times.append(optimized_data_loading_time)

    # Print information for analysis
    print(
        f"Number of Workers in DataLoader: {optimized_train_loader.num_workers}")
    print(f"Pin Memory in DataLoader: {optimized_train_loader.pin_memory}")
    print(
        f"Optimized Data Loading Time with {num_workers} workers: {optimized_data_loading_time} seconds")

# Visualize the impact of num_workers on data loading time
plt.plot(num_workers_values, data_loading_times, marker='o')
plt.title('Impact of num_workers on Data Loading Time')
plt.xlabel('num_workers')
plt.ylabel('Data Loading Time (seconds)')
plt.show()

# Function for mixed-precision training

def train_mixed_precision(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Automatic mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Model Training


# Measure training time with mixed-precision
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

mixed_precision_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

# Enable mixed-precision training
scaler = torch.cuda.amp.GradScaler()

mixed_precision_training_time = measure_time(
    train_mixed_precision, model, mixed_precision_train_loader, criterion, optimizer, device, scaler
)
print(
    f"Training Time with Mixed-Precision on {device}: {mixed_precision_training_time} seconds")


# Function for training with learning rate scheduling
def train_with_scheduler(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()


baseline_training_time = measure_time(
    train, model, train_loader, criterion, optimizer, device)

# Measure training time with learning rate scheduling
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

lr_schedule_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.1)

lr_schedule_training_time = measure_time(
    train_with_scheduler, model, lr_schedule_train_loader, criterion, optimizer, device, lr_scheduler
)
print(
    f"Training Time with Learning Rate Scheduling on {device}: {lr_schedule_training_time} seconds")

# Model Training with Profiling and Data Parallelism

# Profiling setup
profiler_enabled = True  # Set to False if we don't want profiling
profiling_output_file = "profiling_results.txt"

# Wrap the model with DataParallel for multi-GPU training
model = nn.DataParallel(model)

# Training loop with profiling
if profiler_enabled:
    with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            model.eval()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
else:
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Measure training time with data parallelism
parallel_training_time = measure_time(
    train, model, train_loader, criterion, optimizer, device)
print(
    f"Training Time with Parallel training {device}: {parallel_training_time} seconds")

# Print profiling results
if profiler_enabled:
    print(prof.key_averages().table(sort_by="cpu_time_total"))

# Inference
model.eval()  # Set the model to evaluation mode
inference_start_time = time.time()
for inputs, labels in train_loader:
    # Perform baseline inference operations
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
inference_end_time = time.time()
inference_time = inference_end_time - inference_start_time
print(f"Baseline Inference Time on {device}: {inference_time} seconds")

# Implement inference optimizations if necessary

# Measure inference time with batch inference
model.to(device)
batch_inference_start_time = time.time()
batch_inference_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)
for inputs, labels in batch_inference_loader:
    # Perform batch inference operations
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
batch_inference_end_time = time.time()
batch_inference_time = batch_inference_end_time - batch_inference_start_time
print(
    f"Inference Time with Batch Inference on {device}: {batch_inference_time} seconds")

# Function for training with gradient clipping
def train_with_gradient_clip(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        optimizer.step()

# Experiment with gradient clipping
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
clipping_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

# Apply gradient clipping
gradient_clip_value = 0.1
gradient_clipper = torch.nn.utils.clip_grad_norm_(
    model.parameters(), gradient_clip_value)

clipping_training_time = measure_time(
    train_with_gradient_clip, model, clipping_train_loader, criterion, optimizer, device)
print(
    f"Training Time with Gradient Clipping on {device}: {clipping_training_time} seconds")

# Plotting the results
optimizations = ['Baseline', 'Mixed-Precision',
                 'LR Scheduling', 'Gradient Clipping']
times = [
    baseline_training_time,
    mixed_precision_training_time,
    lr_schedule_training_time,
    clipping_training_time
]

plt.figure(figsize=(10, 6))
plt.bar(optimizations, times, color=['blue', 'orange', 'green', 'red'])
plt.title('Training Time with Different Optimizations')
plt.xlabel('Optimization Techniques')
plt.ylabel('Training Time (seconds)')
plt.show()
