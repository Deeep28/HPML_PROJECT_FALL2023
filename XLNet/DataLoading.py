import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import XLNetForSequenceClassification, Trainer, TrainingArguments
from custom_dataset import CustomDataset
import time
import psutil  # Library for system monitoring

# Function to measure memory usage
def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / float(2 ** 20)  # Memory usage in MB
    return mem

# Function to measure time and memory for a specific phase
def measure_phase_time_memory(phase_name, phase_func):
    start_time = time.time()
    start_memory = get_memory_usage()
    
    # Execute the phase
    phase_func()
    
    end_time = time.time()
    end_memory = get_memory_usage()

    # Calculate time and memory usage
    phase_time = end_time - start_time
    phase_memory = end_memory - start_memory
    
    print(f"{phase_name} time: {phase_time} seconds")
    print(f"{phase_name} memory usage: {phase_memory} MB")

# Function for data loading and pre-processing without optimization
def load_and_preprocess_data_no_opt():
    start_time = time.time()

    # Load tokenized data from saved .pt files
    train_data = torch.load('train_dataset.pt')
    test_data = torch.load('test_dataset.pt')

    train_encodings = train_data['train_encodings']
    train_labels = train_data['train_labels']
    test_encodings = test_data['test_encodings']
    test_labels = test_data['test_labels']

    # Create train and test datasets using CustomDataset
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Define batch size and create data loaders without optimizations
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    end_time = time.time()
    print(f"Data loading time without optimization: {end_time - start_time} seconds")

    return train_loader, test_loader

# Function for data loading and pre-processing with optimization
def load_and_preprocess_data_with_opt(num_workers):
    start_time = time.time()

    # Load tokenized data from saved .pt files
    train_data = torch.load('train_dataset.pt')
    test_data = torch.load('test_dataset.pt')

    train_encodings = train_data['train_encodings']
    train_labels = train_data['train_labels']
    test_encodings = test_data['test_encodings']
    test_labels = test_data['test_labels']

    # Create train and test datasets using CustomDataset
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Define batch size and create data loaders with optimizations
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    end_time = time.time()
    print(f"Data loading time with {num_workers} workers: {end_time - start_time} seconds")

    return train_loader, test_loader

# Function for model training
def train_model():
    trainer.train()

# Function for model evaluation
def evaluate_model():
    trainer.evaluate()
    
# Define XLNet model for sequence classification
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100000,
    gradient_accumulation_steps=4
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  
    eval_dataset=None  
)

# Measure time and memory for data loading without and with optimization
train_loader_no_opt, test_loader_no_opt = load_and_preprocess_data_no_opt()
trainer.train_dataset = train_loader_no_opt
trainer.eval_dataset = test_loader_no_opt
measure_phase_time_memory("Data Loading without Optimization", lambda: None)

for num_workers in [2, 4, 8, 14, 16, 32]:  # Test different values of num_workers
    train_loader_opt, test_loader_opt = load_and_preprocess_data_with_opt(num_workers)
    trainer.train_dataset = train_loader_opt
    trainer.eval_dataset = test_loader_opt
    measure_phase_time_memory(f"Data Loading with {num_workers} workers", lambda: None)