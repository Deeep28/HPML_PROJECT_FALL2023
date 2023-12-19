import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import XLNetForSequenceClassification, Trainer, TrainingArguments
from custom_dataset import CustomDataset
import time
import psutil  # Library for system monitoring
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.parallel
import cProfile

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

# Load tokenized data from saved .pt files
train_data = torch.load('train_dataset.pt')
test_data = torch.load('test_dataset.pt')

train_encodings = train_data['train_encodings']
train_labels = train_data['train_labels']
test_encodings = test_data['test_encodings']
test_labels = test_data['test_labels']

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Function for optimized data loading and pre-processing
def load_and_preprocess_data_optimized():
    # Create train and test datasets using CustomDataset
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Define batch size and create data loaders with optimized settings
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), num_workers=4)  # Adjust num_workers as needed
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  # Adjust num_workers as needed

    return train_loader, test_loader

# Function for model training using distributed data parallelism and mixed precision
def train_model_parallel_mixed_precision(train_loader, optimizer):
    model_parallel = nn.DataParallel(model)
    scaler = amp.GradScaler()

    for epoch in range(training_args.num_train_epochs):
        for batch in train_loader:
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model_parallel(inputs, labels=labels)
                loss = outputs.loss.mean()  # Adjust aggregation if needed
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}

# Define XLNet model for sequence classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2).to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100000,
    gradient_accumulation_steps=10
)

# Define Trainer for mixed precision training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Measure time and memory for data loading and pre-processing
train_loader, _ = load_and_preprocess_data_optimized()

measure_phase_time_memory("Data Loading and Preprocessing", lambda: None)

# Training and Evaluation on GPU (if available)
if torch.cuda.is_available():
    def train_and_evaluate_on_gpu():
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Profiling the training process
        profiler = cProfile.Profile()
        profiler.enable()

        start_time_gpu = time.time()
        train_model_parallel_mixed_precision(train_loader, optimizer)
        end_time_gpu = time.time()
        training_time_gpu = end_time_gpu - start_time_gpu
        print(f"Training time on GPU: {training_time_gpu} seconds")

        profiler.disable()
        profiler.print_stats(sort='time')

        # Model Evaluation on GPU
        start_time_eval_gpu = time.time()
        trainer.evaluate()
        end_time_eval_gpu = time.time()
        evaluation_time_gpu = end_time_eval_gpu - start_time_eval_gpu
        print(f"Evaluation time on GPU: {evaluation_time_gpu} seconds")

    # Measure time and memory for training and evaluation on GPU
    measure_phase_time_memory("Training and Evaluation on GPU", train_and_evaluate_on_gpu)
else:
    print("No GPU available, skipping GPU training and evaluation.")
