import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import XLNetForSequenceClassification, Trainer, TrainingArguments
from custom_dataset import CustomDataset
import time
import psutil  # Library for system monitoring
import torch.cuda.amp as amp

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

# Function for model training using mixed precision
def train_model_mixed_precision(train_loader, optimizer):
    scaler = amp.GradScaler()
    model.to('cpu')  # Move model to CPU
    for epoch in range(training_args.num_train_epochs):
        for batch in train_loader:
            inputs = batch["input_ids"]
            labels = batch["labels"]
            optimizer.zero_grad()
            with amp.autocast():
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {"accuracy": (preds == labels).mean()}

# Define XLNet model for sequence classification
device = torch.device("cpu")  # Using CPU for training
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

# Training and Evaluation on CPU
def train_and_evaluate_on_cpu():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model_mixed_precision(train_loader, optimizer)
    trainer.evaluate()

measure_phase_time_memory("Training and Evaluation on CPU", train_and_evaluate_on_cpu)
