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

# Load tokenized data from saved .pt files
train_data = torch.load('train_dataset.pt')
test_data = torch.load('test_dataset.pt')

train_encodings = train_data['train_encodings']
train_labels = train_data['train_labels']
test_encodings = test_data['test_encodings']
test_labels = test_data['test_labels']

train_dataset = CustomDataset(train_encodings, train_labels)
test_dataset = CustomDataset(test_encodings, test_labels)

# Function for data loading and pre-processing
def load_and_preprocess_data():
    # Create train and test datasets using CustomDataset
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)

    # Define batch size and create data loaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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
    gradient_accumulation_steps=10
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Measure time and memory for data loading and pre-processing
measure_phase_time_memory("Data Loading and Preprocessing", load_and_preprocess_data)

if torch.cuda.is_available():
    device = torch.device('cuda')
    model.to(device)
    
    def train_and_evaluate_on_gpu():
        # Training on GPU
        start_time_gpu = time.time()
        trainer.train()
        end_time_gpu = time.time()
        training_time_gpu = end_time_gpu - start_time_gpu
        print(f"Training time on GPU: {training_time_gpu} seconds")

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
