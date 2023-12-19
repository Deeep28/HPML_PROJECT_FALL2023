# HPML_PROJECT_FALL2023
Team: Deep Jiteshkumar Sakhiya (ds7000) | Harshkumar Navadiya (hn2276)

# Project Title: Multi-Domain Machine Learning Model Optimization and Comparative Analysis

## Description
This project focuses on a comprehensive study and optimization of machine learning models across multiple domains, primarily emphasizing Image Classification using the CIFAR10, CIFAR100, and ImageNet datasets. The project utilizes the ResNet50 model implemented in PyTorch for benchmarking and optimization purposes. It also uses the XLNet for NLP and using the Amazon review Datasets

## Project Milestones

### Milestone 1: Baseline Model Implementation
- [x] Implement ResNet50 for image classification on CIFAR10, CIFAR100 and XLNet for Text Classification.
- [x] Train the baseline model on CPU and GPU.
- [x] Measure time and memory/I.O. usage during data pre-processing, loading, training, and inference.

### Milestone 2: Performance Tuning
- [x] Apply optimization techniques including data loading optimizations and mixed-precision training.
- [x] Re-train models after optimizations and measure performance improvements.
- [x] Data Loading Optimizations, Gradient accumulation, Automatic Mixed Precision Training, Pin_memory


### Milestone 3: Comparative Analysis
- [x] Compare and analyze the results to identify the most effective optimizations for each domain.
- [x] Implement distributed training techniques and measure their impact on training and inference.
- [x] Conduct additional optimizations and analyze their impact on performance.

## Repository Structure

- **datasets:** CIFAR10 & CIFAR100 https://www.cs.toronto.edu/~kriz/cifar.html, ImageNet https://www.image-net.org/download.php, Amazon review https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews 
- **models:** Includes the implementation of ResNet50 and XLNet.
- **GPU:** All measurements were done on Nvidia v100 gpu with mixed precision.
- **results:** Storage for logs, charts, and tables generated during experiments.

## Code Execution

### Dependencies
- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib

### Commands
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Install dependencies
pip install -r requirements.txt

# Execute baseline model training on CPU
python resnet50_100.py --device cpu

# Execute baseline model training on GPU
python resnet50_100.py --device gpu

# Execute optimized model training
python resnet50_100_optimize.py

# Additional commands for distributed training, profiling, and other optimizations
# ...

```
### Example commands to execute the code

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output="/scratch/hn2276/Project/c100.txt"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:4

cd /scratch/hn2276/Project

module load intel/19.1.2
module load anaconda3/2020.07
module load python/intel/3.8.6
module load cuda/11.6.2

# Run your Python script
python resnet50_100.py

```
## Results and Observations

- **Baseline Model Results:**
  - ImageNet: 77.54% accuracy (Adam), 78.24% accuracy (SGD)
  - CIFAR10: 97.67% accuracy (Adam), 97.46% accuracy (SGD)
  - CIFAR100: 86.07% accuracy (Adam), 85.17% accuracy (SGD)

- **Optimized Model Results:**
  - Significant improvements in training and inference times.
  - Data loading optimizations and mixed-precision training were particularly effective.

- **Charts and Tables:**
  - See the `results` directory for visualizations, charts, and tables generated during experiments.

For detailed analysis and additional insights, refer to the notebooks in the `notebooks` directory.

Feel free to explore the code, experiment with different configurations, and contribute to the project!
