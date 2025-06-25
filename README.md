# ComplexNeuralNetworks

ComplexNeuralNetworks is a research-oriented repository focused on the development and experimentation of neural network architectures that operate on complex-valued data. This project includes implementations of both real and complex-valued models, custom CNNs, ResNet variants, and segmentation networks such as UNet and LadderNet. The codebase is modular and supports training, evaluation, and visualization for a variety of datasets.

## Features

- **Complex-valued neural network layers and models**
- **Custom CNN architectures** for both real and complex domains
- **ResNet variants** (real and complex)
- **Segmentation models**: UNet, LadderNet
- **Flexible dataloaders** for standard and complex datasets
- **Training and evaluation scripts** with experiment tracking
- **Visualization tools** for results and feature analysis
- **Kubernetes deployment** scripts for scalable training

## Directory Structure


## Getting Started

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Snehgal/ComplexNeuralNetworks.git
    cd ComplexNeuralNetworks
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

- **Training a model:**  
  Use the scripts in `Concept/` (e.g., `main.py`, `train.py`, `newTrain.py`) to train and evaluate models.  
  Example:
    ```bash
    python Concept/main.py
    ```

- **Segmentation tasks:**  
  Use the scripts in `Segmentation/` for segmentation experiments.

- **Kubernetes deployment:**  
  See `Concept/Kubernetes/README.md` for instructions on running distributed training on a cluster.

## Supported Models

- LeNet, LeNet2x, ComplexLeNet
- CustomCNN, CustomCNN2x, ComplexCustomCNN
- ResNet18, ResNet18x2, ComplexResNet18
- UNet, LadderNet (for segmentation)

## Datasets

- FashionMNIST, ComplexFashionMNIST
- CIFAR10, ComplexCIFAR10
- Custom datasets (see dataloader scripts for details)

## Benchmarks

| Model                | Dataset           | Params    | Accuracy (%) | Notes                |
|----------------------|------------------|-----------|--------------|----------------------|
| LeNet                | FashionMNIST     |           |              |                      |
| ComplexLeNet         | ComplexFashionMNIST |         |              |                      |
| CustomCNN            | CIFAR10          |           |              |                      |
| ComplexCustomCNN     | ComplexCIFAR10   |           |              |                      |
| ResNet18             | CIFAR10          |           |              |                      |
| ComplexResNet18      | ComplexCIFAR10   |           |              |                      |
| UNet                 | Custom           |           |              | Segmentation         |
| LadderNet            | Custom           |           |              | Segmentation         |

## Credits

Chirag Sehgal and Jasjyot Singh Gulati, under the guidance of Dr. Anubha Gupta.

