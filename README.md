<p align="center">
  <img src="assets/logo.svg" alt="Learning-DL Logo" width="540">
</p>

# Learning-DL

Implementations of fundamental neural network architectures and computer vision classifiers from first principles in NumPy and PyTorch.

## Architecture

* **Vanilla Neural Network (`Vanilla-Neural-Network/`)**: Multilayer perceptron built from scratch in NumPy using He weight initialization, forward affine transformations, manual backpropagation, and activation functions (ReLU, Sigmoid, Softmax).
* **MNIST Handwritten Digit Classifier (`MNIST-Digits/`)**: PyTorch convolutional network featuring 2D convolutions, batch normalization, max pooling, and negative log likelihood loss.
* **Fashion-MNIST Classifier (`MNIST-Fashion/`)**: Multi-stage feature extraction CNN with batch normalization, dropout regularization, and dense linear classification heads.
* **AlexNet CIFAR-10 (`CIFAR-10/AlexNet/`)**: Scaled PyTorch implementation of the classic 8-layer deep convolutional network (5 convolutional layers + 3 dense layers) with adaptive average pooling and dropout.

## Repository Layout

```text
├── Object-Classification/
│   ├── Vanilla-Neural-Network/
│   │   ├── neural_network.py    # NumPy MLP with He init and backprop
│   │   └── logistic_regression.py
│   ├── MNIST-Digits/
│   │   ├── model.py             # Conv2D + BatchNorm digit extractor
│   │   └── train.py             # PyTorch training and evaluation loop
│   ├── MNIST-Fashion/
│   │   ├── model.py             # Multi-stage Fashion-MNIST CNN
│   │   └── train.py
│   └── CIFAR-10/
│       └── AlexNet/
│           ├── alexnet.py       # 8-layer AlexNet architecture
│           ├── train.py         # DataLoader pipeline and checkpointing
│           └── requirements.txt
```

## Prerequisites

* Python 3.9+
* PyTorch and Torchvision
* NumPy, Matplotlib, Pillow

## Usage

### Run Scratch Neural Network

```bash
python3 Object-Classification/Vanilla-Neural-Network/neural_network.py
```

### Train AlexNet on CIFAR-10

```bash
cd Object-Classification/CIFAR-10/AlexNet
pip install -r requirements.txt
python3 train.py
```

### Train MNIST Classifiers

```bash
python3 Object-Classification/MNIST-Digits/train.py
python3 Object-Classification/MNIST-Fashion/train.py
```

## License

GPL-3.0-or-later. See LICENSE for details.