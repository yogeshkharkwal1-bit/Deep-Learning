# CNN Model - Convolutional Neural Network for Image Classification

This folder contains a **Convolutional Neural Network (CNN)** implementation using PyTorch for image classification on the CIFAR-10 dataset.

## 📁 Files Overview

### **CNN_for_CIFAR10.ipynb** - Image Classification Model
A deep CNN model trained to classify images into 10 different categories using the CIFAR-10 dataset.

#### Dataset
- **Dataset**: CIFAR-10 (Automatically downloaded from torchvision)
- **Images**: 32×32 RGB color images
- **Classes**: 10 object categories
  1. Airplane
  2. Automobile
  3. Bird
  4. Cat
  5. Deer
  6. Dog
  7. Frog
  8. Horse
  9. Ship
  10. Truck
- **Data Split**: 50,000 training images + 10,000 test images

#### Preprocessing
- **Transformation Pipeline**:
  - Convert images to tensors
  - Normalize with mean (0.5, 0.5, 0.5) and std (0.5, 0.5, 0.5)

#### Model Architecture

**Convolutional Layers:**
```
Input (3×32×32) 
  ↓
Conv2d (3→32 filters, 3×3 kernel, padding=1) + ReLU
  ↓
MaxPool2d (2×2 stride=2) → 32×16×16
  ↓
Conv2d (32→64 filters, 3×3 kernel, padding=1) + ReLU
  ↓
MaxPool2d (2×2 stride=2) → 64×8×8
  ↓
Conv2d (64→128 filters, 3×3 kernel, padding=1) + ReLU
  ↓
MaxPool2d (2×2 stride=2) → 128×4×4
```

**Fully Connected Layers:**
```
Flattened (4×4×128 = 2048) 
  ↓
Dense Layer (2048→256) + ReLU
  ↓
Output Layer (256→10 classes)
```

#### Key Components
- **Loss Function**: CrossEntropyLoss (multi-class classification)
- **Optimizer**: Adam
- **Training Parameters**:
  - Epochs: 10
  - Batch Size: 64
  - Data shuffling: Enabled during training
- **Features**:
  - Three convolutional blocks with max pooling for feature extraction
  - Two fully connected layers for classification
  - Model evaluation with accuracy metrics

#### Output
- Training loss per epoch
- Test set accuracy percentage
- Model predictions on test images

---

## 🔧 Technologies Used
- **PyTorch**: Neural network framework
- **Torchvision**: Vision utilities and CIFAR-10 dataset
- **Torch**: Tensor operations and GPU support (if available)

---

## 📊 How the Model Works

### Feature Extraction (Convolutional Layers)
1. **Conv Block 1**: 32 filters extract basic features (edges, colors)
2. **Conv Block 2**: 64 filters build intermediate features (textures, patterns)
3. **Conv Block 3**: 128 filters extract high-level features (object parts)
4. **MaxPooling**: Reduces spatial dimensions after each convolution, improving efficiency

### Classification (Fully Connected Layers)
1. **Flattening**: Converts 2D feature maps to 1D vector (2048 neurons)
2. **Hidden Layer**: 256 neurons with ReLU activation for feature combination
3. **Output Layer**: 10 fully connected neurons (one per class)

---

## 🚀 Training Process

1. **Forward Pass**: Image → CNN → 10 logits (class scores)
2. **Loss Calculation**: CrossEntropyLoss compares predicted vs actual class
3. **Backward Pass**: Backpropagation computes gradients
4. **Parameter Update**: Adam optimizer updates all weights
5. **Epoch Loop**: Repeats for 10 epochs over entire training set

---

## 📈 Model Evaluation

### Metrics Tracked
- **Training Loss**: Per-epoch average loss over all batches
- **Test Accuracy**: Percentage of correctly classified test images

### Evaluation Method
- No-gradient forward pass on test set for efficiency
- Predicted class = argmax of output logits
- Accuracy = (correct predictions / total images) × 100

---

## 💡 Key Features
- **3-Layer CNN**: Progressively deeper feature extraction (32→64→128 filters)
- **MaxPooling**: Reduces computation and improves translation invariance
- **ReLU Activation**: Non-linearity for learning complex patterns
- **Batch Processing**: Efficient training with batches of 64 images
- **Adam Optimizer**: Adaptive learning rates for stable training

---

## 📝 Notes
- CIFAR-10 is automatically downloaded and cached in `./data/` directory
- Images are normalized to [-1, 1] range for better training stability
- Model uses 10 epochs, which provides reasonable accuracy while maintaining reasonable training time
- The architecture is designed to balance accuracy with computational efficiency for 32×32 images
- MaxPooling (2×2 stride) reduces spatial dimensions from 32→16→8→4, eventually becoming 4×4
