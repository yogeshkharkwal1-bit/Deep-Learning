# Deep Learning Repository

This repository covers fundamental **Deep Learning** concepts and implementations using PyTorch, including Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

## 📚 Overview

Deep Learning is a branch of Machine Learning that uses artificial neural networks with multiple layers to learn hierarchical representations of data. This repo demonstrates core deep learning techniques through practical implementations on real-world datasets.

### What is Deep Learning?
Deep Learning is based on **Artificial Neural Networks** inspired by the human brain's structure. Key concepts include:
- **Neurons**: Basic computational units that perform weighted sum + activation
- **Layers**: Organized groups of neurons (input, hidden, output)
- **Weights & Biases**: Learnable parameters that are optimized during training
- **Activation Functions**: Non-linear functions (ReLU, Sigmoid, Tanh) that enable networks to learn complex patterns
- **Backpropagation**: Efficient algorithm to compute gradients and update weights
- **Loss Functions**: Measure the difference between predicted and actual values

---

## 📁 Project Structure

### **1. ANN Model** - Artificial Neural Networks
Two different neural network applications demonstrating regression and classification:

#### Deep_Learning.ipynb - Regression
- **Task**: Predict power energy output from physical parameters
- **Features**: Atmospheric temperature, vacuum, pressure, humidity
- **Architecture**: 2-layer ANN (4 → 6 → 6 → 1)
- **Loss Function**: Mean Squared Error (MSE)
- **Key Feature**: Model checkpointing (saves best model)
- **Metrics**: MSE, R² Score

#### DL_for_classification.ipynb - Multi-class Classification
- **Task**: Classify date fruit varieties
- **Features**: Physical characteristics of date fruits
- **Target**: 7 different fruit classes
- **Architecture**: 2-layer ANN (Features → 64 → 64 → 7)
- **Loss Function**: CrossEntropyLoss
- **Metrics**: Classification accuracy

**When to use ANN:**
- Tabular/structured data
- Quick training on smaller datasets
- Regression and basic classification tasks

---

### **2. CNN Model** - Convolutional Neural Networks
Deep CNN for image classification on CIFAR-10 dataset:

#### CNN_for_CIFAR10.ipynb
- **Task**: Classify 10 object categories in images
- **Dataset**: CIFAR-10 (50K training, 10K test 32×32 RGB images)
- **Architecture**: 3-layer CNN with fully connected layers
  - Conv Block 1: 32 filters → MaxPool
  - Conv Block 2: 64 filters → MaxPool
  - Conv Block 3: 128 filters → MaxPool
  - Dense layers: 256 hidden units → 10 output classes
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Metrics**: Classification accuracy

**Why CNN for Images:**
- **Convolutional Layers**: Extract spatial features (edges, textures, patterns)
- **Parameter Sharing**: Reduces parameters compared to fully connected networks
- **Translation Invariance**: Robust to small shifts in images
- **Hierarchical Learning**: Lower layers learn simple features, deeper layers learn complex objects

---

## 🧠 Key Deep Learning Concepts

### Neural Network Components
| Component | Purpose | Example |
|-----------|---------|---------|
| **Input Layer** | Receives raw data | 4 features or 3×32×32 image pixels |
| **Hidden Layers** | Learn representations | Multiple layers with varying neurons |
| **Output Layer** | Produces predictions | 1 value (regression) or 10 classes |
| **Activation** | Add non-linearity | ReLU, Sigmoid, Tanh |
| **Weights/Bias** | Learnable parameters | Optimized during training |

### Training Process
1. **Forward Pass**: Data flows through network → predictions
2. **Loss Calculation**: Compare predictions vs actual values
3. **Backward Pass**: Compute gradients using backpropagation
4. **Parameter Update**: Optimizer updates weights to reduce loss
5. **Repeat**: Multiple epochs until convergence

### Key Optimizers Implemented
- **Adam**: Adaptive moment estimation, combines momentum and RMSprop
- Used in both ANN and CNN models
- Effective for most deep learning tasks

### Loss Functions
- **MSE (Mean Squared Error)**: For regression tasks → ANN regression
- **CrossEntropyLoss**: For multi-class classification → ANN + CNN

---

## 🛠 Technologies & Libraries

```python
PyTorch          # Deep learning framework
Torchvision      # Computer vision utilities
Pandas           # Data manipulation
NumPy            # Numerical computations
Scikit-learn     # Preprocessing & metrics
Matplotlib       # Visualization
```

---

## 📊 Workflow Comparison

### Regression (ANN with MSE)
```
Data → Preprocessing → Train/Test Split → Scale Data → ANN Model 
→ MSE Loss → Backpropagation → Save Best Model → Evaluate
```

### Classification - Tabular (ANN with CrossEntropyLoss)
```
Data → Label Encoding → Scale Data → ANN Model → CrossEntropyLoss 
→ Backpropagation → Evaluate Accuracy
```

### Classification - Images (CNN with CrossEntropyLoss)
```
Images → Normalize → DataLoader → CNN Model (Conv+Pool) 
→ CrossEntropyLoss → Backpropagation → Evaluate Accuracy
```

---

## 🚀 How to Use This Repository

### For ANN Models
1. Navigate to `ANN Model/` folder
2. Open either notebook:
   - `Deep_Learning.ipynb` for regression
   - `DL_for_classification.ipynb` for classification
3. Run cells sequentially to train and evaluate
4. Review loss plots and metrics

### For CNN Model
1. Navigate to `CNN Model/` folder
2. Open `CNN_for_CIFAR10.ipynb`
3. CIFAR-10 data auto-downloads on first run
4. View training loss and test accuracy

---

## 📈 Learning Progression

**ANN Models**: Great for understanding:
- Data preprocessing and normalization
- Neural network fundamentals
- Backpropagation and optimization
- Regression vs classification concepts
- Model evaluation metrics

**CNN Model**: Advances to:
- Convolutional layers and filters
- Spatial feature extraction
- Image-specific architectures
- Deep learning best practices

---

## 💡 Key Takeaways

1. **Deep Learning Works with Layers**: More layers = higher complexity, but risk of overfitting
2. **Data Preprocessing Matters**: Normalization (StandardScaler) is crucial
3. **Loss Functions Guide Learning**: Different tasks need different loss functions
4. **Batch Processing Improves Efficiency**: Batches enable parallel computation
5. **Model Checkpointing Saves Time**: Save best model during training
6. **Different Architectures for Different Data**:
   - ANN: Tabular data
   - CNN: Image data
   - RNN: Sequential/Time-series data
   - Transformers: NLP and complex sequences

---

## 📚 Concepts Covered

✅ Neural Network fundamentals  
✅ Activation functions and ReLU  
✅ Backpropagation algorithm  
✅ Gradient descent and optimizers (Adam)  
✅ Loss functions (MSE, CrossEntropyLoss)  
✅ Data preprocessing and normalization  
✅ Train/validation/test splits  
✅ Convolutional layers and filters  
✅ Max pooling and feature maps  
✅ Multi-class classification  
✅ Regression with neural networks  
✅ Model evaluation and metrics  

---

## 📝 Notes
- All models use PyTorch for flexibility and efficiency
- Both ANN notebooks include data preprocessing and model evaluation
- CNN_for_CIFAR10 demonstrates image-specific deep learning techniques
- Models can be adapted for different datasets by modifying input/output dimensions
