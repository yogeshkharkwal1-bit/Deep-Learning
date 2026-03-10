# ANN Model - Deep Learning Implementation

This folder contains Artificial Neural Network (ANN) implementations using PyTorch for both **regression** and **classification** tasks.

## 📁 Files Overview

### 1. **Deep_Learning.ipynb** - Regression Model
A regression task that predicts power energy output based on physical parameters.

#### Dataset
- **File**: `deep_learning_dataset1.csv`
- **Features** (4 input variables):
  - `AT` - Atmospheric Temperature
  - `V` - Vacuum
  - `AP` - Atmospheric Pressure
  - `RH` - Relative Humidity
- **Target**: `PE` - Power Energy Output (continuous value)

#### Model Architecture
```
Input Layer (4) → Hidden Layer 1 (6 neurons + ReLU) → 
Hidden Layer 2 (6 neurons + ReLU) → Output Layer (1)
```

#### Key Components
- **Preprocessing**: StandardScaler normalization
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training Parameters**:
  - Epochs: 100
  - Batch Size: 32
  - Train-Test Split: 80-20
- **Features**: 
  - Early model checkpointing (saves best model based on validation loss)
  - Training and validation loss tracking
  - Model evaluation with test predictions

#### Output
- `best_model.pt` - Saved model weights with best validation performance
- Loss visualization plots (training vs validation)
- R² score and MSE metrics

---

### 2. **DL_for_classification.ipynb** - Multi-class Classification Model
A classification task that classifies date fruit varieties using physical characteristics.

#### Dataset
- **File**: `DateFruit_Dataset - Copy.csv`
- **Features**: Multiple physical characteristics of date fruits (normalized features)
- **Target**: `Class` - 7 different date fruit varieties (multi-class classification)
- **Classes**: 7 fruit types encoded with LabelEncoder

#### Model Architecture
```
Input Layer (Features) → Hidden Layer 1 (64 neurons + ReLU) → 
Hidden Layer 2 (64 neurons + ReLU) → Output Layer (7)
```

#### Key Components
- **Preprocessing**: 
  - StandardScaler normalization
  - LabelEncoder for categorical target variable
- **Loss Function**: CrossEntropyLoss (multi-class classification)
- **Optimizer**: Adam
- **Training Parameters**:
  - Epochs: 100
  - Batch Size: 32
  - Train-Test Split: 80-20
- **Evaluation**: 
  - Classification accuracy on test set
  - Predicted vs actual class comparison

#### Output
- Accuracy metrics on test data
- Per-batch training loss monitoring
- Predicted class labels for test samples

---

## 🔧 Technologies Used
- **PyTorch**: Neural network framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Scikit-learn**: Data preprocessing, metrics, train-test splitting
- **Matplotlib**: Visualization

---

## 💾 Model Files
- `best_model.pt` - Saved weights from the best regression model (Deep_Learning.ipynb)

---

## 📊 Data Files
- `deep_learning_dataset1.csv` - Regression dataset (4 features, continuous target)
- `DateFruit_Dataset - Copy.csv` - Classification dataset (7 fruit classes)

---

## 🚀 How to Use

### For Regression (Deep_Learning.ipynb)
1. Load the dataset with features: AT, V, AP, RH
2. Model trained on 100 epochs, saves best model based on validation loss
3. Evaluate using MSE and R² score
4. Check loss visualization to monitor training behavior

### For Classification (DL_for_classification.ipynb)
1. Load DateFruit dataset with 7 fruit classes
2. Model trained for 100 epochs with CrossEntropyLoss
3. Calculate classification accuracy on test set
4. Review predicted vs actual classifications

---

## 📈 Model Performance Metrics

### Regression Model
- **Training Loss**: Tracked throughout epochs
- **Validation Loss**: Used for model selection
- **Test Evaluation**: MSE and R² score

### Classification Model
- **Training Loss**: CrossEntropyLoss per epoch
- **Test Accuracy**: Percentage of correct classifications

---

## 📝 Notes
- Both models use standardized/normalized inputs for better training stability
- Batch normalization via StandardScaler is applied in preprocessing
- Early model checkpointing used in regression to save best model
- Adam optimizer works well for both regression and classification tasks

