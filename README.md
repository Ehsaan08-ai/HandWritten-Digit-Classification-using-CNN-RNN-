# ✍️ Handwritten Digit Classification using CNN & RNN

A deep learning project that classifies handwritten digits (0–9) from the **MNIST dataset** using two different neural network architectures — a **Convolutional Neural Network (CNN)** and a **Recurrent Neural Network (RNN)** — built with PyTorch.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
  - [CNN Architecture](#cnn-architecture)
  - [RNN Architecture](#rnn-architecture)
- [Results](#results)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Saved Models](#saved-models)
- [License](#license)

---

## 🔍 Overview

This project explores and compares two neural network approaches for image classification on the classic MNIST handwritten digits dataset:

| Model | Approach | Test Accuracy |
|-------|----------|--------------|
| CNN   | Spatial feature extraction via convolution | **98.6%** |
| RNN   | Sequential row-by-row pixel processing | **96.33%** |

Both models were trained for **10 epochs** using the **Adam optimizer** and **Cross-Entropy Loss**.

---

## 📁 Project Structure

```
HandWritten-Digit-Classification-using-CNN-RNN/
│
├── CNN.ipynb              # CNN model: training, evaluation, and saving
├── RNN.ipynb              # RNN model: training, evaluation, and saving
│
├── My_best_model.pth      # Saved CNN model weights
├── RNN_parameters.pth     # Saved RNN model weights
│
├── data/                  # MNIST dataset (auto-downloaded)
│
├── .gitignore
└── README.md
```

---

## 🧠 Model Architectures

### CNN Architecture

The CNN processes images as 2D spatial data, extracting features through stacked convolutional layers.

```
Input (1×28×28)
    │
    ├── Conv2d(1→32, 3×3) → ReLU → MaxPool(2×2)
    ├── Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)
    ├── Conv2d(64→64, 3×3) → ReLU → MaxPool(2×2)
    │
    └── Flatten
        ├── Linear(64 → 128) → ReLU
        └── Linear(128 → 10)   ← Output (10 digit classes)
```

**Key details:**
- 3 convolutional blocks with ReLU activation and max pooling
- 2 fully connected layers for classification
- Trained with Adam (lr=0.001), batch size=64

---

### RNN Architecture

The RNN treats each MNIST image as a **sequence of 28 rows** (each row being 28 pixel values), learning temporal dependencies across rows.

```
Input (28 time steps × 28 features)
    │
    └── RNN(input=28, hidden=128, layers=1, batch_first=True)
        │
        └── Linear(128 → 10)   ← Output (10 digit classes)
```

**Key details:**
- Vanilla RNN with hidden size of 128
- Image reshaped to `(batch, 28, 28)` — 28 rows as time steps
- Trained with Adam (lr=0.001), batch size=64

---

## 📊 Results

### CNN — Test Accuracy: **98.6%**

```
Confusion Matrix:
 [[ 971    1    2    1    0    0    3    2    0    0]
  [   0 1132    1    1    0    0    0    1    0    0]
  [   1    0 1015    4    0    0    2    8    1    1]
  [   0    0    2 1004    0    3    0    0    1    0]
  [   1    0    1    1  960    0    1    0    0   18]
  [   3    1    2    6    0  874    2    1    3    0]
  [   4    6    0    0    3    3  942    0    0    0]
  [   0    4    9    1    0    0    0 1012    2    0]
  [   2    1    3    4    1    1    0    0  959    3]
  [   3    1    1    1    3    2    0    6    1  991]]
```

### RNN — Test Accuracy: **96.33%**

```
Confusion Matrix:
 [[ 963    0    2    0    0    2    7    1    4    1]
  [   0 1114    2    3    1    3    2    1    8    1]
  [   2    5  995    4    0    0    5   12    9    0]
  [   0    0    9  978    0    3    0    6    6    8]
  [   0    1    6    1  913    0    8    0    6   47]
  [   3    1    0   33    0  812    9    1   16   17]
  [   3    2    0    1    3    1  932    0   16    0]
  [   3    1    6    2    0    2    0  997    3   14]
  [   1    5    3    2    1    2    1    4  951    4]
  [   2    1    0    8    3    1    0    7    9  978]]
```

> **Key Takeaway:** The CNN outperforms the RNN by ~2.3%, which is expected since CNNs are inherently designed for spatial/image data, while RNNs are more suitable for sequential/time-series data.

---

## 📦 Dataset

The project uses the **[MNIST Dataset](http://yann.lecun.com/exdb/mnist/)** — a benchmark dataset of handwritten digits.

| Split | Samples |
|-------|---------|
| Training | 60,000 images |
| Testing  | 10,000 images |

- Image size: **28×28 pixels**, grayscale
- Classes: **10** (digits 0 to 9)
- Preprocessing: `ToTensor()` + `Normalize(mean=0.5, std=0.5)`

The dataset is **automatically downloaded** by torchvision into the `./data` directory on first run.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- Jupyter Notebook / JupyterLab

### Install dependencies

```bash
pip install torch torchvision scikit-learn jupyter
```

---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/Ehsaan08-ai/HandWritten-Digit-Classification-using-CNN-RNN-.git
cd HandWritten-Digit-Classification-using-CNN-RNN-
```

2. **Install requirements:**

```bash
pip install torch torchvision scikit-learn jupyter
```

3. **Run the notebooks:**

```bash
jupyter notebook
```

Then open either:
- `CNN.ipynb` — for the CNN model
- `RNN.ipynb` — for the RNN model

4. **To use the pre-trained models** (skip training):
   - The saved weights (`My_best_model.pth` and `RNN_parameters.pth`) are included in the repo.
   - In each notebook, simply run the cell that loads the `state_dict` from the `.pth` file, then run the evaluation cell directly.

---

## 💾 Saved Models

| File | Model | Size |
|------|-------|------|
| `My_best_model.pth` | CNN | ~266 KB |
| `RNN_parameters.pth` | RNN | ~89 KB |

These files store the learned weights and can be loaded without retraining the models.

---

## 🛠️ Training Details

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Cross-Entropy Loss |
| Batch Size | 64 |
| Epochs | 10 |
| Device | CPU / CUDA (auto-detected) |

---

## 📄 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 👤 Author

**Ehsaan** — [@Ehsaan08-ai](https://github.com/Ehsaan08-ai)

> *Feel free to open an issue or submit a pull request if you'd like to contribute or have any suggestions!*
