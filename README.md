# 🧮 NumPy & TensorFlow Digit Recognizer

A deep learning project to classify handwritten digits (**MNIST dataset**).
This repository contains **two implementations**:

1. **From Scratch (NumPy)** – manual implementation of neural networks (forward/backward propagation, Adam optimizer, L2 regularization).
2. **TensorFlow/Keras** – high-level framework achieving state-of-the-art accuracy with much less code.

---

## 🚀 Features

### 🔹 NumPy Implementation

* Fully connected neural network (multi-layer perceptron) built **from scratch**
* Activation functions: ReLU & Softmax
* Cross-entropy loss function
* Mini-batch gradient descent with **Adam optimizer**
* Supports **L2 regularization** for better generalization
* Achieves **\~97% accuracy** on test data

### 🔹 TensorFlow Implementation

* Built with **Keras Sequential API**
* Uses **dense layers with dropout** for regularization
* Achieves **\~99.7% training accuracy** and **\~99.0% test accuracy**
* Much faster training and simpler implementation

---

## 📂 Project Structure

```
├── train.csv              # Training dataset (Kaggle MNIST)
├── train_model.ipynb      # NumPy implementation (from scratch)
├── model_tf.ipynb         # TensorFlow implementation
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/AlsoMeParth/Numpy-Digit-Recognizer-.git
cd numpy-digit-recognizer
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib tensorflow
```

### 3. Train the models

* **NumPy model**
  Run the Jupyter notebook:

  ```bash
  jupyter notebook train_model.ipynb
  ```

* **TensorFlow model**
  Run the notebook:

  ```bash
  jupyter notebook model_tf.ipynb
  ```

---

## 📊 Results

| Implementation       | Training Accuracy | Test Accuracy |
| -------------------- | ----------------- | ------------- |
| NumPy (from scratch) | \~98.8%           | \~96.9%       |
| TensorFlow (Keras)   | \~99.7%           | \~99.0%       |

---

## 🔮 Future Work

* Implement **Convolutional Neural Networks (CNNs)** from scratch
* Add **data augmentation** for TensorFlow model
* Explore **regularization techniques** like BatchNorm & Dropout in NumPy version

---

## 🙌 Acknowledgements

* [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)
