# 🧮 NumPy Digit Recognizer

A deep learning model built **from scratch using only NumPy** to classify handwritten digits (MNIST dataset).
This project avoids high-level libraries like TensorFlow or PyTorch, focusing instead on implementing all the core neural network components manually — forward propagation, backward propagation, optimization (Adam), and regularization.

---

## 🚀 Features

* Fully connected neural network (multi-layer perceptron) built from scratch
* Activation functions: ReLU & Softmax
* Cross-entropy loss function
* Mini-batch gradient descent with **Adam optimizer**
* Supports **L2 regularization** for better generalization
* Achieves **\~97% accuracy** on test data (similar to Kaggle Digit Recognizer benchmark)

---

## 📂 Project Structure

```
├── train.csv              # Training dataset (Kaggle MNIST)
├── train_model.ipynb      # Jupyter notebook for training
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
pip install numpy pandas matplotlib
```

### 3. Train the model

Run the Jupyter notebook or script:

```bash
python model.ipynb
```


## 📊 Results

* **Training accuracy**: \~98.8%
* **Test accuracy**: \~96.9%
* Competitive with traditional deep learning frameworks for MNIST classification.

---

## 🔮 Future Work

* Implement **Convolutional Neural Networks (CNNs)** from scratch for better performance

---

## 🙌 Acknowledgements

* [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)

---


