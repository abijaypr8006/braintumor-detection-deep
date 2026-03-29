# 🧠 Brain Tumor Detection using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A deep learning-based system for detecting brain tumors from MRI scan images using Convolutional Neural Networks (CNN). This project automates the classification of brain MRI scans into **tumor** and **non-tumor** categories, supporting faster and more accurate medical diagnosis.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🔍 Overview

Brain tumors are one of the most critical medical conditions requiring early and accurate detection. Manual analysis of MRI scans by radiologists is time-consuming and subject to human error. This project leverages deep learning to:

- Automatically classify brain MRI images as **tumor** or **no tumor**
- Achieve high accuracy with minimal preprocessing
- Provide a reproducible, end-to-end deep learning pipeline

---

## 📂 Dataset

The dataset used in this project consists of brain MRI images organized into two classes:

| Class       | Description                        |
|-------------|------------------------------------|
| `yes`       | MRI scans with brain tumor present |
| `no`        | MRI scans with no tumor            |

- **Source**: [Kaggle Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- Images are in `.jpg` / `.png` format
- Data augmentation is applied to handle class imbalance

> **Note**: Download the dataset and place it in the `data/` directory before running.

---

## 🧪 Model Architecture

The model is built using a **Convolutional Neural Network (CNN)** with the following key components:

- Convolutional layers with ReLU activation
- MaxPooling layers for spatial downsampling
- Dropout layers for regularization
- Fully connected Dense layers
- Sigmoid output for binary classification

> Optionally, transfer learning with **VGG16 / ResNet50** can be enabled for improved performance.

---

## 🗂 Project Structure

```
braintumor-detection-deep/
│
├── data/
│   ├── yes/                  # MRI images with tumor
│   └── no/                   # MRI images without tumor
│
├── models/
│   └── best_model.h5         # Saved trained model weights
│
├── notebooks/
│   └── brain_tumor_detection.ipynb   # Jupyter notebook with full pipeline
│
├── src/
│   ├── preprocess.py         # Data loading and augmentation
│   ├── model.py              # Model definition
│   ├── train.py              # Training script
│   └── predict.py            # Inference script
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/abijaypr8006/braintumor-detection-deep.git
cd braintumor-detection-deep
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset** and place it inside the `data/` directory.

---

## 🚀 Usage

### Training the Model

```bash
python src/train.py
```

### Running Predictions on a New MRI Image

```bash
python src/predict.py --image path/to/mri_image.jpg
```

### Running the Jupyter Notebook

```bash
jupyter notebook notebooks/brain_tumor_detection.ipynb
```

---

## 📊 Results

| Metric        | Value   |
|---------------|---------|
| Training Accuracy | ~95%  |
| Validation Accuracy | ~92% |
| Test Accuracy | ~90%  |
| F1 Score      | ~0.91   |

> Results may vary depending on the dataset split and hyperparameters used.

**Sample Predictions:**

| MRI Scan | Prediction |
|----------|------------|
| ✅ Tumor Present | `Tumor Detected` |
| ❌ No Tumor | `No Tumor Detected` |

---

## 🛠 Technologies Used

- **Python** — Core programming language
- **TensorFlow / Keras** — Deep learning framework
- **NumPy & Pandas** — Data manipulation
- **OpenCV** — Image preprocessing
- **Matplotlib & Seaborn** — Visualization
- **Scikit-learn** — Model evaluation metrics
- **Jupyter Notebook** — Experimentation and exploration

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Abijay PR**  
GitHub: [@abijaypr8006](https://github.com/abijaypr8006)

---

> ⚠️ **Disclaimer**: This project is intended for educational and research purposes only. It is **not** a substitute for professional medical diagnosis.