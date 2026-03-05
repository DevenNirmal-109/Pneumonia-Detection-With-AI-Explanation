# Medical Image Analysis – Pneumonia Detection from Chest X-rays

## Overview

This project focuses on detecting **pneumonia from chest X-ray images** using deep learning techniques. The model predicts the presence of pneumonia and also localizes the infected region using bounding box regression.

The goal of this project is to assist medical professionals by providing AI-based diagnostic support.

---

## Features

• Pneumonia detection from X-ray images  
• Bounding box localization of infected region  
• Deep learning CNN architecture  
• ROC curve evaluation and threshold tuning  
• Explainable AI pipeline (future integration with LLM)

---

## Dataset

The dataset used is the **RSNA Pneumonia Detection Challenge dataset**.

Dataset structure:

Train folder
- Normal
- Lung Opacity
- No Lung Opacity

Test folder
- X-ray images without labels

CSV files:
- stage_2_train_labels.csv
- stage_2_detailed_class_info.csv

Each **patientId corresponds to a single X-ray image**.

---

## Model Architecture

The model performs two tasks:

1. **Classification**
   - Predicts pneumonia presence

2. **Bounding Box Regression**
   - Predicts infected lung region

Loss Function:

Total Loss = Classification Loss + Bounding Box Loss

---

## Training Details

• Framework: PyTorch  
• Image size: 256x256  
• Optimizer: Adam  
• Loss: BCE + MSE  
• GPU training supported

Example training output:

Epoch [1/10] | Total Loss: 0.6291  
Epoch [2/10] | Total Loss: 0.5897  
Epoch [3/10] | Total Loss: 0.5677  

---

## Evaluation

The model is evaluated using:

• ROC Curve  
• AUC Score  
• Optimal Threshold Selection

---

## Project Structure

```
Medical-Image-Analysis-Pneumonia-Detection

model creation/
    model_creation.ipynb

model_creation/
    pneumonial_cnn_1.pth


README.md
requirements.txt
.gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/DevenNirmal-109/Pneumonia-Detection-With-AI-Explanation
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Future Work

• Integrate LLM for medical explanation  
• Deploy web application using Streamlit  
• Add Grad-CAM visualization  
• Real-time diagnosis interface

---

## Author

**Deven Nirmal**

B.Tech Computer Engineering  
Machine Learning & Backend Development Enthusiast

---

## License

This project is for educational and research purposes.