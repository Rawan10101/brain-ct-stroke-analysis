# Brain CT Stroke Classification

An AI-based medical image classification project that classifies brain CT scans into **Normal, Ischemic Stroke, and Hemorrhagic Stroke** classes using both traditional machine learning and deep learning approaches.

## Overview

The project evaluates four classification approaches:

* **Random Forest**
* **Support Vector Machine (SVM)**
* **EfficientNetB0**
* **ResNet50**

The pipeline includes dataset preprocessing, class balancing, image resizing, normalization, feature extraction, hyperparameter optimization, transfer learning, and model evaluation.

## Dataset

The project uses the **TEKNO21 Brain CT Stroke Dataset** containing 6,650 CT images:

| Class              | Images |
| ------------------ | -----: |
| Normal             |  4,427 |
| Ischemic Stroke    |  1,130 |
| Hemorrhagic Stroke |  1,093 |

To reduce class imbalance and computational cost, the Normal class was undersampled while retaining all stroke cases. Images were converted to grayscale and resized from **384×384 to 128×128**.

## Models & Techniques

### Traditional Machine Learning

* Random Forest with `RandomizedSearchCV`
* Support Vector Machine with `RandomizedSearchCV`
* PCA dimensionality reduction

### Deep Learning

* EfficientNetB0 with ImageNet transfer learning
* ResNet50 with ImageNet transfer learning
* Two-stage training with frozen backbones and fine-tuning
* Early stopping and model checkpointing

## Results

| Model          | Test Accuracy |   Macro F1 |
| -------------- | ------------: | ---------: |
| Random Forest  |        76.46% |          — |
| SVM + PCA      |        85.80% |          — |
| SVM            |        85.99% |          — |
| EfficientNetB0 |        84.63% |     84.61% |
| ResNet50       |    **85.21%** | **85.20%** |

The PCA-based SVM achieved **85.80% accuracy** while reducing training and hyperparameter optimization time to approximately **27 seconds**, compared with approximately 17 minutes without PCA.

## Project Structure

```text
Brain-CT-Stroke-Classification/
├── data/
├── notebooks/
├── models/
├── results/
└── README.md
```
