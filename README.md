# CEDD-Stack: CNN Ensemble with CatBoost Stacking for Diabetic Retinopathy Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"CEDD-Stack: A CNN Ensemble with CatBoost Stacking for Accurate and Explainable Diabetic Retinopathy Detection"**

---

## 🎯 Key Achievements

- **98.91% accuracy** on 5-class DR severity grading
- **Dual-layer XAI** using LIME and Grad-CAM for clinical interpretability
- **Rigorous validation** with multi-fold cross-validation
- **Outperforms** single models by 2.05%

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **CEDD-Stack (Ours)** | **98.91%** | **98.17%** | **98.71%** | **98.43%** |
| EfficientNetV2M | 96.86% | 93.77% | 96.70% | 94.99% |
| DenseNet169 | 96.18% | 92.88% | 96.13% | 94.30% |
| EfficientNetB5 | 95.95% | 92.60% | 95.91% | 94.07% |

---

## 🚀 Quick Start

### Installation

