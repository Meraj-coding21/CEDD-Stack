# CEDD-Stack: CNN Ensemble with CatBoost Stacking for Diabetic Retinopathy Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"CEDD-Stack: A CNN Ensemble with CatBoost Stacking for Accurate and Explainable Diabetic Retinopathy Detection"**

---

## 🎯 Key Achievements

- **98.91% accuracy** on 5-class DR severity grading (APTOS 2019 dataset)
- **Dual-layer explainability** using LIME and Grad-CAM for clinical interpretability
- **Robust validation** with k-fold cross-validation (k=2,4,6)
- **Outperforms** best individual model (EfficientNetV2M) by 2.05%

---

## 📊 Performance Summary

| Method | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **CEDD-Stack (Ours)** | **98.91%** | **98.17%** | **98.71%** | **98.43%** |
| EfficientNetV2M | 96.86% | 93.77% | 96.70% | 94.99% |
| DenseNet169 | 96.18% | 92.88% | 96.13% | 94.30% |
| EfficientNetB5 | 95.95% | 92.60% | 95.91% | 94.07% |

---

## 📋 Table of Contents

1. [Installation](#-installation)
2. [Quick Start](#-quick-start)
3. [Dataset Preparation](#-dataset-preparation)
4. [Training](#-training)
5. [Inference](#-inference)
6. [Evaluation](#-evaluation)
7. [Explainability](#-explainability)
8. [Results](#-results)
9. [Citation](#-citation)

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 11.2+ (for GPU support, recommended)
- 16GB RAM minimum (32GB recommended)
- 10GB disk space (for code + models + data)

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR-USERNAME/CEDD-Stack.git
cd CEDD-Stack
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv cedd-env
source cedd-env/bin/activate  # On Windows: cedd-env\Scripts\activate

# OR using conda
conda create -n cedd-env python=3.10
conda activate cedd-env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "from configs.config import config; print('✓ CEDD-Stack installed successfully')"
```

---

## ⚡ Quick Start

### Run a Complete Demo (5 minutes)

```bash
# 1. Preprocess sample image
python -c "from src.data.preprocessing import preprocess_image_with_ben; \
           img = preprocess_image_with_ben('path/to/fundus_image.png'); \
           print('✓ Preprocessed:', img.shape)"

# 2. Run inference on single image
python scripts/inference.py --image path/to/fundus_image.png --visualize

# 3. View Grad-CAM explanations (in browser)
jupyter notebook notebooks/02_gradcam_efficientnetv2m.ipynb
```

---

## 📂 Dataset Preparation

### Download APTOS 2019 Dataset

1. **Register on Kaggle**: https://www.kaggle.com/
2. **Download dataset**: https://www.kaggle.com/c/aptos2019-blindness-detection/data
3. **Extract files**:

```
# Recommended structure
data/
└── raw/
    └── aptos2019/
        ├── train.csv
        └── train_images/
            ├── 000c1434d8d7.png
            ├── 001639a390f0.png
            └── ... (3662 images)
```

### Alternative: Use Your Own Dataset

Format your CSV with columns: `id_code,diagnosis`

```csv
id_code,diagnosis
image_001,0
image_002,2
image_003,4
```

**Class labels:**
- 0: No DR (No Diabetic Retinopathy)
- 1: Mild NPDR
- 2: Moderate NPDR
- 3: Severe NPDR
- 4: PDR (Proliferative DR)

---

## 🎓 Training

### Train All Base Models (8-12 hours on GPU)

```bash
# Update dataset paths in scripts/train_all_models.py:
# Line 54: csv_path = Path('data/raw/aptos2019/train.csv')
# Line 55: images_dir = Path('data/raw/aptos2019/train_images')

python scripts/train_all_models.py
```

**Output:**
- Trained model weights → `models/weights/`
- Training histories → `results/metrics/training_histories.csv`
- Confusion matrices → `results/confusion_matrices/`

### Train Single Model (1-2 hours)

```bash
# Edit scripts/train_all_models.py to train only one model
# Or use Python API:
python -c "
from src.models.base_models import build_efficientnetv2m
from src.training.trainer import train_model
# ... (see documentation)
"
```

### Create CEDD-Stack Ensemble (30 minutes)

```bash
python scripts/create_ensemble.py
```

**Output:**
- CatBoost meta-learner → `models/ensemble/cedd_stack_top4_meta_learner.cbm`
- Ensemble metrics → `results/metrics/ensemble_comparison.csv`

---

## 🔮 Inference

### Single Image Prediction

```bash
python scripts/inference.py --image path/to/fundus_image.png
```

**Output:**
```
PREDICTION RESULT
==========================================================
Image: fundus_image.png
Predicted Class: 2 - Moderate NPDR
Confidence: 94.37%

Class Probabilities:
  0. No DR                : 0.0012 (0.12%)
  1. Mild NPDR            : 0.0345 (3.45%)
  2. Moderate NPDR        : 0.9437 (94.37%)
  3. Severe NPDR          : 0.0186 (1.86%)
  4. PDR                  : 0.0020 (0.20%)
==========================================================
```

### Batch Prediction

```bash
python scripts/inference.py --image_dir path/to/test_images/ --output predictions.csv
```

### With Visualization

```bash
python scripts/inference.py --image path/to/fundus_image.png --visualize
```

---

## 📊 Evaluation

### Evaluate Trained Models

```bash
# Evaluation is automatic during training
# Results saved to:
# - results/metrics/base_models_comparison.csv
# - results/metrics/ensemble_comparison.csv
```

### Cross-Validation

```bash
# 4-fold CV (primary validation)
python scripts/cross_validation.py --model EfficientNetV2M --k 4

# Complete CV (k=2,4,6)
python scripts/cross_validation.py --all_models --k 2,4,6
```

---

## 🔍 Explainability

### Grad-CAM Visualizations

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks:
# - notebooks/02_gradcam_efficientnetv2m.ipynb
# - notebooks/03_gradcam_densenet169.ipynb
# - notebooks/04_gradcam_efficientnetb5.ipynb
# - notebooks/05_gradcam_inceptionv3.ipynb
```

### LIME Explanations

```python
from src.explainability.lime_explainer import LIMEExplainer
from tensorflow.keras.models import load_model
from src.data.preprocessing import preprocess_image_with_ben

# Load model
model = load_model('models/weights/EfficientNetV2M_best.h5')

# Initialize explainer
explainer = LIMEExplainer(model)

# Generate explanation
image = preprocess_image_with_ben('path/to/fundus_image.png')
explanation, fig, pred = explainer.explain_and_visualize(image)
```

---

## 📈 Results

### Pre-computed Results Included

All experimental results from the paper are provided:

```
results/
├── gradcam_visualizations/    # 25 Grad-CAM images (Figures 12-15)
├── confusion_matrices/         # Confusion matrices for all models
├── training_curves/            # Sample training/validation curves
└── metrics/                    # Performance CSV files
```

### Reproducing Results

```bash
# Complete pipeline
python scripts/train_all_models.py      # → Training metrics
python scripts/create_ensemble.py       # → Ensemble results
python scripts/cross_validation.py --all_models --k 4  # → CV results
```

---

## 📁 Repository Structure

```
CEDD-Stack/
├── configs/
│   └── config.py                      # Centralized configuration
├── src/
│   ├── data/
│   │   ├── preprocessing.py           # Ben Graham preprocessing
│   │   ├── augmentation.py            # Mixup augmentation
│   │   └── balancing.py               # Class balancing
│   ├── models/
│   │   ├── base_models.py             # 8 CNN architectures
│   │   └── ensemble.py                # Ensemble methods
│   ├── training/
│   │   └── trainer.py                 # Training pipeline
│   ├── evaluation/
│   │   └── metrics.py                 # Evaluation metrics
│   └── explainability/
│       └── lime_explainer.py          # LIME implementation
├── scripts/
│   ├── train_all_models.py            # Train all models
│   ├── create_ensemble.py             # Create ensemble
│   ├── inference.py                   # Predict on new images
│   └── cross_validation.py            # K-fold CV
├── notebooks/                          # Jupyter notebooks
├── models/
│   ├── weights/                        # Pre-trained weights (.h5)
│   └── ensemble/                       # Ensemble models (.cbm)
├── results/                            # Experimental results
└── requirements.txt                    # Dependencies
```

---

## 🔧 Configuration

All hyperparameters are centralized in `configs/config.py`:

```python
# Key parameters (from paper Table 5)
LEARNING_RATE = 5e-5           # Adam learning rate
BATCH_SIZE = 32                # Training batch size
EPOCHS = 50                    # Maximum epochs
EARLY_STOPPING_PATIENCE = 7    # Early stopping patience
DROPOUT_RATE = 0.5             # Dropout rate
MIXUP_ALPHA = 0.2              # Mixup alpha parameter

# Modify these for experimentation
```

---

## 🐛 Troubleshooting

### GPU Not Detected

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Should list your GPU(s)
```

**Fix:** Install CUDA-compatible TensorFlow:
```bash
pip install tensorflow[and-cuda]
```

### Out of Memory

Reduce batch size in `configs/config.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Missing Dependencies

```bash
pip install --upgrade -r requirements.txt
```

---

## 📖 Citation

If you use this code or pre-trained models, please cite our paper:

```bibtex
@article{meraj2025cedd,
  title={CEDD-Stack: A CNN Ensemble with CatBoost Stacking for Accurate and Explainable Diabetic Retinopathy Detection},
  author={Meraj, Mehedi Hasan and Tajbid, Mashfiquzzaman and Mojumdar, Mayen Uddin and Banshal, Sumit Kumar},
  journal={[Journal Name]},
  year={2025},
  doi={[DOI]}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: APTOS 2019 Blindness Detection (Kaggle)
- **Preprocessing**: Ben Graham's circle cropping method
- **Frameworks**: TensorFlow, Keras, CatBoost, scikit-learn

---

## 📧 Contact

- **Authors**: Mehedi Hasan Meraj, Mashfiquzzaman Tajbid, Mayen Uddin Mojumdar, Sumit Kumar Banshal
- **Repository**: https://github.com/YOUR-USERNAME/CEDD-Stack
- **Issues**: https://github.com/YOUR-USERNAME/CEDD-Stack/issues

For questions about the paper or code, please open an issue on GitHub.

---

## 📊 Additional Resources

- **Paper PDF**: [Link to published paper]
- **Supplementary Materials**: [Link if available]
- **Demo Video**: [Link if available]
- **Presentation**: [Link if available]

---

**⭐ If you find this work useful, please consider starring the repository!**
