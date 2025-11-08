"""
CEDD-Stack Configuration Module
================================
Centralized configuration for reproducibility.

Based on: CEDD-Stack: A CNN Ensemble with CatBoost Stacking for 
Accurate and Explainable Diabetic Retinopathy Detection
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for CEDD-Stack framework"""
    
    # ========== Project Paths ==========
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # ========== Dataset Configuration ==========
    DATASET_NAME = "APTOS 2019 Blindness Detection"
    NUM_CLASSES = 5
    CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    
    # ========== Preprocessing Parameters ==========
    IMG_SIZE = 224
    BEN_GRAHAM_SIGMA = 40
    CROP_THRESHOLD = 7  # τ in paper
    
    # ========== Data Split Configuration ==========
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # ========== Class Balancing ==========
    TARGET_SAMPLES_PER_CLASS = 1444
    
    # ========== Augmentation Parameters ==========
    MIXUP_ALPHA = 0.2
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.1, 0.5)
    HORIZONTAL_FLIP = True
    VERTICAL_FLIP = True
    
    # ========== Model Architecture ==========
    BASE_MODELS = [
        "EfficientNetV2M",
        "DenseNet169",
        "EfficientNetB5",
        "InceptionV3",
        "ResNet50",
        "DenseNet121",
        "Xception",
        "InceptionResNetV2"
    ]
    
    # Custom classifier head
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 1024
    DENSE_ACTIVATION = "relu"
    OUTPUT_ACTIVATION = "softmax"
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 5e-5
    OPTIMIZER = "adam"
    LOSS_FUNCTION = "categorical_crossentropy"
    
    # ========== Cross-Validation ==========
    CV_FOLDS = [2, 4, 6]
    
    # ========== Callbacks Configuration ==========
    EARLY_STOPPING_PATIENCE = 7
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 4
    
    # ========== Ensemble Configuration ==========
    TOP_N_MODELS = [2, 3, 4]
    
    # CatBoost Meta-Learner
    CATBOOST_ITERATIONS = 1000
    CATBOOST_LEARNING_RATE = 0.1
    CATBOOST_DEPTH = 6
    CATBOOST_LOSS_FUNCTION = "MultiClass"
    CATBOOST_EVAL_METRIC = "Accuracy"
    
    # ========== LIME Configuration ==========
    LIME_NUM_SAMPLES = 1000
    LIME_KERNEL_SIZE = 4
    LIME_MAX_DIST = 200
    LIME_RATIO = 0.2
    
    # ========== Reproducibility ==========
    SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary project directories"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.DATA_DIR / "splits",
            cls.MODELS_DIR / "weights",
            cls.MODELS_DIR / "ensemble",
            cls.RESULTS_DIR / "figures",
            cls.RESULTS_DIR / "confusion_matrices",
            cls.RESULTS_DIR / "metrics",
            Path(cls.PROJECT_ROOT) / "logs"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("✓ Project directories created")
    
    @classmethod
    def set_seeds(cls):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        import tensorflow as tf
        
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        tf.random.set_seed(cls.SEED)
        os.environ['PYTHONHASHSEED'] = str(cls.SEED)
        print("✓ Random seeds set for reproducibility")
    
    @classmethod
    def configure_gpu(cls):
        """Configure GPU settings"""
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✓ GPU configured: {len(gpus)} device(s) available")
            except RuntimeError as e:
                print(f"⚠ GPU configuration error: {e}")
        else:
            print("⚠ No GPU detected, using CPU")


# Global config instance
config = Config()
