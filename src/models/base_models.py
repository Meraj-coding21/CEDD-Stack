"""
CEDD-Stack Base Models Module
==============================
Implements all 8 CNN architectures with custom classifier head.

Base Models (from paper Table 4):
    1. EfficientNetV2M
    2. DenseNet169
    3. EfficientNetB5
    4. InceptionV3
    5. ResNet50
    6. DenseNet121
    7. Xception
    8. InceptionResNetV2
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.applications import (
    EfficientNetV2M,
    DenseNet169,
    EfficientNetB5,
    InceptionV3,
    ResNet50,
    DenseNet121,
    Xception,
    InceptionResNetV2
)
from tensorflow.keras.optimizers import Adam
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


def build_model(
    base_model,
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    num_classes=config.NUM_CLASSES,
    dropout_rate=config.DROPOUT_RATE,
    dense_units=config.DENSE_UNITS,
    learning_rate=config.LEARNING_RATE,
    trainable=True
):
    """
    Build model with custom classifier head - YOUR EXACT ARCHITECTURE.
    
    Implements architecture from paper Section 3.3.1 and Figure 3:
        Base CNN → Global Average Pooling → Dropout(0.5) → 
        Dense(1024, ReLU) → Dropout(0.5) → Dense(5, Softmax)
    
    Args:
        base_model: Pre-trained base model
        input_shape: Input image shape (default: 224×224×3)
        num_classes: Number of output classes (default: 5)
        dropout_rate: Dropout rate (default: 0.5)
        dense_units: Dense layer units (default: 1024)
        learning_rate: Adam learning rate (default: 5e-5)
        trainable: Whether to train base model layers (default: True)
    
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Set all layers trainable (fine-tuning)
    for layer in model.layers:
        layer.trainable = trainable
    
    # Compile with categorical crossentropy (as in your code)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    return model


# ========== Model Builder Functions ==========

def build_efficientnetv2m(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build EfficientNetV2M model (Top performer in paper).
    
    Performance from Table 6:
        - Accuracy: 96.86%
        - Precision: 93.77%
        - Recall: 96.70%
        - F1-Score: 94.99%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled EfficientNetV2M model
    """
    base_model = EfficientNetV2M(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_densenet169(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build DenseNet169 model (Second best performer).
    
    Performance from Table 6:
        - Accuracy: 96.18%
        - Precision: 92.88%
        - Recall: 96.13%
        - F1-Score: 94.30%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled DenseNet169 model
    """
    base_model = DenseNet169(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_efficientnetb5(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build EfficientNetB5 model (Third best performer).
    
    Performance from Table 6:
        - Accuracy: 95.95%
        - Precision: 92.60%
        - Recall: 95.91%
        - F1-Score: 94.07%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled EfficientNetB5 model
    """
    base_model = EfficientNetB5(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_inceptionv3(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build InceptionV3 model.
    
    Performance from Table 6:
        - Accuracy: 95.27%
        - Precision: 91.63%
        - Recall: 95.23%
        - F1-Score: 93.25%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled InceptionV3 model
    """
    base_model = InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_resnet50(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build ResNet50 model.
    
    Performance from Table 6:
        - Accuracy: 94.36%
        - Precision: 90.18%
        - Recall: 94.33%
        - F1-Score: 92.06%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled ResNet50 model
    """
    base_model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_densenet121(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build DenseNet121 model.
    
    Performance from Table 6:
        - Accuracy: 93.67%
        - Precision: 89.33%
        - Recall: 93.64%
        - F1-Score: 91.31%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled DenseNet121 model
    """
    base_model = DenseNet121(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_xception(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build Xception model.
    
    Performance from Table 6:
        - Accuracy: 93.45%
        - Precision: 89.05%
        - Recall: 93.42%
        - F1-Score: 91.06%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled Xception model
    """
    base_model = Xception(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


def build_inceptionresnetv2(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    weights='imagenet',
    trainable=True
):
    """
    Build InceptionResNetV2 model.
    
    Performance from Table 6:
        - Accuracy: 92.99%
        - Precision: 88.38%
        - Recall: 92.96%
        - F1-Score: 90.48%
    
    Args:
        input_shape: Input shape
        weights: Pre-trained weights source
        trainable: Whether to train base model
    
    Returns:
        Compiled InceptionResNetV2 model
    """
    base_model = InceptionResNetV2(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    model = build_model(base_model, input_shape=input_shape, trainable=trainable)
    return model


# ========== Model Factory ==========

MODEL_BUILDERS = {
    'EfficientNetV2M': build_efficientnetv2m,
    'DenseNet169': build_densenet169,
    'EfficientNetB5': build_efficientnetb5,
    'InceptionV3': build_inceptionv3,
    'ResNet50': build_resnet50,
    'DenseNet121': build_densenet121,
    'Xception': build_xception,
    'InceptionResNetV2': build_inceptionresnetv2
}


def get_model(model_name: str, **kwargs):
    """
    Get model by name using factory pattern.
    
    Args:
        model_name: Name of model (e.g., 'EfficientNetV2M')
        **kwargs: Additional arguments for model builder
    
    Returns:
        Compiled Keras model
    
    Raises:
        ValueError: If model_name not recognized
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_BUILDERS.keys())}"
        )
    
    builder = MODEL_BUILDERS[model_name]
    return builder(**kwargs)


def build_all_models():
    """
    Build all 8 base models for ensemble.
    
    Returns:
        Dictionary mapping model names to compiled models
    """
    models_dict = {}
    
    print("="*60)
    print("BUILDING ALL BASE MODELS")
    print("="*60)
    
    for model_name in config.BASE_MODELS:
        print(f"\nBuilding {model_name}...")
        try:
            model = get_model(model_name)
            models_dict[model_name] = model
            print(f"   ✓ {model_name} built successfully")
            print(f"   - Total params: {model.count_params():,}")
        except Exception as e:
            print(f"   ⚠ Failed to build {model_name}: {e}")
    
    print("\n" + "="*60)
    print(f"✓ Successfully built {len(models_dict)}/{len(config.BASE_MODELS)} models")
    print("="*60 + "\n")
    
    return models_dict


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Base Models Module Test")
    print("="*60)
    
    # Test building single model
    print("\n1. Testing single model (EfficientNetV2M)...")
    model = build_efficientnetv2m()
    print(f"   ✓ Model built successfully")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Total parameters: {model.count_params():,}")
    print(f"   - Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    
    # Display model architecture
    print("\n2. Model architecture summary:")
    print("-" * 60)
    model.summary()
    
    # Test model factory
    print("\n3. Testing model factory...")
    try:
        model_from_factory = get_model('DenseNet169')
        print(f"   ✓ Factory successfully created DenseNet169")
    except Exception as e:
        print(f"   ⚠ Factory test failed: {e}")
    
    # Test building all models (will take time)
    print("\n4. Testing build_all_models() function...")
    response = input("   Build all 8 models? (y/n): ")
    if response.lower() == 'y':
        all_models = build_all_models()
        print(f"\n   Summary:")
        for name, model in all_models.items():
            print(f"   - {name}: {model.count_params():,} parameters")
    
    print("\n✓ All base model tests completed!")
    print("="*60)
