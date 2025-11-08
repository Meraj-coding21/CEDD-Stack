"""
CEDD-Stack Complete Training Pipeline
======================================
Train all 8 base models and create ensemble.

Usage:
    python scripts/train_all_models.py
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import config
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import create_mixup_generator_with_augmentation
from src.data.balancing import stratified_split_and_balance
from src.models.base_models import get_model, MODEL_BUILDERS
from src.training.trainer import train_model, save_training_histories
from src.evaluation.metrics import evaluate_model, compare_models
from tensorflow.keras.utils import to_categorical


def load_and_prepare_data(csv_path, images_dir):
    """
    Load APTOS 2019 dataset and prepare for training.
    
    Args:
        csv_path: Path to CSV file with labels
        images_dir: Path to images directory
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, balanced_train_df)
    """
    print("\n" + "="*60)
    print("LOADING AND PREPARING DATA")
    print("="*60)
    
    # Load CSV
    print(f"\n1. Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Loaded {len(df)} samples")
    
    # Stratified split and balance
    print("\n2. Performing stratified split and balancing...")
    original_train_df, test_df, balanced_train_df = stratified_split_and_balance(df)
    
    # Preprocess images
    print("\n3. Preprocessing images with Ben Graham method...")
    preprocessor = ImagePreprocessor()
    
    # Training images
    train_paths = [images_dir / f"{id_code}.png" for id_code in balanced_train_df['id_code']]
    X_train = preprocessor.preprocess_batch(train_paths, normalize=True, verbose=True)
    y_train = to_categorical(balanced_train_df['diagnosis'].values, num_classes=5)
    
    print(f"   ✓ Training data shape: {X_train.shape}")
    print(f"   ✓ Training labels shape: {y_train.shape}")
    
    # Test images
    test_paths = [images_dir / f"{id_code}.png" for id_code in test_df['id_code']]
    X_test = preprocessor.preprocess_batch(test_paths, normalize=True, verbose=True)
    y_test = to_categorical(test_df['diagnosis'].values, num_classes=5)
    
    print(f"   ✓ Test data shape: {X_test.shape}")
    print(f"   ✓ Test labels shape: {y_test.shape}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60 + "\n")
    
    return X_train, y_train, X_test, y_test, balanced_train_df


def train_all_base_models(X_train, y_train, X_test, y_test):
    """
    Train all 8 base CNN models.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    
    Returns:
        Tuple of (trained_models, histories, results)
    """
    print("\n" + "="*60)
    print("TRAINING ALL BASE MODELS")
    print("="*60)
    
    # Create Mixup generator
    print("\n1. Creating Mixup data generator...")
    mixup_gen = create_mixup_generator_with_augmentation(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        alpha=config.MIXUP_ALPHA,
        use_geometric_aug=True
    )
    print(f"   ✓ Mixup generator created (α={config.MIXUP_ALPHA})")
    
    # Train each model
    trained_models = {}
    histories = {}
    results = {}
    
    model_names = list(MODEL_BUILDERS.keys())
    
    for idx, model_name in enumerate(model_names, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(model_names)}] TRAINING {model_name}")
        print(f"{'='*60}")
        
        try:
            # Build model
            print(f"\nBuilding {model_name}...")
            model = get_model(model_name)
            print(f"   ✓ Model built ({model.count_params():,} parameters)")
            
            # Train model
            history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                model_name=model_name,
                batch_size=config.BATCH_SIZE,
                epochs=config.EPOCHS,
                use_mixup=True,
                mixup_generator=mixup_gen,
                verbose=1
            )
            
            # Evaluate model
            result = evaluate_model(model, X_test, y_test, model_name)
            
            # Store results
            trained_models[model_name] = model
            histories[model_name] = history
            results[model_name] = result
            
            print(f"\n✓ {model_name} training complete!")
            
        except Exception as e:
            print(f"\n⚠ Failed to train {model_name}: {e}")
            continue
    
    # Save training histories
    save_training_histories(histories)
    
    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison_path = config.RESULTS_DIR / "metrics" / "base_models_comparison.csv"
    compare_models(results, save_path=comparison_path)
    
    return trained_models, histories, results


def main():
    """Main training pipeline."""
    
    print("\n" + "="*80)
    print(" " * 20 + "CEDD-STACK TRAINING PIPELINE")
    print("="*80)
    
    # Setup
    print("\n1. Setting up environment...")
    config.create_directories()
    config.set_seeds()
    config.configure_gpu()
    
    # Data paths (UPDATE THESE WITH YOUR PATHS)
    csv_path = config.DATA_DIR / "raw" / "aptos2019" / "train.csv"
    images_dir = config.DATA_DIR / "raw" / "aptos2019" / "train_images"
    
    # Check if data exists
    if not csv_path.exists():
        print(f"\n⚠ ERROR: CSV file not found at {csv_path}")
        print("\nPlease update the paths in scripts/train_all_models.py:")
        print("  - csv_path: Path to train.csv")
        print("  - images_dir: Path to train_images folder")
        print("\nExample:")
        print("  csv_path = Path('/kaggle/input/aptos2019/train.csv')")
        print("  images_dir = Path('/kaggle/input/aptos2019/train_images')")
        return
    
    # Load and prepare data
    X_train, y_train, X_test, y_test, balanced_train_df = load_and_prepare_data(
        csv_path, images_dir
    )
    
    # Train all base models
    trained_models, histories, results = train_all_base_models(
        X_train, y_train, X_test, y_test
    )
    
    # Summary
    print("\n" + "="*80)
    print(" " * 25 + "TRAINING COMPLETE")
    print("="*80)
    print(f"\n✓ Successfully trained {len(trained_models)}/{len(MODEL_BUILDERS)} models")
    print(f"✓ Model weights saved in: {config.MODELS_DIR / 'weights'}")
    print(f"✓ Results saved in: {config.RESULTS_DIR / 'metrics'}")
    print(f"✓ Confusion matrices saved in: {config.RESULTS_DIR / 'confusion_matrices'}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Create ensemble: python scripts/create_ensemble.py")
    print("2. Generate LIME explanations using notebooks/")
    print("3. Review results in results/ directory")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
