"""
CEDD-Stack Ensemble Creation Script
====================================
Create ensemble using trained base models with CatBoost meta-learner.

Usage:
    python scripts/create_ensemble.py

Prerequisites:
    - Trained base model weights in models/weights/
    - Test data available
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tensorflow.keras.models import load_model

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import config
from src.data.preprocessing import ImagePreprocessor
from src.data.balancing import stratified_split_and_balance
from src.models.ensemble import EnsembleMethods
from src.evaluation.metrics import compare_models
from tensorflow.keras.utils import to_categorical


def load_trained_models():
    """
    Load all trained base models from weights directory.
    
    Returns:
        Tuple of (models_list, model_names_list)
    """
    print("\n" + "="*60)
    print("LOADING TRAINED MODELS")
    print("="*60)
    
    weights_dir = config.MODELS_DIR / "weights"
    
    models = []
    model_names = []
    
    for model_name in config.BASE_MODELS:
        weight_path = weights_dir / f"{model_name}_best.h5"
        
        if weight_path.exists():
            print(f"\nLoading {model_name}...")
            try:
                model = load_model(str(weight_path))
                models.append(model)
                model_names.append(model_name)
                print(f"   ✓ {model_name} loaded successfully")
            except Exception as e:
                print(f"   ⚠ Failed to load {model_name}: {e}")
        else:
            print(f"   ⚠ Weights not found: {weight_path}")
    
    print("\n" + "="*60)
    print(f"✓ Loaded {len(models)}/{len(config.BASE_MODELS)} models")
    print("="*60 + "\n")
    
    if len(models) < 2:
        raise ValueError("Need at least 2 trained models for ensemble")
    
    return models, model_names


def load_test_data(csv_path, images_dir):
    """
    Load and preprocess test data.
    
    Args:
        csv_path: Path to CSV file
        images_dir: Path to images directory
    
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    
    # Load and split data
    print("\n1. Loading data...")
    df = pd.read_csv(csv_path)
    original_train_df, test_df, balanced_train_df = stratified_split_and_balance(df)
    
    # Preprocess
    print("\n2. Preprocessing images...")
    preprocessor = ImagePreprocessor()
    
    # For meta-learner training, use a subset of training data
    train_subset_size = min(5000, len(balanced_train_df))
    train_subset_df = balanced_train_df.sample(n=train_subset_size, random_state=config.RANDOM_STATE)
    
    train_paths = [images_dir / f"{id_code}.png" for id_code in train_subset_df['id_code']]
    X_train = preprocessor.preprocess_batch(train_paths, normalize=True, verbose=True)
    y_train = to_categorical(train_subset_df['diagnosis'].values, num_classes=5)
    
    test_paths = [images_dir / f"{id_code}.png" for id_code in test_df['id_code']]
    X_test = preprocessor.preprocess_batch(test_paths, normalize=True, verbose=True)
    y_test = to_categorical(test_df['diagnosis'].values, num_classes=5)
    
    print(f"\n   ✓ Training subset shape: {X_train.shape}")
    print(f"   ✓ Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test


def create_ensemble(models, model_names, X_train, y_train, X_test, y_test):
    """
    Create and evaluate all ensemble strategies.
    
    Args:
        models: List of trained models
        model_names: List of model names
        X_train: Training data for meta-learner
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
    
    Returns:
        Dictionary with ensemble results
    """
    print("\n" + "="*60)
    print("CREATING ENSEMBLE")
    print("="*60)
    
    # Initialize ensemble methods
    ensemble = EnsembleMethods(models, model_names)
    
    # Evaluate all ensemble strategies
    print("\nEvaluating all ensemble strategies...")
    print("  - Majority Voting")
    print("  - Bayesian Model Averaging")
    print("  - CEDD-Stack (CatBoost Meta-Learner)")
    print(f"\nTesting with top-{config.TOP_N_MODELS} model combinations...")
    
    results = ensemble.evaluate_all_ensembles(
        X_train, y_train, X_test, y_test
    )
    
    return results, ensemble


def save_ensemble_results(results):
    """
    Save ensemble comparison results to CSV.
    
    Args:
        results: Dictionary of ensemble results
    """
    print("\n" + "="*60)
    print("SAVING ENSEMBLE RESULTS")
    print("="*60)
    
    data = []
    
    for top_n_key, methods in results.items():
        top_n = int(top_n_key.split('_')[1])
        
        for method_name, metrics in methods.items():
            data.append({
                'Top_N_Models': top_n,
                'Method': method_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = config.RESULTS_DIR / "metrics" / "ensemble_comparison.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Ensemble results saved: {output_path}")
    
    # Display comparison
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def main():
    """Main ensemble creation pipeline."""
    
    print("\n" + "="*80)
    print(" " * 20 + "CEDD-STACK ENSEMBLE CREATION")
    print("="*80)
    
    # Setup
    print("\n1. Setting up environment...")
    config.set_seeds()
    config.configure_gpu()
    
    # Data paths (UPDATE THESE WITH YOUR PATHS)
    csv_path = config.DATA_DIR / "raw" / "aptos2019" / "train.csv"
    images_dir = config.DATA_DIR / "raw" / "aptos2019" / "train_images"
    
    # Check if data exists
    if not csv_path.exists():
        print(f"\n⚠ ERROR: CSV file not found at {csv_path}")
        print("\nPlease update the paths in scripts/create_ensemble.py")
        return
    
    # Load trained models
    try:
        models, model_names = load_trained_models()
    except Exception as e:
        print(f"\n⚠ ERROR: {e}")
        print("\nPlease train base models first:")
        print("  python scripts/train_all_models.py")
        return
    
    # Load test data
    X_train, y_train, X_test, y_test = load_test_data(csv_path, images_dir)
    
    # Create ensemble
    results, ensemble = create_ensemble(
        models, model_names, X_train, y_train, X_test, y_test
    )
    
    # Save results
    comparison_df = save_ensemble_results(results)
    
    # Summary
    print("\n" + "="*80)
    print(" " * 25 + "ENSEMBLE CREATION COMPLETE")
    print("="*80)
    print(f"\n✓ Evaluated {len(config.TOP_N_MODELS)} ensemble configurations")
    print(f"✓ CatBoost meta-learners saved in: {config.MODELS_DIR / 'ensemble'}")
    print(f"✓ Results saved in: {config.RESULTS_DIR / 'metrics'}")
    
    # Best ensemble
    best_config = comparison_df.loc[comparison_df['Accuracy'].astype(float).idxmax()]
    print("\n" + "="*80)
    print("BEST ENSEMBLE CONFIGURATION")
    print("="*80)
    print(f"Method: {best_config['Method']}")
    print(f"Top N Models: {best_config['Top_N_Models']}")
    print(f"Accuracy: {best_config['Accuracy']}")
    print(f"Precision: {best_config['Precision']}")
    print(f"Recall: {best_config['Recall']}")
    print(f"F1-Score: {best_config['F1-Score']}")
    print("="*80 + "\n")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Review ensemble results in results/metrics/")
    print("2. Generate LIME explanations using notebooks/")
    print("3. Your Grad-CAM notebook can now use the best ensemble")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
