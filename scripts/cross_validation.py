"""
CEDD-Stack Cross-Validation Script
===================================
Perform k-fold cross-validation (k=2, 4, 6) as reported in paper.

Usage:
    python scripts/cross_validation.py --model EfficientNetV2M --k 4
    python scripts/cross_validation.py --all_models --k 2,4,6
"""

import argparse
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import config
from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import create_mixup_generator_with_augmentation
from src.models.base_models import get_model, MODEL_BUILDERS
from src.training.trainer import train_model
from src.evaluation.metrics import calculate_metrics


def perform_cross_validation(
    model_name,
    X_data,
    y_data,
    k_folds=4,
    use_mixup=True
):
    """
    Perform k-fold cross-validation for a single model.
    
    Args:
        model_name: Name of model to train
        X_data: All training data
        y_data: All training labels (integer)
        k_folds: Number of folds
        use_mixup: Whether to use Mixup augmentation
    
    Returns:
        Dictionary with cross-validation results
    """
    print(f"\n{'='*60}")
    print(f"{k_folds}-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*60}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx}/{k_folds}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_val = X_data[train_idx], X_data[val_idx]
        y_train_int, y_val_int = y_data[train_idx], y_data[val_idx]
        
        # Convert to one-hot
        y_train = to_categorical(y_train_int, num_classes=config.NUM_CLASSES)
        y_val = to_categorical(y_val_int, num_classes=config.NUM_CLASSES)
        
        print(f"\nFold {fold_idx} data split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        
        # Build model
        print(f"\nBuilding {model_name}...")
        model = get_model(model_name)
        
        # Create Mixup generator if needed
        mixup_gen = None
        if use_mixup:
            mixup_gen = create_mixup_generator_with_augmentation(
                X_train, y_train,
                batch_size=config.BATCH_SIZE,
                alpha=config.MIXUP_ALPHA
            )
        
        # Train model
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_name=f"{model_name}_fold{fold_idx}",
            use_mixup=use_mixup,
            mixup_generator=mixup_gen,
            verbose=1
        )
        
        # Evaluate on validation set
        print(f"\nEvaluating fold {fold_idx}...")
        y_pred = model.predict(X_val, verbose=0)
        metrics = calculate_metrics(y_val_int, y_pred, average='macro')
        
        print(f"\nFold {fold_idx} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        
        fold_results.append({
            'fold': fold_idx,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    # Calculate mean and std across folds
    df_results = pd.DataFrame(fold_results)
    
    summary = {
        'model': model_name,
        'k_folds': k_folds,
        'mean_accuracy': df_results['accuracy'].mean(),
        'std_accuracy': df_results['accuracy'].std(),
        'mean_precision': df_results['precision'].mean(),
        'std_precision': df_results['precision'].std(),
        'mean_recall': df_results['recall'].mean(),
        'std_recall': df_results['recall'].std(),
        'mean_f1_score': df_results['f1_score'].mean(),
        'std_f1_score': df_results['f1_score'].std(),
        'fold_results': fold_results
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{k_folds}-FOLD CV SUMMARY: {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"Precision: {summary['mean_precision']:.4f} ± {summary['std_precision']:.4f}")
    print(f"Recall:    {summary['mean_recall']:.4f} ± {summary['std_recall']:.4f}")
    print(f"F1-Score:  {summary['mean_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
    print(f"{'='*60}\n")
    
    return summary


def load_data_for_cv(csv_path, images_dir):
    """
    Load and preprocess data for cross-validation.
    
    Args:
        csv_path: Path to CSV file
        images_dir: Path to images directory
    
    Returns:
        Tuple of (X_data, y_data)
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Preprocess images
    preprocessor = ImagePreprocessor()
    image_paths = [images_dir / f"{id_code}.png" for id_code in df['id_code']]
    
    X_data = preprocessor.preprocess_batch(image_paths, normalize=True, verbose=True)
    y_data = df['diagnosis'].values
    
    print(f"✓ Loaded {len(X_data)} samples")
    
    return X_data, y_data


def save_cv_results(all_results, output_path):
    """
    Save cross-validation results to CSV.
    
    Args:
        all_results: List of CV result dictionaries
        output_path: Path to save CSV
    """
    data = []
    for result in all_results:
        data.append({
            'model': result['model'],
            'k_folds': result['k_folds'],
            'mean_accuracy': f"{result['mean_accuracy']:.4f}",
            'std_accuracy': f"{result['std_accuracy']:.4f}",
            'mean_precision': f"{result['mean_precision']:.4f}",
            'std_precision': f"{result['std_precision']:.4f}",
            'mean_recall': f"{result['mean_recall']:.4f}",
            'std_recall': f"{result['std_recall']:.4f}",
            'mean_f1_score': f"{result['mean_f1_score']:.4f}",
            'std_f1_score': f"{result['std_f1_score']:.4f}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"\n✓ CV results saved: {output_path}")
    
    return df


def main():
    """Main cross-validation pipeline."""
    parser = argparse.ArgumentParser(
        description='CEDD-Stack Cross-Validation'
    )
    parser.add_argument('--model', type=str,
                       help='Model name to validate')
    parser.add_argument('--all_models', action='store_true',
                       help='Validate all base models')
    parser.add_argument('--k', type=str, default='4',
                       help='K-fold values (comma-separated, e.g., "2,4,6")')
    parser.add_argument('--csv_path', type=str,
                       default=str(config.DATA_DIR / "raw" / "aptos2019" / "train.csv"),
                       help='Path to CSV file')
    parser.add_argument('--images_dir', type=str,
                       default=str(config.DATA_DIR / "raw" / "aptos2019" / "train_images"),
                       help='Path to images directory')
    parser.add_argument('--use_mixup', action='store_true', default=True,
                       help='Use Mixup augmentation')
    
    args = parser.parse_args()
    
    # Setup
    config.set_seeds()
    config.configure_gpu()
    
    # Parse k values
    k_values = [int(k) for k in args.k.split(',')]
    
    print(f"\n{'='*80}")
    print(" " * 25 + "CEDD-STACK CROSS-VALIDATION")
    print(f"{'='*80}")
    print(f"\nK-fold values: {k_values}")
    print(f"Mixup augmentation: {args.use_mixup}")
    
    # Load data
    X_data, y_data = load_data_for_cv(args.csv_path, Path(args.images_dir))
    
    # Determine models to validate
    if args.all_models:
        models_to_validate = list(MODEL_BUILDERS.keys())
    elif args.model:
        if args.model not in MODEL_BUILDERS:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available models: {list(MODEL_BUILDERS.keys())}")
            return
        models_to_validate = [args.model]
    else:
        print("Error: Provide either --model or --all_models")
        return
    
    # Perform cross-validation
    all_results = []
    
    for model_name in models_to_validate:
        for k in k_values:
            try:
                result = perform_cross_validation(
                    model_name=model_name,
                    X_data=X_data,
                    y_data=y_data,
                    k_folds=k,
                    use_mixup=args.use_mixup
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n⚠ Failed CV for {model_name} (k={k}): {e}")
    
    # Save results
    if all_results:
        output_path = config.RESULTS_DIR / "metrics" / "cross_validation_results.csv"
        comparison_df = save_cv_results(all_results, output_path)
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(comparison_df.to_string(index=False))
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
