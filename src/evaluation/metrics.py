"""
CEDD-Stack Evaluation & Metrics Module
=======================================
Comprehensive evaluation metrics and confusion matrix generation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculate all evaluation metrics used in paper.
    
    Metrics from paper Table 6:
        - Accuracy
        - Precision (macro-averaged)
        - Recall (macro-averaged)
        - F1-Score (macro-averaged)
    
    Args:
        y_true: True labels (integer or one-hot)
        y_pred: Predicted labels (integer or one-hot)
        average: Averaging method ('macro', 'weighted', 'micro')
    
    Returns:
        Dictionary with all metrics
    """
    # Convert one-hot to integer labels if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics


def calculate_per_class_metrics(y_true, y_pred):
    """
    Calculate metrics for each class individually.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        DataFrame with per-class metrics
    """
    # Convert one-hot to integer if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    # Calculate per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create DataFrame
    data = []
    for class_id in range(config.NUM_CLASSES):
        data.append({
            'class': class_id,
            'class_name': config.CLASS_NAMES[class_id],
            'precision': precision_per_class[class_id],
            'recall': recall_per_class[class_id],
            'f1_score': f1_per_class[class_id]
        })
    
    df = pd.DataFrame(data)
    return df


def generate_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix array
    """
    # Convert one-hot to integer if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(
    cm,
    model_name="Model",
    normalize=False,
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        cm: Confusion matrix array
        model_name: Model name for title
        normalize: Whether to normalize values
        save_path: Path to save figure (optional)
        figsize: Figure size tuple
    
    Returns:
        Matplotlib figure object
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = f'Normalized Confusion Matrix - {model_name}'
    else:
        fmt = 'd'
        title = f'Confusion Matrix - {model_name}'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved: {save_path}")
    
    return fig


def print_classification_report(y_true, y_pred, model_name="Model"):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Model name for header
    """
    # Convert one-hot to integer if needed
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT - {model_name}")
    print(f"{'='*60}\n")
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    print(report)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Complete evaluation of a single model.
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels (one-hot)
        model_name: Model name
    
    Returns:
        Dictionary with metrics and predictions
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*60}")
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, average='macro')
    
    # Print results
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  - Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  - Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    # Generate confusion matrix
    cm = generate_confusion_matrix(y_true, y_pred)
    
    # Save confusion matrix plot
    cm_path = config.RESULTS_DIR / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, model_name=model_name, save_path=cm_path)
    plt.close()
    
    # Per-class metrics
    per_class_df = calculate_per_class_metrics(y_true, y_pred)
    print(f"\nPer-Class Metrics:")
    print(per_class_df.to_string(index=False))
    
    # Classification report
    print_classification_report(y_true, y_pred, model_name)
    
    print(f"{'='*60}\n")
    
    return {
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_df
    }


def compare_models(results_dict, save_path=None):
    """
    Compare multiple models in a summary table.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
        save_path: Path to save comparison CSV
    
    Returns:
        DataFrame with model comparison
    """
    data = []
    for model_name, result in results_dict.items():
        metrics = result.get('metrics', result)
        data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Sort by accuracy
    df = df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Model comparison saved: {save_path}")
    
    return df


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Evaluation & Metrics Module Test")
    print("="*60)
    
    # Create dummy predictions
    print("\n1. Creating dummy test data...")
    n_samples = 200
    y_true_dummy = np.random.randint(0, 5, n_samples)
    y_pred_dummy = np.random.randint(0, 5, n_samples)
    
    # Make some predictions correct for realistic metrics
    correct_indices = np.random.choice(n_samples, size=int(0.7*n_samples), replace=False)
    y_pred_dummy[correct_indices] = y_true_dummy[correct_indices]
    
    print(f"   ✓ Created {n_samples} dummy predictions")
    
    # Test metrics calculation
    print("\n2. Testing metrics calculation...")
    metrics = calculate_metrics(y_true_dummy, y_pred_dummy)
    print(f"   ✓ Metrics calculated:")
    for metric, value in metrics.items():
        print(f"      - {metric}: {value:.4f}")
    
    # Test per-class metrics
    print("\n3. Testing per-class metrics...")
    per_class_df = calculate_per_class_metrics(y_true_dummy, y_pred_dummy)
    print(per_class_df)
    
    # Test confusion matrix
    print("\n4. Testing confusion matrix generation...")
    cm = generate_confusion_matrix(y_true_dummy, y_pred_dummy)
    print(f"   ✓ Confusion matrix shape: {cm.shape}")
    print(f"\n{cm}")
    
    # Test plotting
    print("\n5. Testing confusion matrix plotting...")
    cm_path = config.RESULTS_DIR / "confusion_matrices" / "test_confusion_matrix.png"
    fig = plot_confusion_matrix(cm, model_name="Test Model", save_path=cm_path)
    plt.close()
    
    # Test classification report
    print("\n6. Testing classification report...")
    print_classification_report(y_true_dummy, y_pred_dummy, "Test Model")
    
    # Test model comparison
    print("\n7. Testing model comparison...")
    dummy_results = {
        'Model A': {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.94, 'f1_score': 0.93},
        'Model B': {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.91, 'f1_score': 0.90},
        'Model C': {'accuracy': 0.97, 'precision': 0.96, 'recall': 0.96, 'f1_score': 0.96}
    }
    comparison_df = compare_models(dummy_results)
    
    print("\n✓ All evaluation tests completed!")
    print("="*60)
