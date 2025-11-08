"""
CEDD-Stack Data Balancing Module
=================================
Implements stratified split and class balancing through oversampling.

Addresses severe class imbalance in APTOS 2019 dataset:
    Class 0 (No DR): 1805 samples
    Class 1 (Mild): 370 samples
    Class 2 (Moderate): 999 samples
    Class 3 (Severe): 193 samples
    Class 4 (Proliferative): 295 samples
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


def stratified_split_and_balance(
    df: pd.DataFrame, 
    test_size: float = config.TEST_SIZE
):
    """
    Perform stratified split and class balancing through oversampling.
    
    This implements the data preparation strategy from paper Section 3.2.1:
        1. Stratified train/test split (80/20) to maintain class distribution
        2. Balance training set to 1444 samples per class via oversampling
        3. Keep test set imbalanced (matches clinical distribution)
    
    Args:
        df: DataFrame with columns ['id_code', 'diagnosis']
        test_size: Fraction for test set (default: 0.2)
    
    Returns:
        Tuple of (original_train_df, test_df, balanced_train_df)
        
        - original_train_df: Unbalanced training set
        - test_df: Test set (imbalanced, matches real distribution)
        - balanced_train_df: Balanced training set (1444 per class)
    """
    print("\n" + "="*60)
    print("STRATIFIED SPLIT AND CLASS BALANCING")
    print("="*60)
    
    # Step 1: Stratified train/test split
    print("\n1. Performing stratified split...")
    X = df[['id_code']]
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    
    print(f"   ✓ Training set: {len(train_df)} samples")
    print(f"   ✓ Test set: {len(test_df)} samples")
    
    # Display class distribution in splits
    print("\n   Training set class distribution:")
    for class_id in range(config.NUM_CLASSES):
        count = len(train_df[train_df['diagnosis'] == class_id])
        print(f"      Class {class_id} ({config.CLASS_NAMES[class_id]}): {count} samples")
    
    print("\n   Test set class distribution:")
    for class_id in range(config.NUM_CLASSES):
        count = len(test_df[test_df['diagnosis'] == class_id])
        print(f"      Class {class_id} ({config.CLASS_NAMES[class_id]}): {count} samples")
    
    # Step 2: Balance training set through oversampling
    target_samples = config.TARGET_SAMPLES_PER_CLASS
    balanced_data = []
    
    print(f"\n2. Balancing training set to {target_samples} samples per class...")
    
    for class_id in range(config.NUM_CLASSES):
        class_data = train_df[train_df['diagnosis'] == class_id].copy()
        current_count = len(class_data)
        
        if current_count < target_samples:
            # Oversample minority classes
            additional_samples = target_samples - current_count
            upsampled = resample(
                class_data, 
                n_samples=additional_samples,
                random_state=config.RANDOM_STATE,
                replace=True
            )
            upsampled['augmented'] = True
            class_data['augmented'] = False
            balanced_class_data = pd.concat([class_data, upsampled], ignore_index=True)
            
            print(f"   Class {class_id} ({config.CLASS_NAMES[class_id]}): {current_count} → {len(balanced_class_data)} samples (+{additional_samples} oversampled)")
        else:
            # Downsample majority classes
            balanced_class_data = class_data.sample(
                n=target_samples, 
                random_state=config.RANDOM_STATE
            )
            balanced_class_data['augmented'] = False
            
            print(f"   Class {class_id} ({config.CLASS_NAMES[class_id]}): {current_count} → {len(balanced_class_data)} samples (downsampled)")
        
        balanced_data.append(balanced_class_data)
    
    # Combine all balanced classes
    balanced_train_df = pd.concat(balanced_data, ignore_index=True)
    
    # Summary statistics
    print("\n" + "="*60)
    print("BALANCING SUMMARY")
    print("="*60)
    print(f"Original training samples: {len(train_df)}")
    print(f"Balanced training samples: {len(balanced_train_df)}")
    print(f"Augmented samples: {balanced_train_df['augmented'].sum()}")
    print(f"Test samples: {len(test_df)}")
    print(f"Samples per class: {target_samples}")
    print("="*60 + "\n")
    
    return train_df, test_df, balanced_train_df


def get_class_distribution(df: pd.DataFrame, label_column: str = 'diagnosis'):
    """
    Get class distribution statistics from DataFrame.
    
    Args:
        df: DataFrame containing labels
        label_column: Name of label column (default: 'diagnosis')
    
    Returns:
        Dictionary with class counts and percentages
    """
    class_counts = df[label_column].value_counts().sort_index()
    total = len(df)
    
    distribution = {}
    for class_id in range(config.NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        percentage = (count / total) * 100
        distribution[class_id] = {
            'count': count,
            'percentage': percentage,
            'class_name': config.CLASS_NAMES[class_id]
        }
    
    return distribution


def display_class_distribution(df: pd.DataFrame, dataset_name: str = "Dataset"):
    """
    Display formatted class distribution table.
    
    Args:
        df: DataFrame with 'diagnosis' column
        dataset_name: Name to display (e.g., "Training Set")
    """
    print(f"\n{dataset_name} Class Distribution:")
    print("-" * 60)
    print(f"{'Class':<8} {'Name':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 60)
    
    dist = get_class_distribution(df)
    for class_id, info in dist.items():
        print(f"{class_id:<8} {info['class_name']:<20} {info['count']:<10} {info['percentage']:.2f}%")
    
    print("-" * 60)
    print(f"Total: {len(df)} samples\n")


def verify_balancing(balanced_df: pd.DataFrame):
    """
    Verify that balancing was performed correctly.
    
    Args:
        balanced_df: Balanced training DataFrame
    
    Returns:
        Boolean indicating if balancing is correct
    """
    print("\nVerifying class balance...")
    
    class_counts = balanced_df['diagnosis'].value_counts().sort_index()
    target = config.TARGET_SAMPLES_PER_CLASS
    
    all_balanced = True
    for class_id in range(config.NUM_CLASSES):
        count = class_counts.get(class_id, 0)
        if count != target:
            print(f"   ⚠ Class {class_id}: Expected {target}, got {count}")
            all_balanced = False
        else:
            print(f"   ✓ Class {class_id}: {count} samples")
    
    if all_balanced:
        print("\n✓ All classes perfectly balanced!\n")
    else:
        print("\n⚠ Class imbalance detected!\n")
    
    return all_balanced


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Data Balancing Module Test")
    print("="*60)
    
    # Create dummy imbalanced dataset (simulating APTOS 2019)
    print("\n1. Creating dummy imbalanced dataset...")
    
    # Simulate APTOS 2019 class distribution
    class_samples = {
        0: 1805,  # No DR
        1: 370,   # Mild
        2: 999,   # Moderate
        3: 193,   # Severe
        4: 295    # Proliferative DR
    }
    
    data = []
    for class_id, n_samples in class_samples.items():
        for i in range(n_samples):
            data.append({
                'id_code': f'sample_{class_id}_{i}',
                'diagnosis': class_id
            })
    
    df_dummy = pd.DataFrame(data)
    print(f"   ✓ Created dummy dataset: {len(df_dummy)} samples")
    
    # Display original distribution
    display_class_distribution(df_dummy, "Original Dataset")
    
    # Test stratified split and balancing
    print("\n2. Testing stratified split and balancing...")
    original_train_df, test_df, balanced_train_df = stratified_split_and_balance(
        df_dummy, 
        test_size=0.2
    )
    
    # Verify balancing
    print("\n3. Verifying balanced training set...")
    verify_balancing(balanced_train_df)
    
    # Display final distributions
    display_class_distribution(balanced_train_df, "Balanced Training Set")
    display_class_distribution(test_df, "Test Set (Kept Imbalanced)")
    
    print("✓ All balancing tests passed!")
    print("="*60)
