"""
CEDD-Stack Training Module
===========================
Training utilities with callbacks matching paper configuration.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


def get_callbacks(
    model_name: str,
    monitor: str = 'val_loss',
    save_best_only: bool = True
):
    """
    Get training callbacks matching paper Table 5 configuration.
    
    Callbacks:
        1. EarlyStopping (patience=7, monitor=val_loss)
        2. ReduceLROnPlateau (factor=0.5, patience=4)
        3. ModelCheckpoint (save best model)
    
    Args:
        model_name: Name of model for saving
        monitor: Metric to monitor (default: 'val_loss')
        save_best_only: Only save best weights
    
    Returns:
        List of Keras callbacks
    """
    # Ensure model weights directory exists
    weights_dir = config.MODELS_DIR / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        # Early Stopping - patience=7 from paper
        EarlyStopping(
            monitor=monitor,
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning Rate Scheduler - factor=0.5, patience=4 from paper
        ReduceLROnPlateau(
            monitor=monitor,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        
        # Model Checkpoint - save best weights
        ModelCheckpoint(
            filepath=str(weights_dir / f"{model_name}_best.h5"),
            monitor=monitor,
            save_best_only=save_best_only,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
    ]
    
    return callbacks


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model_name="model",
    batch_size=config.BATCH_SIZE,
    epochs=config.EPOCHS,
    use_mixup=True,
    mixup_generator=None,
    verbose=1
):
    """
    Train a single model with proper configuration.
    
    Training configuration from paper Table 5:
        - Batch size: 32
        - Epochs: 50 (with early stopping)
        - Optimizer: Adam (lr=5e-5)
        - Loss: Categorical crossentropy
        - Data augmentation: Mixup (α=0.2)
    
    Args:
        model: Compiled Keras model
        X_train: Training data
        y_train: Training labels (one-hot)
        X_val: Validation data (optional, will use validation_split if None)
        y_val: Validation labels (optional)
        model_name: Name for saving model
        batch_size: Training batch size
        epochs: Maximum epochs
        use_mixup: Whether to use Mixup augmentation
        mixup_generator: Pre-configured MixupGenerator (required if use_mixup=True)
        verbose: Verbosity level
    
    Returns:
        Training history object
    """
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    
    # Get callbacks
    callbacks = get_callbacks(model_name)
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max epochs: {epochs}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Optimizer: {config.OPTIMIZER}")
    print(f"  - Loss function: {config.LOSS_FUNCTION}")
    print(f"  - Mixup augmentation: {use_mixup}")
    print(f"  - Training samples: {len(X_train)}")
    
    # Determine validation data
    validation_data = None
    validation_split = 0.0
    
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
        print(f"  - Validation samples: {len(X_val)}")
    else:
        validation_split = 0.2
        print(f"  - Validation split: {validation_split}")
    
    # Train with or without Mixup
    if use_mixup:
        if mixup_generator is None:
            raise ValueError("mixup_generator required when use_mixup=True")
        
        print(f"\nTraining with Mixup augmentation (α={config.MIXUP_ALPHA})...")
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // (batch_size * 2)  # Mixup uses 2x batch_size
        validation_steps = None
        
        if validation_data:
            validation_steps = len(X_val) // batch_size
        
        history = model.fit(
            mixup_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_data,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=verbose
        )
    else:
        print(f"\nTraining without augmentation...")
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
    
    # Training summary
    final_epoch = len(history.history['loss'])
    best_val_loss = min(history.history.get('val_loss', [float('inf')]))
    best_val_acc = max(history.history.get('val_accuracy', [0]))
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {model_name}")
    print(f"{'='*60}")
    print(f"  - Epochs trained: {final_epoch}/{epochs}")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print(f"  - Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  - Model saved: {config.MODELS_DIR / 'weights' / f'{model_name}_best.h5'}")
    print(f"{'='*60}\n")
    
    return history


def train_all_base_models(
    models_dict,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    mixup_generator=None,
    save_histories=True
):
    """
    Train all base models sequentially.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        mixup_generator: MixupGenerator instance
        save_histories: Whether to save training histories
    
    Returns:
        Dictionary of {model_name: history}
    """
    histories = {}
    
    print("\n" + "="*60)
    print(f"TRAINING ALL BASE MODELS ({len(models_dict)} models)")
    print("="*60)
    
    for idx, (model_name, model) in enumerate(models_dict.items(), 1):
        print(f"\n[{idx}/{len(models_dict)}] Training {model_name}...")
        
        try:
            history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_name=model_name,
                use_mixup=(mixup_generator is not None),
                mixup_generator=mixup_generator
            )
            histories[model_name] = history
            
        except Exception as e:
            print(f"⚠ Training failed for {model_name}: {e}")
            histories[model_name] = None
    
    # Summary
    successful = sum(1 for h in histories.values() if h is not None)
    print("\n" + "="*60)
    print(f"TRAINING SUMMARY")
    print("="*60)
    print(f"  - Successfully trained: {successful}/{len(models_dict)} models")
    print(f"  - Failed: {len(models_dict) - successful}")
    print("="*60 + "\n")
    
    # Save histories
    if save_histories:
        save_training_histories(histories)
    
    return histories


def save_training_histories(histories, filename="training_histories.csv"):
    """
    Save training histories to CSV for analysis.
    
    Args:
        histories: Dictionary of training histories
        filename: Output CSV filename
    """
    results_dir = config.RESULTS_DIR / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    for model_name, history in histories.items():
        if history is not None:
            final_epoch = len(history.history['loss'])
            data.append({
                'model': model_name,
                'epochs_trained': final_epoch,
                'final_loss': history.history['loss'][-1],
                'final_accuracy': history.history['accuracy'][-1],
                'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
                'best_val_accuracy': max(history.history.get('val_accuracy', [0]))
            })
    
    df = pd.DataFrame(data)
    output_path = results_dir / filename
    df.to_csv(output_path, index=False)
    print(f"✓ Training histories saved: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Training Module Test")
    print("="*60)
    
    # Test callbacks creation
    print("\n1. Testing callbacks creation...")
    callbacks = get_callbacks("test_model")
    print(f"   ✓ Created {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"      - {cb.__class__.__name__}")
    
    # Test with dummy data and model
    print("\n2. Testing training function (dry run)...")
    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input
    
    # Create simple test model
    test_model = Sequential([
        Input(shape=(224, 224, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    test_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"   ✓ Test model created")
    
    # Create tiny dummy dataset
    X_train_tiny = np.random.rand(100, 224, 224, 3).astype(np.float32)
    y_train_tiny = np.eye(5)[np.random.randint(0, 5, 100)]
    X_val_tiny = np.random.rand(20, 224, 224, 3).astype(np.float32)
    y_val_tiny = np.eye(5)[np.random.randint(0, 5, 20)]
    
    print(f"   ✓ Dummy data created")
    
    # Test training (1 epoch only)
    response = input("\n   Run test training (1 epoch)? (y/n): ")
    if response.lower() == 'y':
        history = train_model(
            model=test_model,
            X_train=X_train_tiny,
            y_train=y_train_tiny,
            X_val=X_val_tiny,
            y_val=y_val_tiny,
            model_name="test_model",
            epochs=1,
            use_mixup=False,
            verbose=1
        )
        print(f"\n   ✓ Training completed successfully")
    
    print("\n✓ All training module tests completed!")
    print("="*60)
