"""
CEDD-Stack Data Augmentation Module
====================================
Implements Mixup augmentation for addressing class imbalance.

"""

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


class MixupGenerator:
    """
    Mixup data augmentation generator for training.
    
    Implements Equation 5-7 from paper:
        λ ~ Beta(α, α)
        x̃ = λ·xᵢ + (1-λ)·xⱼ
        ỹ = λ·yᵢ + (1-λ)·yⱼ
    
    where α=0.2 (from paper Table 5)
    
    Attributes:
        X_train: Training images array
        y_train: Training labels (one-hot encoded)
        batch_size: Batch size (default: 32)
        alpha: Mixup parameter α (default: 0.2)
        shuffle: Whether to shuffle data each epoch
        datagen: Optional ImageDataGenerator for additional augmentation
    """
    
    def __init__(
        self, 
        X_train, 
        y_train, 
        batch_size: int = config.BATCH_SIZE,
        alpha: float = config.MIXUP_ALPHA,
        shuffle: bool = True,
        datagen=None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        """
        Generator that yields mixup batches indefinitely.
        
        Yields:
            Tuple of (X_batch, y_batch) with mixup applied
        """
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        """
        Get shuffled indices for epoch.
        
        Returns:
            Array of shuffled indices
        """
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        """
        Generate one batch with mixup augmentation.
        
        Implements mixup equations:
            λ ~ Beta(α, α)
            x̃ = λ·x₁ + (1-λ)·x₂
            ỹ = λ·y₁ + (1-λ)·y₂
        
        Args:
            batch_ids: Indices for current batch (2×batch_size)
        
        Returns:
            Tuple of (X_mixed, y_mixed)
        """
        _, h, w, c = self.X_train.shape
        
        # Sample λ from Beta(α, α) for each sample in batch
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        # Split batch_ids into two halves
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        
        # Apply mixup: x̃ = λ·x₁ + (1-λ)·x₂
        X = X1 * X_l + X2 * (1 - X_l)

        # Apply additional augmentation if provided
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        # Handle labels (can be single or multiple outputs)
        if isinstance(self.y_train, list):
            y = []
            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            # Apply mixup: ỹ = λ·y₁ + (1-λ)·y₂
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y


def create_augmentation_generator(
    rotation_range: int = config.ROTATION_RANGE,
    zoom_range: tuple = config.ZOOM_RANGE,
    horizontal_flip: bool = config.HORIZONTAL_FLIP,
    vertical_flip: bool = config.VERTICAL_FLIP,
    fill_mode: str = config.FILL_MODE
):
    """
    Create ImageDataGenerator for geometric augmentation.
    
    Implements augmentation strategy from paper Section 3.2.2:
        - Random rotation
        - Random zoom (0.1-0.5)
        - Horizontal/vertical flips
    
    Args:
        rotation_range: Degree range for random rotations
        zoom_range: Range for random zoom (min, max)
        horizontal_flip: Enable horizontal flipping
        vertical_flip: Enable vertical flipping
        fill_mode: Points outside boundaries filled according to mode
    
    Returns:
        Configured ImageDataGenerator
    """
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        fill_mode=fill_mode
    )
    
    return datagen


def create_mixup_generator_with_augmentation(
    X_train,
    y_train,
    batch_size: int = config.BATCH_SIZE,
    alpha: float = config.MIXUP_ALPHA,
    use_geometric_aug: bool = True
):
    """
    Create complete augmentation pipeline: Geometric + Mixup.
    
    This combines both augmentation strategies from the paper:
        1. Geometric augmentation (rotation, zoom, flip)
        2. Mixup augmentation (α=0.2)
    
    Args:
        X_train: Training images
        y_train: Training labels (one-hot)
        batch_size: Batch size
        alpha: Mixup parameter
        use_geometric_aug: Whether to apply geometric augmentation
    
    Returns:
        MixupGenerator instance ready for model.fit()
    """
    datagen = None
    if use_geometric_aug:
        datagen = create_augmentation_generator()
    
    mixup_gen = MixupGenerator(
        X_train=X_train,
        y_train=y_train,
        batch_size=batch_size,
        alpha=alpha,
        shuffle=True,
        datagen=datagen
    )
    
    return mixup_gen


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Augmentation Module Test")
    print("="*60)
    
    # Create dummy data for testing
    print("\n1. Creating dummy data...")
    n_samples = 100
    X_dummy = np.random.rand(n_samples, 224, 224, 3).astype(np.float32)
    y_dummy = np.eye(5)[np.random.randint(0, 5, n_samples)]  # One-hot encoded
    print(f"   X_train shape: {X_dummy.shape}")
    print(f"   y_train shape: {y_dummy.shape}")
    
    # Test MixupGenerator
    print("\n2. Testing MixupGenerator...")
    mixup_gen = MixupGenerator(
        X_train=X_dummy,
        y_train=y_dummy,
        batch_size=32,
        alpha=0.2,
        shuffle=True
    )
    print(f"   ✓ MixupGenerator created")
    print(f"   - Batch size: 32")
    print(f"   - Mixup α: 0.2")
    print(f"   - Total samples: {mixup_gen.sample_num}")
    
    # Generate one batch
    print("\n3. Generating test batch...")
    generator = mixup_gen()
    X_batch, y_batch = next(generator)
    print(f"   ✓ Batch generated successfully")
    print(f"   - X_batch shape: {X_batch.shape}")
    print(f"   - y_batch shape: {y_batch.shape}")
    print(f"   - X_batch range: [{X_batch.min():.3f}, {X_batch.max():.3f}]")
    
    # Test with geometric augmentation
    print("\n4. Testing with geometric augmentation...")
    mixup_gen_aug = create_mixup_generator_with_augmentation(
        X_train=X_dummy,
        y_train=y_dummy,
        batch_size=32,
        alpha=0.2,
        use_geometric_aug=True
    )
    generator_aug = mixup_gen_aug()
    X_batch_aug, y_batch_aug = next(generator_aug)
    print(f"   ✓ Augmented batch generated")
    print(f"   - X_batch shape: {X_batch_aug.shape}")
    print(f"   - y_batch shape: {y_batch_aug.shape}")
    
    # Verify mixup properties
    print("\n5. Verifying mixup properties...")
    print(f"   - Sum of mixed labels (should be ~1.0): {y_batch[0].sum():.4f}")
    print(f"   - Sample label distribution: {y_batch[0]}")
    
    print("\n✓ All augmentation tests passed!")
    print("="*60)
