"""
CEDD-Stack Preprocessing Module
================================
Implements Ben Graham preprocessing for fundus images.

Reference:
    Graham, B. (2015). Kaggle diabetic retinopathy detection 
    competition report. University of Warwick.
"""

import cv2
import numpy as np
import math
from pathlib import Path
from typing import Union, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


class ImagePreprocessor:
    """
    Preprocessing pipeline for fundus images.
    
    Pipeline (from paper Section 3.2):
        1. Load RGB image
        2. Crop black borders using grayscale thresholding (τ=7)
        3. Resize to 224×224
        4. Apply Ben Graham contrast enhancement (σ=40)
    
    Attributes:
        img_size (int): Target image size (default: 224)
        sigma (int): Gaussian blur sigma (default: 40)
        threshold (int): Grayscale threshold τ (default: 7)
    """
    
    def __init__(
        self,
        img_size: int = config.IMG_SIZE,
        sigma: int = config.BEN_GRAHAM_SIGMA,
        threshold: int = config.CROP_THRESHOLD
    ):
        self.img_size = img_size
        self.sigma = sigma
        self.threshold = threshold
    
    @staticmethod
    def get_pad_width(im: np.ndarray, new_shape: int, is_rgb: bool = True):
        """
        Calculate padding widths for centering image.
        
        Args:
            im: Input image array
            new_shape: Target dimension (square)
            is_rgb: Whether image is RGB (True) or grayscale (False)
        
        Returns:
            Tuple of padding widths for np.pad
        """
        pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
        t, b = math.floor(pad_diff[0] / 2), math.ceil(pad_diff[0] / 2)
        l, r = math.floor(pad_diff[1] / 2), math.ceil(pad_diff[1] / 2)
        
        if is_rgb:
            pad_width = ((t, b), (l, r), (0, 0))
        else:
            pad_width = ((t, b), (l, r))
        
        return pad_width
    
    def crop_image_from_gray(self, img: np.ndarray, tol: Optional[int] = None) -> np.ndarray:
        """
        Crop black borders from fundus images using grayscale thresholding.
        
        Implements Equation 2 from paper:
            M(x,y) = 1 if I_gray(x,y) > τ, else 0
        
        Args:
            img: Input image (RGB or grayscale)
            tol: Tolerance threshold τ (default: 7 from paper)
        
        Returns:
            Cropped image with black borders removed
        """
        if tol is None:
            tol = self.threshold
        
        if img.ndim == 2:
            # Grayscale image
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        
        elif img.ndim == 3:
            # RGB image - convert to grayscale for masking
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            
            check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            
            if check_shape == 0:
                # Image too dark - return original
                return img
            else:
                # Crop each channel independently
                img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            
            return img
        
        else:
            raise ValueError(f"Invalid image dimensions: {img.ndim}")
    
    def preprocess_image_with_ben(
        self, 
        path: Union[str, Path], 
        sigmaX: Optional[int] = None
    ) -> np.ndarray:
        """
        Complete Ben Graham preprocessing pipeline.
        
        Implements Equation 4 from paper:
            I_enhanced = α·I + β·GaussianBlur(I,σ) + γ
            where α=4, β=-4, γ=128, σ=40
        
        Args:
            path: Path to input fundus image
            sigmaX: Gaussian blur sigma (default: 40)
        
        Returns:
            Preprocessed image array of shape (224, 224, 3)
        
        Raises:
            FileNotFoundError: If image path doesn't exist
            ValueError: If image cannot be loaded
        """
        if sigmaX is None:
            sigmaX = self.sigma
        
        # Load image
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 1: Crop black borders (Eq. 2)
        image = self.crop_image_from_gray(image, tol=self.threshold)
        
        # Step 2: Resize to target dimensions
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Step 3: Ben Graham enhancement (Eq. 4)
        image = cv2.addWeighted(
            image, 4,  # α = 4
            cv2.GaussianBlur(image, (0, 0), sigmaX), -4,  # β = -4
            128  # γ = 128
        )
        
        return image
    
    def preprocess_batch(
        self, 
        paths: list, 
        normalize: bool = True,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Preprocess multiple images in batch.
        
        Args:
            paths: List of image paths
            normalize: Whether to normalize to [0, 1] range
            verbose: Print progress
        
        Returns:
            Batch array of shape (n, 224, 224, 3)
        """
        processed_images = []
        failed_count = 0
        
        for idx, path in enumerate(paths):
            try:
                img = self.preprocess_image_with_ben(path)
                if normalize:
                    img = img.astype(np.float32) / 255.0
                processed_images.append(img)
                
                if verbose and (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(paths)} images")
                    
            except Exception as e:
                if verbose:
                    print(f"⚠ Failed to process {path}: {e}")
                # Create blank image as placeholder
                blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                processed_images.append(blank)
                failed_count += 1
        
        if verbose and failed_count > 0:
            print(f"⚠ {failed_count}/{len(paths)} images failed preprocessing")
        
        return np.array(processed_images)


def preprocess_image_with_ben(path: Union[str, Path], sigmaX: int = 40) -> np.ndarray:
    """
    Standalone function matching your original code exactly.
    
    This is a wrapper for easy import and usage in notebooks.
    
    Args:
        path: Path to fundus image
        sigmaX: Gaussian blur sigma (default: 40)
    
    Returns:
        Preprocessed image (224, 224, 3)
    """
    preprocessor = ImagePreprocessor(
        img_size=config.IMG_SIZE,
        sigma=sigmaX,
        threshold=config.CROP_THRESHOLD
    )
    return preprocessor.preprocess_image_with_ben(path, sigmaX)


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Standalone function matching your original code exactly.
    
    Args:
        img: RGB or grayscale image
        tol: Threshold value (default: 7)
    
    Returns:
        Cropped image
    """
    preprocessor = ImagePreprocessor()
    return preprocessor.crop_image_from_gray(img, tol)


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Preprocessing Module Test")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    print(f"✓ Preprocessor initialized")
    print(f"  - Image size: {preprocessor.img_size}×{preprocessor.img_size}")
    print(f"  - Ben Graham σ: {preprocessor.sigma}")
    print(f"  - Crop threshold τ: {preprocessor.threshold}")
    
    # Test with sample image (update path to your data)
    sample_image_path = config.DATA_DIR / "raw" / "aptos2019" / "train_images" / "sample.png"
    
    if sample_image_path.exists():
        print(f"\n✓ Testing preprocessing on: {sample_image_path.name}")
        preprocessed = preprocessor.preprocess_image_with_ben(sample_image_path)
        print(f"  - Output shape: {preprocessed.shape}")
        print(f"  - Output dtype: {preprocessed.dtype}")
        print(f"  - Value range: [{preprocessed.min()}, {preprocessed.max()}]")
        print(f"\n✓ Preprocessing test successful!")
    else:
        print(f"\n⚠ Sample image not found at: {sample_image_path}")
        print("  Update the path to test preprocessing on your data.")
