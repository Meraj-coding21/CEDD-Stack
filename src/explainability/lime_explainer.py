"""
CEDD-Stack LIME Explainability Module
======================================
Local Interpretable Model-agnostic Explanations for fundus images.

"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


class LIMEExplainer:
    """
    LIME explainer for diabetic retinopathy classification.
    
    Uses quickshift superpixel segmentation with parameters from paper:
        - kernel_size: 4
        - max_dist: 200
        - ratio: 0.2
    
    Attributes:
        model: Trained model to explain
        class_names: List of class names
    """
    
    def __init__(self, model, class_names=None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained Keras model
            class_names: List of class names (default: from config)
        """
        self.model = model
        self.class_names = class_names or config.CLASS_NAMES
        
        # Initialize LIME image explainer with quickshift segmentation
        self.explainer = lime_image.LimeImageExplainer()
        
    def predict_fn(self, images):
        """
        Prediction function wrapper for LIME.
        
        Args:
            images: Batch of images
        
        Returns:
            Prediction probabilities
        """
        return self.model.predict(images, verbose=0)
    
    def explain_instance(
        self,
        image,
        top_labels=5,
        num_samples=config.LIME_NUM_SAMPLES,
        hide_rest=False,
        positive_only=True
    ):
        """
        Generate LIME explanation for a single image.
        
        Uses quickshift superpixel segmentation from paper Section 3.5.2:
            - kernel_size: 4
            - max_dist: 200  
            - ratio: 0.2
        
        Args:
            image: Input image (normalized, 224x224x3)
            top_labels: Number of top classes to explain
            num_samples: Number of perturbed samples (default: 1000)
            hide_rest: Whether to hide non-important regions
            positive_only: Show only positive contributions
        
        Returns:
            LIME explanation object
        """
        # Define segmentation function with paper's parameters
        def segmentation_fn(img):
            return quickshift(
                img,
                kernel_size=config.LIME_KERNEL_SIZE,
                max_dist=config.LIME_MAX_DIST,
                ratio=config.LIME_RATIO
            )
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn
        )
        
        return explanation
    
    def visualize_explanation(
        self,
        image,
        explanation,
        predicted_class,
        true_class=None,
        num_features=5,
        positive_only=True,
        save_path=None
    ):
        """
        Visualize LIME explanation with highlighted regions.
        
        Args:
            image: Original image
            explanation: LIME explanation object
            predicted_class: Predicted class index
            true_class: True class index (optional)
            num_features: Number of features to highlight
            positive_only: Show only positive contributions
            save_path: Path to save visualization
        
        Returns:
            Matplotlib figure
        """
        # Get image and mask
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=positive_only,
            num_features=num_features,
            hide_rest=False
        )
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # LIME explanation (heatmap overlay)
        axes[1].imshow(mark_boundaries(temp, mask))
        title = f'LIME Explanation\nPredicted: {self.class_names[predicted_class]}'
        if true_class is not None:
            title += f'\nTrue: {self.class_names[true_class]}'
        axes[1].set_title(title, fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Highlighted important regions
        axes[2].imshow(mask, cmap='RdYlGn', alpha=0.8)
        axes[2].set_title('Important Regions\n(Green: Positive)', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ LIME visualization saved: {save_path}")
        
        return fig
    
    def explain_and_visualize(
        self,
        image,
        true_label=None,
        num_samples=config.LIME_NUM_SAMPLES,
        num_features=5,
        save_path=None
    ):
        """
        Complete LIME explanation pipeline: explain + visualize.
        
        Args:
            image: Input image
            true_label: True label (optional)
            num_samples: Number of LIME samples
            num_features: Number of features to show
            save_path: Save path for visualization
        
        Returns:
            Tuple of (explanation, figure, predicted_class)
        """
        # Get prediction
        pred_proba = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        predicted_class = np.argmax(pred_proba)
        
        print(f"\nGenerating LIME explanation...")
        print(f"  - Predicted class: {self.class_names[predicted_class]} ({pred_proba[predicted_class]:.4f})")
        if true_label is not None:
            print(f"  - True class: {self.class_names[true_label]}")
        print(f"  - Num samples: {num_samples}")
        print(f"  - Num features: {num_features}")
        
        # Generate explanation
        explanation = self.explain_instance(
            image,
            top_labels=5,
            num_samples=num_samples,
            positive_only=True
        )
        
        # Visualize
        fig = self.visualize_explanation(
            image,
            explanation,
            predicted_class,
            true_class=true_label,
            num_features=num_features,
            save_path=save_path
        )
        
        return explanation, fig, predicted_class
    
    def batch_explain(
        self,
        images,
        true_labels=None,
        num_samples=config.LIME_NUM_SAMPLES,
        save_dir=None
    ):
        """
        Generate LIME explanations for multiple images.
        
        Args:
            images: Array of images
            true_labels: Array of true labels (optional)
            num_samples: Number of LIME samples per image
            save_dir: Directory to save visualizations
        
        Returns:
            List of explanations
        """
        explanations = []
        
        print(f"\n{'='*60}")
        print(f"GENERATING LIME EXPLANATIONS FOR {len(images)} IMAGES")
        print(f"{'='*60}")
        
        for idx, image in enumerate(images):
            print(f"\n[{idx+1}/{len(images)}] Processing image {idx}...")
            
            true_label = true_labels[idx] if true_labels is not None else None
            
            if save_dir:
                save_path = Path(save_dir) / f"lime_explanation_{idx}.png"
            else:
                save_path = None
            
            explanation, fig, pred_class = self.explain_and_visualize(
                image,
                true_label=true_label,
                num_samples=num_samples,
                save_path=save_path
            )
            
            explanations.append({
                'explanation': explanation,
                'predicted_class': pred_class,
                'true_label': true_label
            })
            
            plt.close(fig)
        
        print(f"\n{'='*60}")
        print(f"✓ LIME explanations complete: {len(explanations)} images")
        print(f"{'='*60}\n")
        
        return explanations


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack LIME Explainability Module Test")
    print("="*60)
    
    # Note: Requires a trained model to test properly
    print("\n⚠ This module requires a trained model for full testing.")
    print("   Running basic initialization test...\n")
    
    # Test with mock model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Input
    
    # Create simple mock model
    mock_model = Sequential([
        Input(shape=(224, 224, 3)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')
    ])
    
    print("1. Testing LIME explainer initialization...")
    explainer = LIMEExplainer(mock_model)
    print(f"   ✓ LIME explainer initialized")
    print(f"   - Class names: {explainer.class_names}")
    
    # Test with dummy image
    print("\n2. Testing with dummy image...")
    dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
    
    response = input("\n   Generate LIME explanation (takes ~30s)? (y/n): ")
    if response.lower() == 'y':
        print("\n   Generating explanation...")
        try:
            explanation, fig, pred_class = explainer.explain_and_visualize(
                dummy_image,
                num_samples=100,  # Reduced for testing
                num_features=5,
                save_path=config.RESULTS_DIR / "test_lime_explanation.png"
            )
            print(f"\n   ✓ LIME explanation generated successfully")
            print(f"   - Predicted class: {pred_class}")
            plt.show()
        except Exception as e:
            print(f"\n   ⚠ Error during explanation: {e}")
    
    print("\n✓ LIME module test completed!")
    print("="*60)
    print("\nNote: For real usage, train a model first using:")
    print("  - src/training/trainer.py")
    print("  - Then load model weights for LIME explanations")
    print("="*60)
