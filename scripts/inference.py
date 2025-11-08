"""
CEDD-Stack Inference Script
============================
Predict diabetic retinopathy severity on new fundus images.

Usage:
    # Single image
    python scripts/inference.py --image path/to/image.png
    
    # Batch inference
    python scripts/inference.py --image_dir path/to/images/ --output results.csv
    
    # With visualization
    python scripts/inference.py --image path/to/image.png --visualize
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import config
from src.data.preprocessing import preprocess_image_with_ben


class CEDDStackPredictor:
    """
    CEDD-Stack predictor for diabetic retinopathy classification.
    
    Attributes:
        models: List of loaded base models
        meta_learner: CatBoost meta-learner
        class_names: DR severity class names
    """
    
    def __init__(self, top_n=4):
        """
        Initialize predictor with trained models.
        
        Args:
            top_n: Number of top models to use (2, 3, or 4)
        """
        self.top_n = top_n
        self.class_names = config.CLASS_NAMES
        self.models = []
        self.model_names = []
        self.meta_learner = None
        
        self._load_models()
    
    def _load_models(self):
        """Load trained base models and meta-learner."""
        print(f"\nLoading CEDD-Stack (Top {self.top_n} models)...")
        
        weights_dir = config.MODELS_DIR / "weights"
        
        # Load top models based on paper performance ranking
        top_model_order = [
            "EfficientNetV2M",
            "DenseNet169", 
            "EfficientNetB5",
            "InceptionV3",
            "ResNet50",
            "DenseNet121",
            "Xception",
            "InceptionResNetV2"
        ]
        
        loaded_count = 0
        for model_name in top_model_order:
            if loaded_count >= self.top_n:
                break
                
            weight_path = weights_dir / f"{model_name}_best.h5"
            if weight_path.exists():
                try:
                    model = load_model(str(weight_path))
                    self.models.append(model)
                    self.model_names.append(model_name)
                    print(f"  ✓ Loaded {model_name}")
                    loaded_count += 1
                except Exception as e:
                    print(f"  ⚠ Failed to load {model_name}: {e}")
        
        if len(self.models) < self.top_n:
            raise ValueError(f"Could not load {self.top_n} models. Only {len(self.models)} available.")
        
        # Load meta-learner
        meta_path = config.MODELS_DIR / "ensemble" / f"cedd_stack_top{self.top_n}_meta_learner.cbm"
        if meta_path.exists():
            self.meta_learner = CatBoostClassifier()
            self.meta_learner.load_model(str(meta_path))
            print(f"  ✓ Loaded CatBoost meta-learner")
        else:
            print(f"  ⚠ Meta-learner not found. Run scripts/create_ensemble.py first.")
            print(f"  → Will use Bayesian Model Averaging instead.")
        
        print(f"\n✓ CEDD-Stack ready with {len(self.models)} models\n")
    
    def predict_single(self, image_path, return_probabilities=False):
        """
        Predict DR severity for a single image.
        
        Args:
            image_path: Path to fundus image
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Predicted class (and probabilities if requested)
        """
        # Preprocess image
        image = preprocess_image_with_ben(str(image_path))
        image = np.expand_dims(image, axis=0)
        
        # Get predictions from base models
        base_predictions = []
        for model in self.models:
            pred = model.predict(image, verbose=0)[0]
            base_predictions.append(pred)
        
        # Combine features for meta-learner
        meta_features = np.concatenate(base_predictions).reshape(1, -1)
        
        # Final prediction
        if self.meta_learner is not None:
            # Use CatBoost meta-learner (CEDD-Stack)
            predicted_class = self.meta_learner.predict(meta_features)[0]
            probabilities = self.meta_learner.predict_proba(meta_features)[0]
        else:
            # Fallback to Bayesian Model Averaging
            avg_probs = np.mean(base_predictions, axis=0)
            predicted_class = np.argmax(avg_probs)
            probabilities = avg_probs
        
        if return_probabilities:
            return predicted_class, probabilities
        return predicted_class
    
    def predict_batch(self, image_paths, verbose=True):
        """
        Predict DR severity for multiple images.
        
        Args:
            image_paths: List of image paths
            verbose: Print progress
        
        Returns:
            List of predicted classes
        """
        predictions = []
        
        if verbose:
            print(f"\nProcessing {len(image_paths)} images...")
        
        for idx, path in enumerate(image_paths):
            try:
                pred = self.predict_single(path)
                predictions.append(pred)
                
                if verbose and (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{len(image_paths)}")
            except Exception as e:
                print(f"  ⚠ Failed to process {path}: {e}")
                predictions.append(-1)  # Error indicator
        
        if verbose:
            print(f"✓ Completed {len(image_paths)} predictions\n")
        
        return predictions
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with probabilities.
        
        Args:
            image_path: Path to fundus image
            save_path: Path to save visualization (optional)
        """
        # Get prediction and probabilities
        predicted_class, probabilities = self.predict_single(
            image_path, return_probabilities=True
        )
        
        # Load original image
        image = preprocess_image_with_ben(str(image_path))
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Image
        axes[0].imshow(image)
        axes[0].set_title(
            f'Prediction: {self.class_names[predicted_class]}\n'
            f'Confidence: {probabilities[predicted_class]:.2%}',
            fontsize=14, fontweight='bold'
        )
        axes[0].axis('off')
        
        # Probability bar chart
        colors = ['green' if i == predicted_class else 'gray' 
                  for i in range(len(self.class_names))]
        bars = axes[1].barh(self.class_names, probabilities, color=colors)
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            axes[1].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        plt.show()
        
        return predicted_class, probabilities


def main():
    """Main inference pipeline."""
    parser = argparse.ArgumentParser(
        description='CEDD-Stack Inference for Diabetic Retinopathy Detection'
    )
    parser.add_argument('--image', type=str, help='Path to single fundus image')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file for batch predictions')
    parser.add_argument('--top_n', type=int, default=4, choices=[2, 3, 4],
                       help='Number of top models to use (default: 4)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize prediction for single image')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = CEDDStackPredictor(top_n=args.top_n)
    
    # Single image inference
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            return
        
        print(f"\nPredicting DR severity for: {image_path.name}")
        predicted_class, probabilities = predictor.predict_single(
            image_path, return_probabilities=True
        )
        
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULT")
        print(f"{'='*60}")
        print(f"Image: {image_path.name}")
        print(f"Predicted Class: {predicted_class} - {config.CLASS_NAMES[predicted_class]}")
        print(f"Confidence: {probabilities[predicted_class]:.2%}")
        print(f"\nClass Probabilities:")
        for i, (class_name, prob) in enumerate(zip(config.CLASS_NAMES, probabilities)):
            print(f"  {i}. {class_name:20s}: {prob:.4f} ({prob*100:.2f}%)")
        print(f"{'='*60}\n")
        
        # Visualize if requested
        if args.visualize:
            predictor.visualize_prediction(image_path)
    
    # Batch inference
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Directory not found: {image_dir}")
            return
        
        # Get all image files
        image_extensions = ['.png', '.jpg', '.jpeg']
        image_paths = [p for p in image_dir.iterdir() 
                      if p.suffix.lower() in image_extensions]
        
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        
        print(f"\nFound {len(image_paths)} images in {image_dir}")
        
        # Predict
        predictions = predictor.predict_batch(image_paths)
        
        # Create results DataFrame
        results = []
        for path, pred in zip(image_paths, predictions):
            if pred >= 0:
                results.append({
                    'image_name': path.name,
                    'predicted_class': pred,
                    'predicted_label': config.CLASS_NAMES[pred]
                })
            else:
                results.append({
                    'image_name': path.name,
                    'predicted_class': -1,
                    'predicted_label': 'ERROR'
                })
        
        df = pd.DataFrame(results)
        
        # Save results
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"✓ Results saved to: {output_path}")
        
        # Summary
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(df['predicted_label'].value_counts())
        print(f"{'='*60}\n")
    
    else:
        print("Error: Provide either --image or --image_dir")
        parser.print_help()


if __name__ == "__main__":
    main()
