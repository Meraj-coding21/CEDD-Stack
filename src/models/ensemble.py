"""
CEDD-Stack Ensemble Module
===========================
Implements three ensemble strategies: Majority Voting, BMA, and CatBoost Meta-Learner.

CEDD-Stack = CNN Ensemble with DenseNet and EfficientNet + CatBoost Stacking
"""

import numpy as np
from collections import Counter
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from configs.config import config


class EnsembleMethods:
    """
    Implementation of ensemble strategies from paper Section 3.4.
    
    Three ensemble approaches:
        1. Majority Voting (baseline)
        2. Bayesian Model Averaging (BMA)
        3. CatBoost Meta-Learner Stacking (CEDD-Stack)
    
    Attributes:
        models: List of trained base models
        model_names: List of model names
        ensemble_results: Dictionary storing results
    """
    
    def __init__(self, models, model_names):
        """
        Initialize ensemble methods.
        
        Args:
            models: List of compiled Keras models
            model_names: List of model name strings
        """
        self.models = models
        self.model_names = model_names
        self.ensemble_results = {}
        self.best_meta_learner = None
        self.best_predictions = None
        self.best_probabilities = None
    
    def majority_voting_ensemble(self, X_test, top_n=4):
        """
        Majority voting ensemble strategy.
        
        Each model votes for a class, and the majority vote wins.
        Simple but effective baseline.
        
        Args:
            X_test: Test data array
            top_n: Number of top models to use (default: 4)
        
        Returns:
            Array of ensemble predictions
        """
        predictions = []
        
        for model in self.models[:top_n]:
            pred = model.predict(X_test, verbose=0)
            pred_classes = np.argmax(pred, axis=1)
            predictions.append(pred_classes)
        
        predictions = np.array(predictions)
        ensemble_predictions = []
        
        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            majority_vote = Counter(votes).most_common(1)[0][0]
            ensemble_predictions.append(majority_vote)
        
        return np.array(ensemble_predictions)
    
    def bayesian_model_averaging(self, X_test, weights=None, top_n=4):
        """
        Bayesian Model Averaging ensemble strategy.
        
        Weighted average of probability distributions from each model.
        
        Args:
            X_test: Test data array
            weights: Optional model weights (default: uniform)
            top_n: Number of top models to use (default: 4)
        
        Returns:
            Tuple of (ensemble_predictions, ensemble_probabilities)
        """
        if weights is None:
            weights = np.ones(top_n) / top_n
        
        ensemble_probs = np.zeros((X_test.shape[0], config.NUM_CLASSES))
        
        for i, model in enumerate(self.models[:top_n]):
            pred_probs = model.predict(X_test, verbose=0)
            ensemble_probs += weights[i] * pred_probs
        
        ensemble_predictions = np.argmax(ensemble_probs, axis=1)
        return ensemble_predictions, ensemble_probs
    
    def extract_features_for_meta_learner(self, X_data, top_n=4):
        """
        Extract probability features from base models for meta-learner.
        
        Each base model outputs 5 class probabilities.
        For top_n=4 models: 4 × 5 = 20 features per sample.
        
        Args:
            X_data: Input data
            top_n: Number of top models to use
        
        Returns:
            Meta-features array of shape (n_samples, top_n × 5)
        """
        features = []
        
        for model in self.models[:top_n]:
            pred_probs = model.predict(X_data, verbose=0)
            features.append(pred_probs)
        
        meta_features = np.concatenate(features, axis=1)
        return meta_features
    
    def catboost_meta_learner(
        self, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        top_n=4
    ):
        """
        CatBoost meta-learner stacking ensemble - CEDD-Stack core algorithm.
        
        Implements the two-stage training process:
            Stage 1: Train base CNN models independently
            Stage 2: Train CatBoost meta-learner on base model predictions
        
        CatBoost configuration from paper Table 5:
            - Iterations: 1000
            - Learning rate: 0.1
            - Depth: 6
            - Loss: MultiClass
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot or integer)
            X_test: Test data
            y_test: Test labels (one-hot or integer)
            top_n: Number of top models to use (2, 3, or 4)
        
        Returns:
            Tuple of (meta_learner, predictions, probabilities, metrics)
        """
        print(f"\n{'='*60}")
        print(f"CEDD-STACK: CatBoost Meta-Learner (Top {top_n} Models)")
        print(f"{'='*60}")
        
        # Extract features from base models
        print(f"\n1. Extracting features from top {top_n} base models...")
        train_features = self.extract_features_for_meta_learner(X_train, top_n)
        test_features = self.extract_features_for_meta_learner(X_test, top_n)
        print(f"   ✓ Training features shape: {train_features.shape}")
        print(f"   ✓ Test features shape: {test_features.shape}")
        
        # Convert labels to integers if one-hot encoded
        if len(y_train.shape) > 1:
            y_train_labels = np.argmax(y_train, axis=1)
        else:
            y_train_labels = y_train
            
        if len(y_test.shape) > 1:
            y_test_labels = np.argmax(y_test, axis=1)
        else:
            y_test_labels = y_test
        
        # Train CatBoost meta-learner
        print(f"\n2. Training CatBoost meta-learner...")
        meta_learner = CatBoostClassifier(
            iterations=config.CATBOOST_ITERATIONS,
            learning_rate=config.CATBOOST_LEARNING_RATE,
            depth=config.CATBOOST_DEPTH,
            loss_function=config.CATBOOST_LOSS_FUNCTION,
            eval_metric=config.CATBOOST_EVAL_METRIC,
            random_seed=config.RANDOM_STATE,
            verbose=False
        )
        
        meta_learner.fit(train_features, y_train_labels)
        print(f"   ✓ CatBoost training complete")
        
        # Make predictions
        print(f"\n3. Generating predictions...")
        meta_predictions = meta_learner.predict(test_features)
        meta_probabilities = meta_learner.predict_proba(test_features)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_labels, meta_predictions)
        precision = precision_score(y_test_labels, meta_predictions, average='macro')
        recall = recall_score(y_test_labels, meta_predictions, average='macro')
        f1 = f1_score(y_test_labels, meta_predictions, average='macro')
        
        print(f"\n4. CEDD-Stack Performance (Top {top_n}):")
        print(f"   {'='*50}")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"   {'='*50}")
        
        # Save meta-learner
        model_path = config.MODELS_DIR / "ensemble" / f"cedd_stack_top{top_n}_meta_learner.cbm"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta_learner.save_model(str(model_path))
        print(f"\n✓ CEDD-Stack model saved: {model_path}")
        
        return meta_learner, meta_predictions, meta_probabilities, {
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1_score': f1
        }
    
    def evaluate_all_ensembles(self, X_train, y_train, X_test, y_test):
        """
        Evaluate all three ensemble methods across different top-n configurations.
        
        Tests ensemble with top 2, 3, and 4 models as reported in paper.
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        
        Returns:
            Dictionary with results for all ensemble configurations
        """
        results = {}
        
        for top_n in config.TOP_N_MODELS:
            print(f"\n{'='*60}")
            print(f"ENSEMBLE EVALUATION - TOP {top_n} MODELS")
            print(f"{'='*60}")
            
            results[f'top_{top_n}'] = {}
            
            # Get true labels
            if len(y_test.shape) > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
            
            # 1. Majority Voting
            print(f"\n1. Majority Voting (Top {top_n})...")
            mv_predictions = self.majority_voting_ensemble(X_test, top_n)
            mv_metrics = {
                'accuracy': accuracy_score(y_true, mv_predictions),
                'precision': precision_score(y_true, mv_predictions, average='macro'),
                'recall': recall_score(y_true, mv_predictions, average='macro'),
                'f1_score': f1_score(y_true, mv_predictions, average='macro')
            }
            results[f'top_{top_n}']['majority_voting'] = mv_metrics
            print(f"   Accuracy: {mv_metrics['accuracy']*100:.2f}%")
            
            # 2. Bayesian Model Averaging
            print(f"\n2. Bayesian Model Averaging (Top {top_n})...")
            bma_predictions, bma_probs = self.bayesian_model_averaging(X_test, top_n=top_n)
            bma_metrics = {
                'accuracy': accuracy_score(y_true, bma_predictions),
                'precision': precision_score(y_true, bma_predictions, average='macro'),
                'recall': recall_score(y_true, bma_predictions, average='macro'),
                'f1_score': f1_score(y_true, bma_predictions, average='macro')
            }
            results[f'top_{top_n}']['bma'] = bma_metrics
            print(f"   Accuracy: {bma_metrics['accuracy']*100:.2f}%")
            
            # 3. CEDD-Stack
            print(f"\n3. CEDD-Stack (Top {top_n})...")
            meta_learner, meta_pred, meta_probs, meta_metrics = self.catboost_meta_learner(
                X_train, y_train, X_test, y_test, top_n
            )
            results[f'top_{top_n}']['cedd_stack'] = meta_metrics
            
            # Store best model (Top 4 as reported in paper)
            if top_n == 4:
                self.best_meta_learner = meta_learner
                self.best_predictions = meta_pred
                self.best_probabilities = meta_probs
        
        self.ensemble_results = results
        return results


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("CEDD-Stack Ensemble Module Test")
    print("="*60)
    
    # Create dummy data and mock models
    print("\n1. Creating dummy test data...")
    n_train = 500
    n_test = 100
    
    X_train_dummy = np.random.rand(n_train, 224, 224, 3).astype(np.float32)
    y_train_dummy = np.eye(5)[np.random.randint(0, 5, n_train)]
    X_test_dummy = np.random.rand(n_test, 224, 224, 3).astype(np.float32)
    y_test_dummy = np.eye(5)[np.random.randint(0, 5, n_test)]
    
    print(f"   ✓ Training data: {X_train_dummy.shape}")
    print(f"   ✓ Test data: {X_test_dummy.shape}")
    
    # Create mock models (simple predictors)
    print("\n2. Creating mock models...")
    
    class MockModel:
        """Mock model for testing"""
        def __init__(self, model_id):
            self.model_id = model_id
        
        def predict(self, X, verbose=0):
            # Return random probabilities
            np.random.seed(self.model_id)
            probs = np.random.dirichlet(np.ones(5), size=len(X))
            return probs.astype(np.float32)
    
    mock_models = [MockModel(i) for i in range(4)]
    model_names = ['Model1', 'Model2', 'Model3', 'Model4']
    print(f"   ✓ Created {len(mock_models)} mock models")
    
    # Test ensemble methods
    print("\n3. Testing Ensemble Methods...")
    ensemble = EnsembleMethods(mock_models, model_names)
    
    # Test majority voting
    print("\n   a) Testing Majority Voting...")
    mv_preds = ensemble.majority_voting_ensemble(X_test_dummy, top_n=4)
    print(f"      ✓ Predictions shape: {mv_preds.shape}")
    
    # Test BMA
    print("\n   b) Testing Bayesian Model Averaging...")
    bma_preds, bma_probs = ensemble.bayesian_model_averaging(X_test_dummy, top_n=4)
    print(f"      ✓ Predictions shape: {bma_preds.shape}")
    print(f"      ✓ Probabilities shape: {bma_probs.shape}")
    
    # Test CatBoost meta-learner
    print("\n   c) Testing CatBoost Meta-Learner...")
    meta_learner, meta_preds, meta_probs, metrics = ensemble.catboost_meta_learner(
        X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy, top_n=4
    )
    print(f"      ✓ Meta-learner trained successfully")
    
    print("\n✓ All ensemble tests passed!")
    print("="*60)
