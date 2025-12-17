"""
Evaluation script for trained RLT models.
Loads model artifacts and computes metrics on test data.
"""

import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)


def load_model_artifacts(model_dir, model_name=None):
    """Load all model artifacts from directory."""
    # Get model name
    if model_name is None:
        latest_path = os.path.join(model_dir, "latest_model.txt")
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                model_name = f.read().strip()
        else:
            raise ValueError("No model_name provided and no latest_model.txt found")
    
    print(f"Loading model: {model_name}")
    
    # Load artifacts
    model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
    preprocessor_path = os.path.join(model_dir, f"{model_name}_preprocessor.joblib")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.joblib")
    test_data_path = os.path.join(model_dir, f"{model_name}_test_data.joblib")
    encoder_path = os.path.join(model_dir, f"{model_name}_label_encoder.joblib")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metadata = joblib.load(metadata_path)
    test_data = joblib.load(test_data_path)
    
    label_encoder = None
    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
    
    return model, preprocessor, metadata, test_data, label_encoder, model_name


def evaluate_classification(y_true, y_pred, y_proba=None, class_names=None):
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    target_names = class_names if class_names else None
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    metrics['classification_report'] = report
    
    return metrics


def evaluate_regression(y_true, y_pred):
    """Compute regression metrics."""
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    }
    return metrics


def print_classification_results(metrics, class_names=None):
    """Print classification results in a formatted way."""
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION METRICS")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>15.4f}")
    print(f"{'Precision (weighted)':<25} {metrics['precision_weighted']:>15.4f}")
    print(f"{'Recall (weighted)':<25} {metrics['recall_weighted']:>15.4f}")
    print(f"{'F1-Score (weighted)':<25} {metrics['f1_weighted']:>15.4f}")
    print(f"{'Precision (macro)':<25} {metrics['precision_macro']:>15.4f}")
    print(f"{'Recall (macro)':<25} {metrics['recall_macro']:>15.4f}")
    print(f"{'F1-Score (macro)':<25} {metrics['f1_macro']:>15.4f}")
    
    print("\n📋 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    if class_names:
        print(f"{'':>15}", end='')
        for name in class_names:
            print(f"{name:>12}", end='')
        print()
    for i, row in enumerate(cm):
        label = class_names[i] if class_names else f"Class {i}"
        print(f"{label:>15}", end='')
        for val in row:
            print(f"{val:>12}", end='')
        print()
    
    print("\n📈 Per-Class Metrics:")
    report = metrics['classification_report']
    print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 63)
    for key, values in report.items():
        if key not in ['accuracy', 'macro avg', 'weighted avg']:
            if isinstance(values, dict):
                print(f"{key:<15} {values['precision']:>12.4f} {values['recall']:>12.4f} {values['f1-score']:>12.4f} {values['support']:>12.0f}")


def print_regression_results(metrics):
    """Print regression results in a formatted way."""
    print("\n" + "=" * 70)
    print("📊 REGRESSION METRICS")
    print("=" * 70)
    
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'MSE':<25} {metrics['mse']:>15.4f}")
    print(f"{'RMSE':<25} {metrics['rmse']:>15.4f}")
    print(f"{'MAE':<25} {metrics['mae']:>15.4f}")
    print(f"{'R² Score':<25} {metrics['r2']:>15.4f}")
    print(f"{'MAPE (%)':<25} {metrics['mape']:>15.2f}")


def evaluate_model(args):
    """Main evaluation function."""
    print("=" * 70)
    print("🔍 RLT MODEL EVALUATION")
    print("=" * 70)
    
    # Load artifacts
    print("\n📂 Loading model artifacts...")
    model, preprocessor, metadata, test_data, label_encoder, model_name = load_model_artifacts(
        args.model_dir, args.model_name
    )
    
    # Print model info
    print(f"\n📋 Model Information:")
    print(f"   Dataset: {metadata['dataset']}")
    print(f"   Task type: {metadata['task_type']}")
    print(f"   Training timestamp: {metadata['timestamp']}")
    print(f"   Training time: {metadata['training_time']:.2f}s")
    print(f"   Training samples: {metadata['n_train_samples']}")
    print(f"   Test samples: {metadata['n_test_samples']}")
    
    print(f"\n⚙️  Hyperparameters:")
    for key, value in metadata['hyperparameters'].items():
        print(f"   {key}: {value}")
    
    # Get test data
    X_test_processed = test_data['X_test_processed']
    y_test = test_data['y_test']
    
    # Make predictions
    print("\n🔮 Making predictions...")
    import time
    start_time = time.time()
    y_pred = model.predict(X_test_processed)
    pred_time = time.time() - start_time
    print(f"   Prediction time: {pred_time:.4f}s")
    print(f"   Samples per second: {len(y_test)/pred_time:.0f}")
    
    # Compute metrics
    task_type = metadata['task_type']
    class_names = metadata.get('class_names')
    
    if task_type == 'classification':
        # Get probabilities if available
        y_proba = None
        try:
            y_proba = model.predict_proba(X_test_processed)
        except:
            pass
        
        metrics = evaluate_classification(y_test, y_pred, y_proba, class_names)
        print_classification_results(metrics, class_names)
    else:
        metrics = evaluate_regression(y_test, y_pred)
        print_regression_results(metrics)
    
    # Feature importance
    print("\n" + "=" * 70)
    print("🎯 TOP FEATURE IMPORTANCES")
    print("=" * 70)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Get feature names after preprocessing
        feature_names = []
        if hasattr(preprocessor, 'get_feature_names_out'):
            try:
                feature_names = list(preprocessor.get_feature_names_out())
            except:
                feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:]
        
        print(f"\n{'Rank':<6} {'Feature':<40} {'Importance':>12}")
        print("-" * 60)
        for rank, idx in enumerate(indices, 1):
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            print(f"{rank:<6} {name:<40} {importances[idx]:>12.6f}")
    
    # Save evaluation results
    if args.save_results:
        results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'prediction_time': pred_time,
            'task_type': task_type,
            'metrics': metrics
        }
        
        results_path = os.path.join(args.model_dir, f"{model_name}_evaluation.json")
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        results = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {results_path}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETED")
    print("=" * 70)
    
    return metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RLT model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model-dir', '-d',
        type=str,
        default='models',
        help='Directory containing model artifacts'
    )
    
    parser.add_argument(
        '--model-name', '-n',
        type=str,
        default=None,
        help='Name of the model to evaluate (default: latest)'
    )
    
    parser.add_argument(
        '--save-results', '-s',
        action='store_true',
        help='Save evaluation results to JSON file'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args)
