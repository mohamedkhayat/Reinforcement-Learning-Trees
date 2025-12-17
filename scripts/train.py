"""
Training script for RLT model with preprocessing pipeline.
Supports hyperparameter configuration via command line arguments.
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from RLT import ReinforcementLearningTrees
from utils.dataset_wrapper import DatasetWrapper, datasets_dict


def load_from_csv(csv_path, target_col, task_type='classification', test_size=0.2, random_state=42):
    """
    Load dataset from external CSV file.
    Automatically detects numerical and categorical features.
    """
    print(f"   Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Detect feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    X = X[numerical_features + categorical_features].copy()
    
    # Encode target for classification
    label_encoder = None
    class_names = None
    
    if task_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        class_names = y.unique().tolist()
        y = label_encoder.fit_transform(y)
    else:
        y = y.values
    
    # Split data
    stratify = y if task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return (X_train, X_test, y_train, y_test, 
            numerical_features, categorical_features, 
            task_type, label_encoder, class_names)


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a preprocessing pipeline with:
    - KNN imputation + StandardScaler for numerical features
    - Mode imputation + OneHotEncoder for categorical features
    """
    # Numerical pipeline: KNN imputation -> StandardScaler
    numerical_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Mode imputation -> OneHotEncoder
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def load_from_csv(csv_path, target_col, task_type='classification', test_size=0.2, random_state=42):
    """
    Load dataset from external CSV file.
    Automatically detects numerical and categorical features.
    """
    print(f"   Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)}")
    
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Detect feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    X = X[numerical_features + categorical_features].copy()
    
    # Encode target for classification
    label_encoder = None
    class_names = None
    
    if task_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        class_names = y.unique().tolist()
        y = label_encoder.fit_transform(y)
    else:
        y = y.values
    
    # Split data
    stratify = y if task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return (X_train, X_test, y_train, y_test, 
            numerical_features, categorical_features, 
            task_type, label_encoder, class_names)


def load_and_prepare_data(dataset_name, test_size=0.2, random_state=42):
    """Load dataset from dataset_wrapper and split into train/test sets."""
    wrapper = DatasetWrapper(dataset_name)
    df = wrapper.df.copy()
    
    # Get target
    target_col = wrapper.target
    y = df[target_col].copy()
    
    # Get features
    numerical_features = wrapper.quantitatives_variables
    categorical_features = wrapper.categorical_variables
    
    X = df[numerical_features + categorical_features].copy()
    
    # Encode target for classification
    task_type = wrapper.task_type
    label_encoder = None
    
    if task_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split data
    stratify = y if task_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    return (X_train, X_test, y_train, y_test, 
            numerical_features, categorical_features, 
            task_type, label_encoder, wrapper.class_names if hasattr(wrapper, 'class_names') else None)


def train_model(args):
    """Train RLT model with given hyperparameters."""
    print("=" * 70)
    print("🌳 RLT MODEL TRAINING")
    print("=" * 70)
    
    # Determine data source
    if args.csv_path:
        dataset_name = os.path.splitext(os.path.basename(args.csv_path))[0]
        print(f"CSV File: {args.csv_path}")
        print(f"Target: {args.target}")
        print(f"Task Type: {args.task_type}")
    else:
        dataset_name = args.dataset
        print(f"Dataset: {args.dataset}")
    
    print(f"Random State: {args.random_state}")
    print("-" * 70)
    
    # Load data
    print("\n📊 Loading and preparing data...")
    
    if args.csv_path:
        # Load from external CSV
        if not args.target:
            raise ValueError("--target is required when using --csv-path")
        (X_train, X_test, y_train, y_test, 
         numerical_features, categorical_features,
         task_type, label_encoder, class_names) = load_from_csv(
            args.csv_path, args.target, args.task_type, args.test_size, args.random_state
        )
    else:
        # Load from dataset_wrapper
        (X_train, X_test, y_train, y_test, 
         numerical_features, categorical_features,
         task_type, label_encoder, class_names) = load_and_prepare_data(
            args.dataset, args.test_size, args.random_state
        )
    
    print(f"   Task type: {task_type}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    
    # Create preprocessor
    print("\n🔧 Creating preprocessing pipeline...")
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    # Fit preprocessor and transform data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    n_features = X_train_processed.shape[1]
    print(f"   Features after preprocessing: {n_features}")
    
    # Calculate auto parameters
    n_samples = len(X_train)
    p0 = max(1, int(np.log(n_features)))
    n_min = max(1, int(n_samples ** (1 / 3)))
    
    # Override with args if provided
    min_protected = args.min_protected if args.min_protected > 0 else p0
    min_samples_split = args.min_samples_split if args.min_samples_split > 0 else n_min
    
    # Create RLT model
    print("\n🏗️  Building RLT model...")
    print(f"   n_rlt_trees: {args.n_rlt_trees}")
    print(f"   n_extra_trees: {args.n_extra_trees}")
    print(f"   muting_rate: {args.muting_rate}")
    print(f"   k: {args.k}")
    print(f"   min_protected: {min_protected}")
    print(f"   min_samples_split: {min_samples_split}")
    
    model = ReinforcementLearningTrees(
        task_type=task_type,
        n_rlt_trees=args.n_rlt_trees,
        n_extra_trees=args.n_extra_trees,
        muting_rate=args.muting_rate,
        min_samples_split=min_samples_split,
        min_protected=min_protected,
        k=args.k,
        n_jobs=args.n_jobs,
        random_state=args.random_state
    )
    
    # Train model
    print("\n🚀 Training model...")
    import time
    start_time = time.time()
    model.fit(X_train_processed, y_train)
    training_time = time.time() - start_time
    print(f"   Training completed in {training_time:.2f} seconds")
    
    # Print split statistics
    if hasattr(model, 'print_split_statistics'):
        model.print_split_statistics()
    
    # Save artifacts
    print("\n💾 Saving model artifacts...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"rlt_{dataset_name}_{timestamp}"
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)
    print(f"   Model saved: {model_path}")
    
    # Save preprocessor
    preprocessor_path = os.path.join(args.output_dir, f"{model_name}_preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   Preprocessor saved: {preprocessor_path}")
    
    # Save label encoder if classification
    if label_encoder is not None:
        encoder_path = os.path.join(args.output_dir, f"{model_name}_label_encoder.joblib")
        joblib.dump(label_encoder, encoder_path)
        print(f"   Label encoder saved: {encoder_path}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'csv_path': args.csv_path,
        'task_type': task_type,
        'timestamp': timestamp,
        'training_time': training_time,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'n_features_raw': len(numerical_features) + len(categorical_features),
        'n_features_processed': n_features,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'class_names': class_names,
        'hyperparameters': {
            'n_rlt_trees': args.n_rlt_trees,
            'n_extra_trees': args.n_extra_trees,
            'muting_rate': args.muting_rate,
            'k': args.k,
            'min_protected': min_protected,
            'min_samples_split': min_samples_split,
            'random_state': args.random_state
        }
    }
    
    metadata_path = os.path.join(args.output_dir, f"{model_name}_metadata.joblib")
    joblib.dump(metadata, metadata_path)
    print(f"   Metadata saved: {metadata_path}")
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'X_test_processed': X_test_processed
    }
    test_data_path = os.path.join(args.output_dir, f"{model_name}_test_data.joblib")
    joblib.dump(test_data, test_data_path)
    print(f"   Test data saved: {test_data_path}")
    
    # Save model name for easy reference
    latest_path = os.path.join(args.output_dir, "latest_model.txt")
    with open(latest_path, 'w') as f:
        f.write(model_name)
    print(f"   Latest model reference: {latest_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return model_name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RLT model with preprocessing pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        choices=list(datasets_dict.keys()),
        help='Name of the dataset from dataset_wrapper'
    )
    
    # External CSV arguments
    parser.add_argument(
        '--csv-path', '-c',
        type=str,
        default=None,
        help='Path to external CSV file (alternative to --dataset)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target column name (required with --csv-path)'
    )
    
    parser.add_argument(
        '--task-type',
        type=str,
        default='classification',
        choices=['classification', 'regression'],
        help='Task type for external CSV (default: classification)'
    )
    
    # Model hyperparameters
    parser.add_argument(
        '--n-rlt-trees', '-t',
        type=int,
        default=50,
        help='Number of RLT trees in the forest'
    )
    
    parser.add_argument(
        '--n-extra-trees', '-e',
        type=int,
        default=100,
        help='Number of trees in the embedded model'
    )
    
    parser.add_argument(
        '--muting-rate', '-m',
        type=float,
        default=0.5,
        help='Rate of feature muting (0.0 to 0.9)'
    )
    
    parser.add_argument(
        '--k', '-k',
        type=int,
        default=4,
        help='Number of top features for linear combination splits'
    )
    
    parser.add_argument(
        '--min-protected', '-p',
        type=int,
        default=0,
        help='Minimum number of protected features (0 = auto: log(n_features))'
    )
    
    parser.add_argument(
        '--min-samples-split', '-s',
        type=int,
        default=0,
        help='Minimum samples to split a node (0 = auto: n^(1/3))'
    )
    
    # Training arguments
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing'
    )
    
    parser.add_argument(
        '--random-state', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--n-jobs', '-j',
        type=int,
        default=-1,
        help='Number of parallel jobs (-1 = all cores)'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='models',
        help='Directory to save model artifacts'
    )
    
    args = parser.parse_args()
    
    # Validation: at least one data source required
    if args.dataset is None and args.csv_path is None:
        parser.error("Either --dataset or --csv-path must be provided")
    
    if args.dataset and args.csv_path:
        parser.error("Use either --dataset or --csv-path, not both")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
