"""
Flask Application for RLT vs Random Forest Comparison
"""

import os
import sys
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
# Add src and utils directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

from RLT import ReinforcementLearningTrees
from utils.dataset_wrapper import datasets_dict, DatasetWrapper
from scripts.data_preparation import prepare_data

app = Flask(__name__)

# Store trained models globally for predictions
trained_models = {
    'rlt': None,
    'rf': None,
    'wrapper': None,
    'feature_names': None,
    'task_type': None
}

# Configuration
DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets')


def get_available_datasets():
    """Get list of available datasets from dataset_wrapper"""
    datasets = []
    for name, config in datasets_dict.items():
        task_type = 'classification' if config['type'] == 'Categorical' else 'regression'
        datasets.append({
            'name': name,
            'filename': config['path'],
            'description': f"{name.replace('_', ' ').title()} - {task_type.title()}",
            'task_type': task_type,
            'target': config['target']
        })
    return datasets


def load_dataset(dataset_name):
    """Load and preprocess dataset using DatasetWrapper and prepare_data"""
    wrapper = DatasetWrapper(dataset_name)
    X_train, X_test, y_train, y_test = prepare_data(wrapper)
    feature_names = wrapper.clean_variables
    task_type = wrapper.task_type
    
    return X_train, X_test, y_train, y_test, feature_names, task_type, wrapper


def create_feature_importance_plot(rlt_importance, rf_importance, feature_names):
    """Create side-by-side feature importance comparison plot"""
    n_features = len(feature_names)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize importances
    rlt_norm = rlt_importance / (rlt_importance.sum() + 1e-10)
    rf_norm = rf_importance / (rf_importance.sum() + 1e-10)
    
    # Sort by RLT importance for better visualization
    sorted_idx = np.argsort(rlt_norm)[::-1]
    
    # Take top 20 features if too many
    if n_features > 20:
        top_idx = sorted_idx[:20]
        display_names = [feature_names[i] for i in top_idx]
        rlt_display = rlt_norm[top_idx]
        rf_display = rf_norm[top_idx]
        title_suffix = " (Top 20 Features)"
    else:
        display_names = [feature_names[i] for i in sorted_idx]
        rlt_display = rlt_norm[sorted_idx]
        rf_display = rf_norm[sorted_idx]
        title_suffix = ""
    
    x_pos = np.arange(len(display_names))
    
    # RLT Feature Importance
    axes[0].bar(x_pos, rlt_display, color='#2E86AB', edgecolor='black', linewidth=0.5)
    axes[0].set_title(f'RLT Feature Importance{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Normalized Importance', fontsize=12)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Random Forest Feature Importance
    axes[1].bar(x_pos, rf_display, color='#A23B72', edgecolor='black', linewidth=0.5)
    axes[1].set_title(f'Random Forest Feature Importance{title_suffix}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Normalized Importance', fontsize=12)
    axes[1].set_xlabel('Features', fontsize=12)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


def create_comparison_bar_plot(rlt_importance, rf_importance, feature_names):
    """Create grouped bar chart comparing both models"""
    n_features = len(feature_names)
    
    # Normalize importances
    rlt_norm = rlt_importance / (rlt_importance.sum() + 1e-10)
    rf_norm = rf_importance / (rf_importance.sum() + 1e-10)
    
    # Sort by average importance
    avg_importance = (rlt_norm + rf_norm) / 2
    sorted_idx = np.argsort(avg_importance)[::-1]
    
    # Take top 15 features
    top_n = min(15, n_features)
    top_idx = sorted_idx[:top_n]
    
    display_names = [feature_names[i] for i in top_idx]
    rlt_display = rlt_norm[top_idx]
    rf_display = rf_norm[top_idx]
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(display_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rlt_display, width, label='RLT', color='#2E86AB', edgecolor='black')
    bars2 = ax.bar(x + width/2, rf_display, width, label='Random Forest', color='#A23B72', edgecolor='black')
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Normalized Importance', fontsize=12)
    ax.set_title('Feature Importance Comparison: RLT vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64


@app.route('/')
def index():
    """Home page"""
    datasets = get_available_datasets()
    return render_template('index.html', datasets=datasets)


@app.route('/run_comparison', methods=['POST'])
def run_comparison():
    """Run model comparison"""
    global trained_models
    
    try:
        # Get parameters from request
        dataset_name = request.form.get('dataset')
        random_state = int(request.form.get('random_state', 42))
        n_rlt_trees = int(request.form.get('n_rlt_trees', 50))
        n_extra_trees = int(request.form.get('n_extra_trees', 100))
        muting_rate = float(request.form.get('muting_rate', 0.5))
        k = int(request.form.get('k', 4))
        
        # Load dataset using dataset_wrapper and prepare_data (train_size fixed at 150)
        X_train, X_test, y_train, y_test, feature_names, task_type, wrapper = load_dataset(dataset_name)
        
        n_samples, n_features = X_train.shape
        
        # Calculate RLT parameters (same as notebook)
        p0 = max(1, int(np.log(n_features)))
        n_min = max(1, int(n_samples ** (1 / 3)))
        
        # Initialize RLT model with user parameters
        rlt = ReinforcementLearningTrees(
            task_type=task_type,
            n_rlt_trees=n_rlt_trees,
            n_extra_trees=n_extra_trees,
            muting_rate=muting_rate,
            min_samples_split=n_min,
            min_protected=p0,
            k=k,
            n_jobs=-1,
            random_state=random_state
        )
        
        # Initialize Random Forest with same random state
        if task_type == 'classification':
            rf = RandomForestClassifier(
                n_estimators=n_rlt_trees,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=n_rlt_trees,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
        
        # Train models
        rlt.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        
        # Store models for prediction
        trained_models['rlt'] = rlt
        trained_models['rf'] = rf
        trained_models['wrapper'] = wrapper
        trained_models['feature_names'] = feature_names
        trained_models['task_type'] = task_type
        
        # Get predictions
        rlt_pred = rlt.predict(X_test)
        rf_pred = rf.predict(X_test)
        
        # Calculate metrics
        if task_type == 'classification':
            rlt_metrics = {
                'Accuracy': float(round(accuracy_score(y_test, rlt_pred), 4)),
                'Precision': float(round(precision_score(y_test, rlt_pred, average='weighted', zero_division=0), 4)),
                'Recall': float(round(recall_score(y_test, rlt_pred, average='weighted', zero_division=0), 4)),
                'F1-Score': float(round(f1_score(y_test, rlt_pred, average='weighted', zero_division=0), 4))
            }
            rf_metrics = {
                'Accuracy': float(round(accuracy_score(y_test, rf_pred), 4)),
                'Precision': float(round(precision_score(y_test, rf_pred, average='weighted', zero_division=0), 4)),
                'Recall': float(round(recall_score(y_test, rf_pred, average='weighted', zero_division=0), 4)),
                'F1-Score': float(round(f1_score(y_test, rf_pred, average='weighted', zero_division=0), 4))
            }
        else:
            rlt_metrics = {
                'MSE': float(round(mean_squared_error(y_test, rlt_pred), 4)),
                'RMSE': float(round(np.sqrt(mean_squared_error(y_test, rlt_pred)), 4)),
                'MAE': float(round(mean_absolute_error(y_test, rlt_pred), 4)),
                'R² Score': float(round(r2_score(y_test, rlt_pred), 4))
            }
            rf_metrics = {
                'MSE': float(round(mean_squared_error(y_test, rf_pred), 4)),
                'RMSE': float(round(np.sqrt(mean_squared_error(y_test, rf_pred)), 4)),
                'MAE': float(round(mean_absolute_error(y_test, rf_pred), 4)),
                'R² Score': float(round(r2_score(y_test, rf_pred), 4))
            }
        
        # Get feature importances
        rlt_importance = rlt.feature_importances_
        rf_importance = rf.feature_importances_
        
        # Create plots
        importance_plot = create_feature_importance_plot(
            rlt_importance, rf_importance, feature_names
        )
        comparison_plot = create_comparison_bar_plot(
            rlt_importance, rf_importance, feature_names
        )
        
        # Prepare feature importance table data
        feature_importance_data = []
        rlt_norm = rlt_importance / (rlt_importance.sum() + 1e-10)
        rf_norm = rf_importance / (rf_importance.sum() + 1e-10)
        
        for i, name in enumerate(feature_names):
            feature_importance_data.append({
                'feature': name,
                'rlt_importance': float(round(rlt_norm[i], 6)),
                'rf_importance': float(round(rf_norm[i], 6)),
                'difference': float(round(rlt_norm[i] - rf_norm[i], 6))
            })
        
        # Sort by RLT importance
        feature_importance_data.sort(key=lambda x: x['rlt_importance'], reverse=True)
        
        # Get feature stats for prediction form
        df = wrapper.df
        feature_stats = {}
        for col in feature_names:
            if col in df.columns:
                feature_stats[col] = {
                    'min': round(float(df[col].min()), 4),
                    'max': round(float(df[col].max()), 4),
                    'mean': round(float(df[col].mean()), 4)
                }
        
        return jsonify({
            'success': True,
            'task_type': task_type,
            'dataset': dataset_name,
            'n_samples': int(len(y_train) + len(y_test)),
            'n_features': int(len(feature_names)),
            'n_train': int(len(y_train)),
            'n_test': int(len(y_test)),
            'random_state': int(random_state),
            'rlt_metrics': rlt_metrics,
            'rf_metrics': rf_metrics,
            'importance_plot': importance_plot,
            'comparison_plot': comparison_plot,
            'feature_importance_data': feature_importance_data,
            'feature_names': list(feature_names),
            'feature_stats': feature_stats,
            'class_names': [str(c) for c in wrapper.class_names] if task_type == 'classification' and wrapper.class_names else None
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions with trained models"""
    global trained_models
    
    try:
        if trained_models['rlt'] is None or trained_models['rf'] is None:
            return jsonify({
                'success': False,
                'error': 'No models trained yet. Please run comparison first.'
            })
        
        # Get feature values from request
        data = request.get_json()
        feature_values = data.get('features', {})
        
        feature_names = trained_models['feature_names']
        wrapper = trained_models['wrapper']
        
        # Build input array
        X_input = []
        for name in feature_names:
            value = float(feature_values.get(name, 0))
            X_input.append(value)
        
        X_input = np.array([X_input])
        
        # Scale input using the wrapper's scaler
        if wrapper.scaler is not None:
            X_input_scaled = wrapper.scaler.transform(X_input)
        else:
            X_input_scaled = X_input
        
        # Make predictions
        rlt_pred = trained_models['rlt'].predict(X_input_scaled)[0]
        rf_pred = trained_models['rf'].predict(X_input_scaled)[0]
        
        task_type = trained_models['task_type']
        
        result = {
            'success': True,
            'task_type': task_type
        }
        
        if task_type == 'classification':
            class_names = wrapper.class_names
            
            # Get probabilities if available
            try:
                rlt_proba = trained_models['rlt'].predict_proba(X_input_scaled)[0]
                rf_proba = trained_models['rf'].predict_proba(X_input_scaled)[0]
                
                result['rlt_prediction'] = {
                    'class': str(class_names[int(rlt_pred)]) if class_names else int(rlt_pred),
                    'probabilities': {str(class_names[i]) if class_names else str(i): float(round(float(p), 4)) for i, p in enumerate(rlt_proba)}
                }
                result['rf_prediction'] = {
                    'class': str(class_names[int(rf_pred)]) if class_names else int(rf_pred),
                    'probabilities': {str(class_names[i]) if class_names else str(i): float(round(float(p), 4)) for i, p in enumerate(rf_proba)}
                }
            except:
                result['rlt_prediction'] = {
                    'class': str(class_names[int(rlt_pred)]) if class_names else int(rlt_pred)
                }
                result['rf_prediction'] = {
                    'class': str(class_names[int(rf_pred)]) if class_names else int(rf_pred)
                }
        else:
            result['rlt_prediction'] = {'value': float(round(float(rlt_pred), 4))}
            result['rf_prediction'] = {'value': float(round(float(rf_pred), 4))}
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


@app.route('/dataset_info/<dataset_name>')
def dataset_info(dataset_name):
    """Get dataset information"""
    try:
        wrapper = DatasetWrapper(dataset_name)
        config = datasets_dict.get(dataset_name, {})
        
        # Get feature stats
        df = wrapper.df
        feature_stats = {}
        for col in wrapper.quantitatives_variables:
            feature_stats[col] = {
                'min': round(float(df[col].min()), 4),
                'max': round(float(df[col].max()), 4),
                'mean': round(float(df[col].mean()), 4)
            }
        
        return jsonify({
            'success': True,
            'n_samples': int(len(wrapper.df)),
            'n_features': int(len(wrapper.quantitatives_variables)),
            'feature_names': wrapper.quantitatives_variables,
            'task_type': wrapper.task_type,
            'description': dataset_name.replace('_', ' ').title(),
            'target': config.get('target', 'Unknown'),
            'feature_stats': feature_stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
