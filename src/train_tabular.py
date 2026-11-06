"""
Random Forest and HistGradientBoosting training script for stock return prediction.
Simplified for local laptop use - no command line arguments.
"""
import random
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

from paths import PREDICTIONS, SEQUENCES


def set_seed(seed=123):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_data(year, data_path):
    """
    Load and prepare data for training.
    
    Args:
        year: Year for test set
        data_path: Path to input data
    
    Returns:
        Dictionary containing train/val/test splits and metadata
    """
    sequences_df = pd.read_parquet(data_path)
    sequences_df = sequences_df.sort_values('date').reset_index(drop=True)
    
    # Extract features from the 22 cumulative return columns
    feature_cols = [col for col in sequences_df.columns if col.startswith('cum_return_') or col == 'return_t1']
    X = sequences_df[feature_cols].values
    
    print(f"Features used: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    
    # Check for NaN or Inf values
    if np.isnan(X).any() or np.isinf(X).any():
        print(f"WARNING: NaN or Inf values found in input features.")
        print(f"NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
        # Replace NaN/Inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prepare labels
    y = sequences_df[['Class0', 'Class1']].values
    y = np.argmax(y, axis=1)
    
    # Train-Test Split based on year
    test_mask = sequences_df['date'].dt.year == int(year)
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    X_train_val = X[~test_mask.values]
    y_train_val = y[~test_mask.values]
    
    # Keep metadata for test set
    test_metadata = sequences_df[test_mask][['stock', 'date', 'return']].reset_index(drop=True)
    
    # Split train_val into train and val (80% train, 20% val)
    split_idx = int(0.8 * len(X_train_val))
    X_train = X_train_val[:split_idx]
    y_train = y_train_val[:split_idx]
    X_val = X_train_val[split_idx:]
    y_val = y_train_val[split_idx:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'test_metadata': test_metadata,
        'feature_cols': feature_cols
    }


def get_model(config):
    """Initialize model based on configuration."""
    if config['model_name'] == 'rf':
        return RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            random_state=config['seed'],
            n_jobs=-1,
            verbose=1
        )
    elif config['model_name'] == 'hgb':
        return HistGradientBoostingClassifier(
            max_iter=config['max_iter'],
            learning_rate=config['learning_rate'],
            max_depth=config['max_depth'],
            min_samples_leaf=config['min_samples_leaf'],
            max_leaf_nodes=config['max_leaf_nodes'],
            l2_regularization=config['l2_regularization'],
            random_state=config['seed'],
            verbose=1
        )
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")


def train_model(year, data, config):
    """
    Train model for a single year.
    
    Args:
        year: Year for test set
        data: Dictionary containing prepared data
        config: Configuration dictionary with hyperparameters
    
    Returns:
        Test metrics (accuracy)
    """
    model_full_name = "Random Forest" if config['model_name'] == 'rf' else "HistGradientBoosting"
    print(f"\n{'='*50}")
    print(f"Training {model_full_name} for year {year}")
    print(f"{'='*50}")
    
    # Initialize model
    model = get_model(config)
    
    # Train the model
    n_trees = config['n_estimators'] if config['model_name'] == 'rf' else config['max_iter']
    print(f"Training {model_full_name} with {n_trees} iterations...")
    model.fit(data['X_train'], data['y_train'])
    print("Training complete!")
    
    # Validation set evaluation
    val_pred = model.predict(data['X_val'])
    val_accuracy = accuracy_score(data['y_val'], val_pred)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Test set evaluation
    test_pred = model.predict(data['X_test'])
    test_pred_proba = model.predict_proba(data['X_test'])
    test_accuracy = accuracy_score(data['y_test'], test_pred)
    
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_test'], test_pred, target_names=['Class0', 'Class1']))
    
    # Save predictions
    results_df = data['test_metadata'].copy()
    results_df['pred_Class0'] = test_pred_proba[:, 0]
    results_df['pred_Class1'] = test_pred_proba[:, 1]
    results_df['predicted_class'] = test_pred
    results_df['actual_class'] = data['y_test']
    
    output_parquet = f"{PREDICTIONS}/{config['model_name']}/{config['model_id']}/predictions_{year}.parquet"
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    results_df.to_parquet(output_parquet, index=False)
    
    print(f'\nSaved predictions to {output_parquet}')
    print('-' * 50)
    
    return test_accuracy


def main():
    # Configuration
    config = {
        'model_name': 'hgb',  # 'rf' for Random Forest, 'hgb' for HistGradientBoosting
        'seed': 123,
        'start_year': 1998,
        'end_year': 2024,
        
        # Random Forest hyperparameters (used when model_name='rf')
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        
        # HistGradientBoosting hyperparameters (used when model_name='hgb')
        'max_iter': 100,
        'learning_rate': 0.1,
        # max_depth, min_samples_leaf are shared
        'max_leaf_nodes': 31,
        'l2_regularization': 0.0,
    }
    
    # Generate model ID based on model type
    if config['model_name'] == 'rf':
        config['model_id'] = f'n{config["n_estimators"]}_d{config["max_depth"]}_ms{config["min_samples_split"]}'
    elif config['model_name'] == 'hgb':
        config['model_id'] = f'iter{config["max_iter"]}_lr{config["learning_rate"]}_d{config["max_depth"]}'
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")
    # Set seed
    set_seed(config['seed'])
    
    test_accuracies = []
    
    for year in range(config['start_year'], config['end_year'] + 1):
        # Load data
        data_path = f'{SEQUENCES}/tabular/returns_{year}.parquet'
        
        if not os.path.exists(data_path):
            print(f"WARNING: Data file not found: {data_path}")
            continue
        
        data = load_data(year, data_path)
        
        # Train model
        test_acc = train_model(year, data, config)
        test_accuracies.append(test_acc)
    
    # Log average test accuracy across all years
    if test_accuracies:
        avg_test_acc = np.mean(test_accuracies)
        print(f"\n{'='*50}")
        print(f"Average Test Accuracy across all years: {avg_test_acc:.4f}")
        print(f"{'='*50}")


if __name__ == '__main__':
    main()
