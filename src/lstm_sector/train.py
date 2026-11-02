"""
Unified training script for LSTM stock return prediction experiments.
"""
import random
import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import torch.optim as optim
from ._mlflow import mlflow
import argparse
from datetime import datetime

from .paths import MODELS, PREDICTIONS, SEQUENCES
from .models import get_model


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(year, data_path, model_architecture, model_experiment):
    """
    Load and prepare data for training.
    
    Args:
        year: Year for test set
        data_path: Path to input data
        model_architecture: Type of architecture ('base', 'covariates', 'heterogeneity')
        model_experiment: Experiment type (only needed for 'heterogeneity_covariates')
    
    Returns:
        Dictionary containing train/val/test splits and metadata
    """
    sequences_df = pd.read_parquet(data_path)
    sequences_df = sequences_df.sort_values('date').reset_index(drop=True)
    
    if model_architecture == 'covariates':
        # Multiple features
        feature_names = [col for col in sequences_df.columns if col.endswith('sequence')]
        X = np.stack([np.array(sequences_df[col].tolist()) for col in feature_names], axis=-1)
        
        # Check for NaN or Inf values
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"WARNING: NaN or Inf values found in input features.")
            print(f"NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
        
        input_size = len(feature_names)
    else:
        if model_experiment == 'heterogeneity_covariates':
            # Multiple features including sector info
            feature_names = [col for col in sequences_df.columns if col.endswith('sequence')]
            X = np.stack([np.array(sequences_df[col].tolist()) for col in feature_names], axis=-1)
            input_size = len(feature_names)
        else:
            # Single feature (returns)
            X = np.array(sequences_df['sequence'].values.tolist())
            num_lags = X.shape[1]
            X = X.reshape(X.shape[0], num_lags, 1)
            input_size = 1
    
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
    
    # Handle sector information for heterogeneity model
    sectors_train = None
    sectors_val = None
    sectors_test = None
    num_sectors = None
    
    if model_architecture == 'heterogeneity':
        sequences_df["sector_id"] = sequences_df["sector"].astype("category").cat.codes
        num_sectors = sequences_df["sector_id"].nunique()
        
        sectors = sequences_df['sector_id'].values
        sectors_test = sectors[test_mask.values]
        sectors_train_val = sectors[~test_mask.values]
        sectors_train = sectors_train_val[:split_idx]
        sectors_val = sectors_train_val[split_idx:]
    
    return {
        'X_train': X_train, 'y_train': y_train, 'sectors_train': sectors_train,
        'X_val': X_val, 'y_val': y_val, 'sectors_val': sectors_val,
        'X_test': X_test, 'y_test': y_test, 'sectors_test': sectors_test,
        'test_metadata': test_metadata,
        'input_size': input_size,
        'num_sectors': num_sectors
    }


def train_epoch(model, train_loader, criterion, optimizer, device, model_architecture):
    """Train for one epoch."""
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch_data in train_loader:
        if model_architecture == 'heterogeneity':
            batch_X, batch_sectors, labels = batch_data
            batch_sectors = batch_sectors.to(device)
            outputs = model(batch_X, batch_sectors)
        else:
            batch_X, labels = batch_data
            outputs = model(batch_X)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    return train_loss / len(train_loader), train_correct / train_total


def validate_epoch(model, val_loader, criterion, device, model_architecture):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            if model_architecture == 'heterogeneity':
                batch_X, batch_sectors, labels = batch_data
                batch_sectors = batch_sectors.to(device)
                outputs = model(batch_X, batch_sectors)
            else:
                batch_X, labels = batch_data
                outputs = model(batch_X)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    return val_loss / len(val_loader), val_correct / val_total


def train_model(args, year, data):
    """
    Train model for a single year.
    
    Args:
        args: Parsed command-line arguments
        year: Year for test set
        data: Dictionary containing prepared data
    
    Returns:
        Test metrics (loss, accuracy)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Initialize model
    model_kwargs = {
        'input_size': data['input_size'],
        'hidden_size': args.hidden_size,
        'num_classes': 2,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    if args.model_architecture == 'heterogeneity':
        model_kwargs['num_sectors'] = data['num_sectors']
        model_kwargs['sector_embedding_dim'] = args.embedding_dim
    
    model = get_model(args.model_architecture, **model_kwargs)
    model.to(device)
    print(model)
    
    # Log parameters
    mlflow.log_param("year", year)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("hidden_size", args.hidden_size)
    mlflow.log_param("num_layers", args.num_layers)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("num_lags", args.num_lags)
    mlflow.log_param("device", device)
    mlflow.log_param("model_architecture", args.model_architecture)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of model parameters: {num_params}')
    mlflow.log_param("num_parameters", num_params)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(data['X_train']).to(device)
    y_train_tensor = torch.LongTensor(data['y_train']).to(device)
    X_val_tensor = torch.FloatTensor(data['X_val']).to(device)
    y_val_tensor = torch.LongTensor(data['y_val']).to(device)
    
    # Create data loaders
    if args.model_architecture == 'heterogeneity':
        sectors_train_tensor = torch.LongTensor(data['sectors_train']).to(device)
        sectors_val_tensor = torch.LongTensor(data['sectors_val']).to(device)
        train_dataset = TensorDataset(X_train_tensor, sectors_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, sectors_val_tensor, y_val_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    # Use sequential sampler if specified (for heterogeneity experiments)
    if args.sequential_sampler:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=SequentialSampler(train_dataset)
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            generator=g
        )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(f"{MODELS}/{args.model_type}/{args.model_id}", exist_ok=True)
    best_model_path = f"{MODELS}/{args.model_type}/{args.model_id}/best_lstm_model_{year}.pth"
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, args.model_architecture
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, args.model_architecture
        )
        
        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)
        mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
        
        # Step the scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch+1}')
                mlflow.log_metric("epochs", epoch+1)
                break
    
    # Load best model and evaluate
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    print('Training complete!')
    
    # Evaluate on test set
    X_test_tensor = torch.FloatTensor(data['X_test']).to(device)
    y_test_tensor = torch.LongTensor(data['y_test']).to(device)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), 256):
            batch_X = X_test_tensor[i:i+256]
            
            if args.model_architecture == 'heterogeneity':
                sectors_test_tensor = torch.LongTensor(data['sectors_test']).to(device)
                batch_sectors = sectors_test_tensor[i:i+256]
                pred = model(batch_X, batch_sectors)
            else:
                pred = model(batch_X)
            
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=0)
        _, predicted_labels = torch.max(predictions, 1)
        test_accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
        test_loss = criterion(predictions, y_test_tensor).item()
    
    # Log test metrics
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_acc", test_accuracy)
    
    # Save predictions
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(predictions, dim=1)
    prob_np = probabilities.cpu().numpy()
    results_df = data['test_metadata'].copy()
    results_df['pred_Class0'] = prob_np[:, 0]
    results_df['pred_Class1'] = prob_np[:, 1]
    results_df['predicted_class'] = np.argmax(prob_np, axis=1)
    results_df['actual_class'] = data['y_test']
    
    output_parquet = f"{PREDICTIONS}/{args.model_type}/{args.model_id}/predictions_{year}.parquet"
    os.makedirs(f"{PREDICTIONS}/{args.model_type}/{args.model_id}", exist_ok=True)
    results_df.to_parquet(output_parquet, index=False)
    
    mlflow.log_artifact(output_parquet)
    mlflow.log_artifact(best_model_path)
    
    print(f'Year {year} complete! Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
    print(f'Saved predictions to {output_parquet}')
    print('-' * 50)
    
    return test_loss, test_accuracy


def main():
    parser = argparse.ArgumentParser(description='Train LSTM for stock return prediction')
    
    # Model architecture
    parser.add_argument('--model_architecture', type=str, required=True,
                        choices=['base', 'covariates', 'heterogeneity'],
                        help='Model architecture type')
    
    # Data settings
    parser.add_argument('--input_subdir', type=str, required=True,
                        help='Input subdirectory (e.g., lag_60, lag_60_cov, lag_60_sector)')
    parser.add_argument('--model_type', type=str, required=True,
                        help='Model type for saving (e.g., lstm, lstm_cov, lstm_sector)')
    
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--hidden_size', type=int, default=25, help='LSTM hidden size')
    parser.add_argument('--num_lags', type=int, default=60, help='Number of lags')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--embedding_dim', type=int, default=3, help='Sector embedding dimension (for heterogeneity model)')
    
    # Training settings
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--start_year', type=int, default=1998, help='Start year')
    parser.add_argument('--end_year', type=int, default=2024, help='End year')
    parser.add_argument('--experiment_type', type=str, required=True,
                        help='MLflow experiment name')
    parser.add_argument('--sequential_sampler', action='store_true',
                        help='Use sequential sampler instead of shuffling')
    parser.add_argument('--job_id', type=str, default=None, help='SLURM job ID')
    
    args = parser.parse_args()
    
    # Auto-set dropout if not specified
    if args.dropout == 0.0 and args.num_layers > 1:
        args.dropout = 0.1
    
    # Generate model ID with job_id for uniqueness
    args.model_id = f'h_{args.hidden_size}_l_{args.num_layers}_lr_{args.learning_rate}'
    if args.batch_size != 64:
        args.model_id += f'_B{args.batch_size}'
    if args.model_architecture == 'heterogeneity':
        args.model_id += f'_D{args.embedding_dim}'
    if args.sequential_sampler:
        args.model_id += '_seq'
    if args.job_id:
        args.model_id += f'_{args.job_id}'
    
    # Set seed
    set_seed(args.seed)
    
    # MLflow setup
    mlflow.set_experiment(args.experiment_type)
    
    # Parent run
    with mlflow.start_run(run_name=f"{args.model_type}_{args.model_id}") as parent_run:
        mlflow.set_tag("hidden_size", args.hidden_size)
        mlflow.set_tag("num_layers", args.num_layers)
        mlflow.set_tag("learning_rate", args.learning_rate)
        
        for year in range(args.start_year, args.end_year + 1):
            with mlflow.start_run(run_name=f"year_{year}", nested=True):
                # Load data
                data_path = f'{SEQUENCES}/{args.input_subdir}/returns_{year}.parquet'
                data = load_data(year, data_path, args.model_architecture, args.experiment_type)
                
                # Train model
                train_model(args, year, data)


if __name__ == '__main__':
    main()