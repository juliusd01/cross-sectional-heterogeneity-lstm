"""
LSTM Model Architectures for Stock Return Prediction
"""
import torch
from torch import nn


class LSTMModel(nn.Module):
    """
    Base LSTM model for stock return prediction.
    Handles single feature (returns) or multiple features (covariates).
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with multiple layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for all layers
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Process sequence through LSTM layers
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        
        # Use final hidden state from last layer
        out = self.fc(h_n[-1])
        return out


class HeterogeneityLSTMModel(nn.Module):
    """
    LSTM model with sector embeddings for capturing heterogeneity.
    Combines LSTM hidden state with sector embeddings before classification.
    """
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        num_classes, 
        num_layers, 
        num_sectors, 
        sector_embedding_dim, 
        dropout
    ):
        super(HeterogeneityLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM with multiple layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Embedding layer for sector information
        self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim)
        
        # Combined input includes LSTM output and sector embedding
        combined_input_size = hidden_size + sector_embedding_dim
        self.fc = nn.Linear(combined_input_size, num_classes)
    
    def forward(self, x, sectors):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden and cell states for all layers
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Process sequence through LSTM
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        
        # Get sector embeddings
        sector_embeds = self.sector_embedding(sectors)
        
        # Concatenate LSTM output with sector embeddings
        out = self.fc(torch.cat((h_n[-1], sector_embeds), dim=1))
        return out


def get_model(model_architecture, **kwargs):
    """
    Factory function to create model based on architecture type.
    
    Args:
        model_architecture: 'base', 'covariates', or 'heterogeneity'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Initialized PyTorch model
    """
    if model_architecture in ['base', 'covariates']:
        return LSTMModel(**kwargs)
    elif model_architecture == 'heterogeneity':
        return HeterogeneityLSTMModel(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")
