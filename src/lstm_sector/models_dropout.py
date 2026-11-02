import torch
from torch import nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, 
                 input_dropout, recurrent_dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout_rate = input_dropout
        self.recurrent_dropout_rate = recurrent_dropout

        # Using LSTMCell for manual control over dropout types
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
            batch_size, seq_len, _ = x.size()
            device = x.device
            
            # Initialize hidden and cell states
            h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
            
            # Process sequence through time
            for t in range(seq_len):
                input_t = x[:, t, :]
                
                # 1. Input Dropout: Regularize the external features (Oil, VIX, etc.)
                if self.training and self.input_dropout_rate > 0:
                    input_t = F.dropout(input_t, p=self.input_dropout_rate)
                
                new_h, new_c = [], []
                for i, cell in enumerate(self.lstm_cells):
                    # Standard LSTM Cell computation
                    h_i, c_i = cell(input_t, (h[i], c[i]))
                    
                    # 2. Recurrent Dropout: Regularize the temporal memory
                    if self.training and self.recurrent_dropout_rate > 0:
                        h_i = F.dropout(h_i, p=self.recurrent_dropout_rate)
                    
                    new_h.append(h_i)
                    new_c.append(c_i)
                    input_t = h_i # Output becomes input for the next layer (if any)
                
                h, c = new_h, new_c
                
            # Classify using the final hidden state of the top layer
            out = self.fc(h[-1])
            return out

class HeterogeneityLSTMModel(nn.Module):
    """
    Heterogeneity model with Sector Embeddings and Dual-Dropout LSTM.
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers, 
                 num_sectors, sector_embedding_dim, input_dropout, recurrent_dropout):
        super(HeterogeneityLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_dropout_rate = input_dropout
        self.recurrent_dropout_rate = recurrent_dropout
        
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.sector_embedding = nn.Embedding(num_sectors, sector_embedding_dim)
        self.fc = nn.Linear(hidden_size + sector_embedding_dim, num_classes)
    
    def forward(self, x, sectors):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        h = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            input_t = x[:, t, :]
            input_t = F.dropout(input_t, p=self.input_dropout_rate, training=self.training)
            
            new_h, new_c = [], []
            for i, cell in enumerate(self.lstm_cells):
                h_i, c_i = cell(input_t, (h[i], c[i]))
                h_i = F.dropout(h_i, p=self.recurrent_dropout_rate, training=self.training)
                new_h.append(h_i)
                new_c.append(c_i)
                input_t = h_i
            
            h, c = new_h, new_c
            
        sector_embeds = self.sector_embedding(sectors)
        return self.fc(torch.cat((h[-1], sector_embeds), dim=1))

def get_model(model_architecture, **kwargs):
    """
    Factory function for dual-dropout models.
    """
    if model_architecture in ['base', 'covariates']:
        return LSTMModel(**kwargs)
    elif model_architecture == 'heterogeneity':
        return HeterogeneityLSTMModel(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_architecture}")