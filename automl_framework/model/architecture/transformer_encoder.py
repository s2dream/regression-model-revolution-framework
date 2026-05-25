import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBasedEncoder(nn.Module):
    """
    Standard Transformer Encoder wrapping PyTorch's nn.TransformerEncoder.
    Can be used independently to encode sequence data.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional embeddings up to maximum length of 1000
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_encoder, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor): optional padding mask of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: encoded representations of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        x = self.input_projection(x)
        
        # Add positional encoding
        pe = self.pos_encoder[:, :seq_len, :]
        x = x + pe
        
        # Pass through transformer encoder
        if mask is not None:
            # PyTorch src_key_padding_mask expects boolean/float where True/1 indicates padding to ignore
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
            
        return x


class TransformerBasedRegression(nn.Module):
    """
    Transformer-based Regression model.
    Processes sequence data, pools the sequence vectors, and applies a regression head.
    Can predict either a single scalar or a distribution (mean and positive variance).
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "mean",
        predict_distribution: bool = False,
        activation: str = "relu"
    ):
        super().__init__()
        self.encoder = TransformerBasedEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        self.pooling = pooling
        self.predict_distribution = predict_distribution
        
        # Non-linear projection representation layer for Regression Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        if self.predict_distribution:
            # Predict mean and logvar separately
            self.mean_head = nn.Linear(d_model, 1)
            self.logvar_head = nn.Linear(d_model, 1)
        else:
            # Predict single target scalar
            self.output_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): sequence data input of shape (batch_size, seq_len, input_dim)
            mask (torch.Tensor): optional padding mask of shape (batch_size, seq_len), where True/1 indicates padding
            
        Returns:
            If predict_distribution is False:
                torch.Tensor: scalar predictions of shape (batch_size, 1)
            If predict_distribution is True:
                tuple: (mean, variance) where both are torch.Tensor of shape (batch_size, 1) and variance > 0
        """ 
        batch_size, seq_len, _ = x.size()
        
        # 1. Encode sequence using TransformerBasedEncoder
        encoded = self.encoder(x, mask=mask)  # (batch_size, seq_len, d_model)
        
        # 2. Pool sequence vectors
        if self.pooling == "mean":
            if mask is not None:
                # Masked mean: mask has True for padding, active is ~mask
                active_mask = (~mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
                pooled = (encoded * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-9)
            else:
                pooled = encoded.mean(dim=1)
        elif self.pooling == "max":
            if mask is not None:
                # Fill padded values with small value to prevent max selecting them
                masked_encoded = encoded.clone()
                masked_encoded[mask] = -1e9
                pooled = masked_encoded.max(dim=1)[0]
            else:
                pooled = encoded.max(dim=1)[0]
        elif self.pooling == "last":
            if mask is not None:
                # Use the last active sequence index per batch
                lengths = (~mask).sum(dim=1) - 1
                lengths = torch.clamp(lengths, min=0)
                pooled = encoded[torch.arange(batch_size), lengths]
            else:
                pooled = encoded[:, -1, :]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")
            
        # 3. Pass through projection layer
        feat = self.fc(pooled)
        
        # 4. Predict outputs
        if self.predict_distribution:
            mean = self.mean_head(feat)
            logvar = self.logvar_head(feat)
            # Clamp logvar to prevent numerical issues (underflow/overflow)
            logvar = torch.clamp(logvar, min=-20.0, max=10.0)
            variance = torch.exp(logvar)
            return mean, variance
        else:
            output = self.output_head(feat)
            return output
