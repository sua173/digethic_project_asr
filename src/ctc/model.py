"""
Speech recognition model implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNEncoder(nn.Module):
    """CNN encoder for acoustic feature extraction"""
    
    def __init__(self, input_dim: int = 80, hidden_dim: int = 256):
        """
        Args:
            input_dim: Input feature dimension (number of mel filter banks)
            hidden_dim: CNN output dimension
        """
        super().__init__()
        
        # CNN layers with batch normalization
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Compress time axis by 1/2
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Removed second MaxPool2d to reduce compression from 1/4 to 1/2
        )
        
        # Calculate CNN output dimension
        # After 1 MaxPool2d layer: input_dim -> input_dim // 2
        cnn_output_dim = (input_dim // 2) * 64
        
        # Projection layer to desired hidden dimension
        self.projection = nn.Linear(cnn_output_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, n_mels, time) 

        Returns:
            features: (batch_size, time_compressed, hidden_dim)
            lengths: (batch_size,) Compressed time lengths
        """
        batch_size, n_mels, time = x.shape
        
        # Add channel dimension: (batch_size, 1, n_mels, time)
        x = x.unsqueeze(1)
        
        # CNN forward
        x = self.conv_layers(x)  # (batch_size, 64, n_mels//4, time//4)
        
        # Reshape for LSTM: (batch_size, time//4, 64 * n_mels//4)
        batch_size, channels, freq, time_compressed = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch_size, time//4, 64, n_mels//4)
        x = x.contiguous().view(batch_size, time_compressed, channels * freq)
        
        # Project to hidden dimension
        features = self.projection(x)  # (batch_size, time//4, hidden_dim)
        
        # Calculate compressed lengths
        lengths = torch.full((batch_size,), time_compressed, dtype=torch.long, device=x.device)
        
        return features, lengths


class CTCModel(nn.Module):
    """End-to-End model using CTC loss"""
    
    def __init__(self, vocab_size: int, input_dim: int = 80, 
                 hidden_dim: int = 256, num_layers: int = 4, dropout: float = 0.1):
        """
        Args:
            vocab_size: Vocabulary size (including blank token)
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # CNN encoder
        self.cnn_encoder = CNNEncoder(input_dim, hidden_dim)
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)  # *2 for bidirectional
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spectrograms: torch.Tensor, spec_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectrograms: (batch_size, n_mels, time)
            spec_lengths: (batch_size,) Original time lengths for each sample

        Returns:
            log_probs: (batch_size, time_compressed, vocab_size) log probabilities
            output_lengths: (batch_size,) Output sequence lengths
        """
        # CNN encoding
        features, cnn_lengths = self.cnn_encoder(spectrograms)
        
        # Calculate actual lengths after CNN compression
        compression_factor = 2  # Compressed by 1/2 with 1 MaxPool2d layer
        output_lengths = (spec_lengths.float() / compression_factor).long()
        output_lengths = torch.clamp(output_lengths, min=1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(features)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output projection
        logits = self.output_projection(lstm_out)  # (batch_size, time, vocab_size)
        
        # Log softmax for CTC
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs, output_lengths
    
    def decode_greedy(self, spectrograms: torch.Tensor, spec_lengths: torch.Tensor,
                     text_processor) -> list:
        """
        Greedy decoding
        
        Args:
            spectrograms: (batch_size, n_mels, time)
            spec_lengths: (batch_size,)
            text_processor: TextProcessor instance
            
        Returns:
            decoded_texts: List of decoded texts
        """
        self.eval()
        with torch.no_grad():
            log_probs, output_lengths = self.forward(spectrograms, spec_lengths)
            
            # Select the class with highest probability
            predictions = torch.argmax(log_probs, dim=-1)  # (batch_size, time)
            
            decoded_texts = []
            for i in range(predictions.shape[0]):
                pred_seq = predictions[i, :output_lengths[i]].cpu().tolist()
                decoded_text = text_processor.decode_ctc(pred_seq)
                decoded_texts.append(decoded_text)
                
        return decoded_texts


def create_ctc_model(vocab_size: int, **kwargs) -> CTCModel:
    """Factory function to create CTC model"""
    return CTCModel(vocab_size, **kwargs)


# Test function
def test_model():
    """Test model functionality"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from src.common.utils import TextProcessor, print_model_info
    
    # Test parameters
    batch_size = 2
    n_mels = 80
    time_steps = 500
    
    # Text processor
    text_processor = TextProcessor()
    vocab_size = text_processor.vocab_size
    
    # Create model
    model = create_ctc_model(vocab_size, hidden_dim=256, num_layers=3)
    print_model_info(model)
    
    # Test input
    spectrograms = torch.randn(batch_size, n_mels, time_steps)
    spec_lengths = torch.tensor([time_steps, time_steps // 2])
    
    print(f"\nInput shape: {spectrograms.shape}")
    print(f"Input lengths: {spec_lengths}")
    
    # Forward pass
    log_probs, output_lengths = model(spectrograms, spec_lengths)
    print(f"Output log_probs shape: {log_probs.shape}")
    print(f"Output lengths: {output_lengths}")
    
    # Greedy decoding test
    decoded = model.decode_greedy(spectrograms, spec_lengths, text_processor)
    print(f"Decoded texts: {decoded}")
    
    print("✅ Model test passed!")


if __name__ == "__main__":
    test_model()