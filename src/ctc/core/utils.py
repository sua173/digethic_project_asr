"""
Utility functions for speech recognition project
"""

import string
from typing import List, Dict


class TextProcessor:
    """Class for converting text to character IDs"""
    
    def __init__(self):
        # English uppercase letters, space, and blank token (for CTC)
        self.chars = [' '] + list(string.ascii_uppercase)
        self.blank_id = len(self.chars)  # ID for CTC blank token
        
        # Character -> ID dictionary
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        # ID -> Character dictionary
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}
        
        self.vocab_size = len(self.chars) + 1  # Including blank token
    
    def text_to_ids(self, text: str) -> List[int]:
        """
        Convert text to character ID list
        
        Args:
            text: Input text (uppercase letters with spaces)
        
        Returns:
            List of character IDs
        """
        ids = []
        for char in text:
            if char in self.char_to_id:
                ids.append(self.char_to_id[char])
            # Ignore unknown characters (LibriSpeech is normalized so this rarely happens)
        return ids
    
    def ids_to_text(self, ids: List[int]) -> str:
        """
        Convert character ID list to text
        
        Args:
            ids: List of character IDs
        
        Returns:
            Reconstructed text
        """
        chars = []
        for id in ids:
            if id in self.id_to_char:
                chars.append(self.id_to_char[id])
        return ''.join(chars)
    
    def decode_ctc(self, ids: List[int]) -> str:
        """
        Decode CTC output (remove duplicates and blank tokens)
        
        Args:
            ids: CTC model output ID sequence
        
        Returns:
            Decoded text
        """
        # Remove consecutive duplicate characters
        prev_id = None
        decoded_ids = []
        
        for id in ids:
            if id != prev_id and id != self.blank_id:
                decoded_ids.append(id)
            prev_id = id
        
        return self.ids_to_text(decoded_ids)


class TransformerTextProcessor(TextProcessor):
    """Text processor for Transformer models with special tokens"""
    
    def __init__(self):
        super().__init__()
        
        # Add special tokens for Transformer
        self.pad_id = self.vocab_size  # PAD token
        self.sos_id = self.vocab_size + 1  # Start of Sequence
        self.eos_id = self.vocab_size + 2  # End of Sequence
        
        # Update vocab size to include special tokens
        self.vocab_size = self.vocab_size + 3
        
        # Special token strings for display
        self.special_tokens = {
            self.blank_id: '<BLANK>',
            self.pad_id: '<PAD>',
            self.sos_id: '<SOS>',
            self.eos_id: '<EOS>'
        }
    
    def text_to_ids_with_eos(self, text: str) -> List[int]:
        """Convert text to IDs and append EOS token"""
        ids = self.text_to_ids(text)
        return ids + [self.eos_id]
    
    def text_to_ids_with_special(self, text: str) -> List[int]:
        """Convert text to IDs with SOS and EOS tokens"""
        ids = self.text_to_ids(text)
        return [self.sos_id] + ids + [self.eos_id]
    
    def ids_to_text(self, ids: List[int]) -> str:
        """
        Convert ID list to text, automatically removing special tokens
        
        Args:
            ids: List of character IDs
        
        Returns:
            Reconstructed text
        """
        # Always filter out special tokens for TransformerTextProcessor
        ids = [id for id in ids if id not in [self.pad_id, self.sos_id, self.eos_id, self.blank_id]]
        
        chars = []
        for id in ids:
            if id < len(self.chars):
                chars.append(self.id_to_char[id])
        
        return ''.join(chars)
    
    def ids_to_text_with_special(self, ids: List[int]) -> str:
        """
        Convert ID list to text including special tokens
        
        Args:
            ids: List of character IDs
        
        Returns:
            Reconstructed text with special tokens
        """
        chars = []
        for id in ids:
            if id in self.special_tokens:
                chars.append(self.special_tokens[id])
            elif id < len(self.chars):
                chars.append(self.id_to_char[id])
        
        return ''.join(chars)


def print_model_info(model):
    """Print model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")  # Assuming float32


def setup_device(force_cpu: bool = False):
    """Setup optimal device"""
    import torch
    
    if force_cpu:
        # Force CPU usage (e.g., for CTC loss)
        device = torch.device("cpu")
        print("Using CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        # MPS is available for Transformer models
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def init_weights(m):
    """Initialize model weights using Xavier initialization"""
    import torch.nn as nn
    
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)