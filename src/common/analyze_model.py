#!/usr/bin/env python3
"""
Analyze trained model checkpoint and display detailed information.

Usage:
    python src/common/analyze_model.py [checkpoint_path]
    
Examples:
    python src/common/analyze_model.py generated/checkpoints/ctc_20240101_120000/best_ctc_model.pt
    python src/common/analyze_model.py generated/checkpoints/attention_20240101_120000/best_model.pt
    
Default: Searches for the latest model in checkpoints directory
"""

import torch
import os
import sys
import glob
from datetime import datetime
import argparse

# Add project root to path for loading checkpoints with custom classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def find_latest_checkpoint(base_dir='generated/checkpoints'):
    """Find the latest checkpoint file in the checkpoints directory."""
    # Search patterns for different model types
    patterns = [
        os.path.join(base_dir, '**/best_ctc_model.pt'),
        os.path.join(base_dir, '**/best_transformer_model.pt'),
        os.path.join(base_dir, '**/best_model.pt'),  # Legacy support
        os.path.join(base_dir, '**/latest_checkpoint.pt')
    ]
    
    checkpoint_files = []
    for pattern in patterns:
        checkpoint_files.extend(glob.glob(pattern, recursive=True))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and return the latest
    return max(checkpoint_files, key=os.path.getmtime)


def detect_model_type(checkpoint):
    """Detect the model type from checkpoint contents."""
    if 'model_state_dict' in checkpoint:
        model_keys = list(checkpoint['model_state_dict'].keys())
        
        # Check for CTC-specific components
        if any('ctc' in key.lower() for key in model_keys):
            return 'CTC'
        if any('lstm' in key.lower() for key in model_keys):
            return 'CTC (LSTM-based)'
            
        # Check for Transformer-specific components
        if any('transformer' in key.lower() for key in model_keys):
            return 'Transformer'
        if any('attention' in key.lower() for key in model_keys):
            return 'Transformer (Attention-based)'
        if any('encoder.layers' in key for key in model_keys):
            return 'Transformer'
    
    # Try to infer from file path
    if 'checkpoint_path' in checkpoint:
        path = checkpoint['checkpoint_path']
        if 'ctc' in path.lower():
            return 'CTC'
        if 'transformer' in path.lower():
            return 'Transformer'
    
    return 'Unknown'


def analyze_checkpoint(checkpoint_path=None):
    """Analyze a saved model checkpoint and print detailed information."""
    
    # If no checkpoint path provided, find the latest
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("Error: No checkpoint files found in checkpoints directory")
            return
        print(f"Found latest checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print('=' * 50)
    print('CHECKPOINT ANALYSIS')
    print('=' * 50)
    
    # Detect model type
    model_type = detect_model_type(checkpoint)
    
    # File information
    print(f'\n=== File Information ===')
    print(f'File path: {checkpoint_path}')
    print(f'File size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB')
    print(f'Last modified: {datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Model type: {model_type}')
    
    # Checkpoint contents
    print(f'\n=== Checkpoint Contents ===')
    print(f'Available keys: {list(checkpoint.keys())}')
    
    # Training state
    print(f'\n=== Training State ===')
    print(f'Epoch: {checkpoint.get("epoch", "N/A")}')
    
    # Model-specific metrics
    if model_type.startswith('CTC'):
        if 'cer' in checkpoint:
            print(f'CER (Character Error Rate): {checkpoint["cer"]:.2%}')
        if 'wer' in checkpoint:
            print(f'WER (Word Error Rate): {checkpoint["wer"]:.2%}')
    elif model_type.startswith('Transformer'):
        # Transformer-specific metrics (to be added)
        if 'bleu' in checkpoint:
            print(f'BLEU Score: {checkpoint["bleu"]:.2f}')
        if 'perplexity' in checkpoint:
            print(f'Perplexity: {checkpoint["perplexity"]:.2f}')
        # Still show CER/WER if available
        if 'cer' in checkpoint:
            print(f'CER (Character Error Rate): {checkpoint["cer"]:.2%}')
        if 'wer' in checkpoint:
            print(f'WER (Word Error Rate): {checkpoint["wer"]:.2%}')
    
    # Model architecture analysis
    if 'model_state_dict' in checkpoint:
        print(f'\n=== Model Architecture ===')
        
        # Detailed layer information
        print('\nLayer Details:')
        total_params = 0
        layer_params = {}
        
        for name, param in checkpoint['model_state_dict'].items():
            num_params = param.numel()
            total_params += num_params
            
            # Group by main layer
            main_layer = name.split('.')[0]
            if main_layer not in layer_params:
                layer_params[main_layer] = 0
            layer_params[main_layer] += num_params
            
            # Print detailed info for weights (not biases or batch norm stats)
            if 'weight' in name and not any(x in name for x in ['running_mean', 'running_var']):
                print(f'  {name}: {list(param.shape)} ({num_params:,} params)')
        
        print(f'\n=== Model Summary ===')
        print(f'Total parameters: {total_params:,}')
        print(f'Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)')
        
        print(f'\n=== Layer-wise Parameter Distribution ===')
        for layer, params in sorted(layer_params.items()):
            percentage = params / total_params * 100
            print(f'{layer:20s}: {params:>10,} params ({percentage:>5.1f}%)')
    
    # Text processor information
    if 'text_processor' in checkpoint:
        print(f'\n=== Text Processor Configuration ===')
        tp = checkpoint['text_processor']
        if hasattr(tp, 'char_to_id'):
            print(f'Vocabulary size: {len(tp.char_to_id)}')
            print(f'Characters: {list(tp.char_to_id.keys())}')
            print(f'Blank token ID: {tp.blank_id}')
    
    # Optimizer state
    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print(f'\n=== Optimizer Configuration ===')
        if 'param_groups' in opt_state and opt_state['param_groups']:
            pg = opt_state['param_groups'][0]
            print(f'Learning rate: {pg.get("lr", "N/A")}')
            print(f'Weight decay: {pg.get("weight_decay", 0)}')
            print(f'Betas: {pg.get("betas", "N/A")}')
            print(f'Eps: {pg.get("eps", "N/A")}')
    
    # Additional training info if available
    print(f'\n=== Additional Information ===')
    for key in checkpoint.keys():
        if key not in ['model_state_dict', 'optimizer_state_dict', 'text_processor', 'epoch', 'cer', 'wer']:
            print(f'{key}: {checkpoint[key]}')
    
    # Model architecture reconstruction (if possible)
    if 'model_state_dict' in checkpoint:
        print(f'\n=== Inferred Architecture ===')
        # CNN layers
        cnn_layers = [k for k in checkpoint['model_state_dict'].keys() if 'cnn_encoder' in k]
        lstm_layers = [k for k in checkpoint['model_state_dict'].keys() if 'lstm' in k]
        output_layers = [k for k in checkpoint['model_state_dict'].keys() if 'output_projection' in k]
        
        if cnn_layers:
            conv_count = sum(1 for k in cnn_layers if 'conv_layers' in k and 'weight' in k and 'running' not in k)
            print(f'CNN Encoder: {conv_count} convolutional layers')
        
        if lstm_layers:
            lstm_count = sum(1 for k in lstm_layers if 'weight_ih_l' in k and not 'reverse' in k)
            print(f'LSTM: {lstm_count} layers (bidirectional)')
        
        if output_layers:
            output_shape = checkpoint['model_state_dict']['output_projection.weight'].shape
            print(f'Output projection: {output_shape[1]} -> {output_shape[0]} (vocab size)')


def main():
    parser = argparse.ArgumentParser(description='Analyze trained model checkpoint')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='Path to checkpoint file (default: searches for latest checkpoint)')
    parser.add_argument('--dir', default='generated/checkpoints',
                        help='Base directory to search for checkpoints (default: checkpoints)')
    
    args = parser.parse_args()
    
    # If a specific checkpoint is provided, use it
    if args.checkpoint:
        analyze_checkpoint(args.checkpoint)
    else:
        # Otherwise, search in the specified directory
        checkpoint_path = find_latest_checkpoint(args.dir)
        if checkpoint_path:
            analyze_checkpoint(checkpoint_path)
        else:
            print(f"No checkpoint files found in {args.dir}")


if __name__ == '__main__':
    main()