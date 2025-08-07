#!/usr/bin/env python3
"""
Test script for inference components
Tests text processing, model loading, and inference functionality

Usage:
    python src/ctc/tests/test_inference.py
    python src/ctc/tests/test_inference.py --skip-audio
    python src/ctc/tests/test_inference.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt
"""

import os
import sys
import argparse
import warnings
import torch
import torchaudio
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Suppress PyTorch nested tensor warnings
warnings.filterwarnings("ignore", message=".*nested tensor.*prototype.*", category=UserWarning)
from src.ctc.core.utils import TextProcessor, TransformerTextProcessor
from src.ctc.model import create_ctc_model
# from src.models.transformer.model import create_transformer_model  # TODO: implement


def test_text_processors():
    """Test both TextProcessor classes"""
    print("=== Testing Text Processors ===\n")
    
    # Test CTC TextProcessor
    print("1. Testing CTC TextProcessor:")
    tp_ctc = TextProcessor()
    print(f"   Vocabulary size: {tp_ctc.vocab_size}")
    print(f"   Blank ID: {tp_ctc.blank_id}")
    
    # Test text encoding/decoding
    test_text = "HELLO WORLD"
    ids = tp_ctc.text_to_ids(test_text)
    decoded = tp_ctc.ids_to_text(ids)
    print(f"   Text conversion: '{test_text}' -> {ids} -> '{decoded}'")
    
    # Test CTC decoding
    ctc_output = [8, 8, 5, 12, 12, 15, 27, 27, 23, 15, 18, 12, 4]
    ctc_decoded = tp_ctc.decode_ctc(ctc_output)
    print(f"   CTC decode: {ctc_output} -> '{ctc_decoded}'")
    
    # Test Transformer TextProcessor
    print("\n2. Testing Transformer TextProcessor:")
    tp_trans = TransformerTextProcessor()
    print(f"   Vocabulary size: {tp_trans.vocab_size}")
    print(f"   Special tokens: PAD={tp_trans.pad_id}, SOS={tp_trans.sos_id}, EOS={tp_trans.eos_id}")
    
    # Test with special tokens
    ids_with_special = tp_trans.text_to_ids_with_special(test_text)
    decoded_no_special = tp_trans.ids_to_text(ids_with_special)
    print(f"   With special tokens: '{test_text}' -> {ids_with_special}")
    print(f"   Auto-removes special: {ids_with_special} -> '{decoded_no_special}'")
    
    print("\n✅ Text processor tests passed!\n")


def test_model_creation():
    """Test model creation and basic functionality"""
    print("=== Testing Model Creation ===\n")
    
    # Test CTC model
    print("1. Creating CTC model:")
    ctc_model = create_ctc_model(vocab_size=28, hidden_dim=128, num_layers=2)
    param_count = sum(p.numel() for p in ctc_model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Model size: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    batch_size, n_mels, time = 2, 80, 100
    dummy_input = torch.randn(batch_size, n_mels, time)
    dummy_lengths = torch.tensor([time, time])
    
    log_probs, out_lengths = ctc_model(dummy_input, dummy_lengths)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {log_probs.shape}")
    print(f"   Output lengths: {out_lengths}")
    
    # Test Transformer model
    print("\n2. Creating Transformer model:")
    trans_model = create_transformer_model(
        vocab_size=31, d_model=256, nhead=4, 
        num_encoder_layers=2, num_decoder_layers=2
    )
    param_count = sum(p.numel() for p in trans_model.parameters())
    print(f"   Parameters: {param_count:,}")
    print(f"   Model size: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    # Test encoder
    encoder_out = trans_model.encode_audio(dummy_input, dummy_lengths)
    print(f"   Encoder output shape: {encoder_out.shape}")
    
    print("\n✅ Model creation tests passed!\n")


def test_checkpoint_loading(checkpoint_path=None):
    """Test checkpoint loading and compatibility"""
    print("=== Testing Checkpoint Loading ===\n")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Testing specific checkpoint: {checkpoint_path}")
        checkpoints = [checkpoint_path]
    else:
        # Find latest checkpoints
        print("Finding latest checkpoints...")
        checkpoints = []
        
        if os.path.exists('checkpoints'):
            for run_dir in sorted(Path('checkpoints').iterdir(), reverse=True):
                if run_dir.is_dir():
                    # Check for CTC model
                    ctc_path = run_dir / 'best_ctc_model.pt'
                    if ctc_path.exists():
                        checkpoints.append(str(ctc_path))
                        break
                    
                    # Check for Transformer model
                    trans_path = run_dir / 'best_transformer_model.pt'
                    if trans_path.exists():
                        checkpoints.append(str(trans_path))
                        break
    
    if not checkpoints:
        print("No checkpoints found. Train a model first.")
        return
    
    for ckpt_path in checkpoints:
        print(f"\nLoading: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            # Check contents
            print(f"  Keys: {list(checkpoint.keys())}")
            
            # Check text processor
            if 'text_processor' in checkpoint:
                tp = checkpoint['text_processor']
                print(f"  TextProcessor type: {type(tp).__name__}")
                print(f"  Has decode_ctc: {hasattr(tp, 'decode_ctc')}")
                print(f"  Vocab size: {tp.vocab_size}")
            
            # Check metrics
            if 'cer' in checkpoint:
                print(f"  Best CER: {checkpoint['cer']:.4f}")
            if 'wer' in checkpoint:
                print(f"  Best WER: {checkpoint['wer']:.4f}")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            
            # Try to determine model type
            state_dict = checkpoint.get('model_state_dict', {})
            if any('cnn_encoder' in k for k in state_dict.keys()):
                if any('transformer' in k for k in state_dict.keys()):
                    print("  Model type: Transformer")
                else:
                    print("  Model type: CTC")
            
            print("  ✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"  ❌ Error loading checkpoint: {e}")
    
    print("\n✅ Checkpoint tests completed!\n")


def test_audio_processing():
    """Test audio loading and preprocessing"""
    print("=== Testing Audio Processing ===\n")
    
    # Find a sample audio file
    audio_paths = [
        "data/dev-clean/1272/128104/1272-128104-0000.flac",
        "data/test-clean/1089/134686/1089-134686-0000.flac",
        "test_subset/1272-128104-0000.flac"
    ]
    
    audio_path = None
    for path in audio_paths:
        if os.path.exists(path):
            audio_path = path
            break
    
    if not audio_path:
        print("No audio file found for testing.")
        return
    
    print(f"Testing with: {audio_path}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"  Original shape: {waveform.shape}")
    print(f"  Sample rate: {sample_rate} Hz")
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        print(f"  Resampled to 16kHz")
    
    # Create mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        win_length=400
    )
    
    mel_spec = mel_transform(waveform)
    print(f"  Mel spectrogram shape: {mel_spec.shape}")
    
    # Log transform
    log_mel = torch.log(mel_spec + 1e-9)
    print(f"  Log mel shape: {log_mel.shape}")
    
    print("\n✅ Audio processing tests passed!\n")


def main():
    parser = argparse.ArgumentParser(description='Test inference components')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint to test')
    parser.add_argument('--skip-audio', action='store_true', help='Skip audio processing tests')
    parser.add_argument('--skip-models', action='store_true', help='Skip model creation tests')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Speech Recognition Inference Component Tests")
    print("=" * 50)
    print()
    
    # Run tests
    test_text_processors()
    
    if not args.skip_models:
        test_model_creation()
    
    test_checkpoint_loading(args.checkpoint)
    
    if not args.skip_audio:
        test_audio_processing()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()