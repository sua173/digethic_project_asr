#!/usr/bin/env python3
"""
Inference script for trained CTC model

Usage:
    python src/ctc/inference.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt --audio path/to/audio.flac
    python src/ctc/inference.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt --audio path/to/audio.flac --output transcription.txt
"""

import os
import sys
import torch
import torchaudio
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ctc.model import create_ctc_model
from src.ctc.core.utils import setup_device


class CTCInference:
    """Class for generating text from audio files using CTC model"""
    
    def __init__(self, checkpoint_path: str, device: torch.device = None):
        """Initialize CTC inference
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        # Setup device
        self.device = device if device is not None else setup_device(force_cpu=True)
        
        # Load checkpoint
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get text processor
        self.text_processor = checkpoint['text_processor']
        
        # Create and load model with correct parameters
        self.model = create_ctc_model(
            vocab_size=self.text_processor.vocab_size,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully. Best CER: {checkpoint.get('cer', 'N/A')}")
        
        # Audio preprocessing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,
            win_length=400,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
            normalized=False
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """Preprocess audio file"""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = mel_spec.squeeze(0)  # Remove batch dimension
        
        # Convert to dB
        mel_spec = self.amplitude_to_db(mel_spec)
        
        return mel_spec
    
    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Preprocess
        print(f"Processing audio: {audio_path}")
        mel_spec = self.preprocess_audio(audio_path)
        
        # Add batch dimension
        spectrograms = mel_spec.unsqueeze(0).to(self.device)
        spec_lengths = torch.tensor([mel_spec.shape[1]], device=self.device)
        
        print(f"Feature shape: {spectrograms.shape}")
        print(f"Decoding with CTC greedy decoding...")
        
        # Inference
        with torch.no_grad():
            predictions = self.model.decode_greedy(spectrograms, spec_lengths, self.text_processor)
        
        return predictions[0]
    
    def transcribe_batch(self, audio_paths: list) -> list:
        """Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of transcribed texts
        """
        transcriptions = []
        for audio_path in audio_paths:
            transcription = self.transcribe(audio_path)
            transcriptions.append(transcription)
        return transcriptions


def main():
    parser = argparse.ArgumentParser(description='Transcribe audio using CTC model')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to audio file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str,
                       help='Output file to save transcription')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    try:
        # Initialize inference
        inferencer = CTCInference(args.checkpoint)
        
        # Transcribe
        transcription = inferencer.transcribe(args.audio)
        
        # Display result
        print("\n" + "="*50)
        print("TRANSCRIPTION:")
        print("="*50)
        print(transcription)
        print("="*50)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(transcription)
            print(f"\nTranscription saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()