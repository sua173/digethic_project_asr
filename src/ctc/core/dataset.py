"""
LibriSpeech dataset implementation for speech recognition
"""

import json
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Any
from .utils import TextProcessor


class LibriSpeechDataset(Dataset):
    """Dataset class for LibriSpeech"""
    
    def __init__(self, manifest_path: str, text_processor: TextProcessor,
                 sample_rate: int = 16000, n_mels: int = 80,
                 hop_length: int = 160, win_length: int = 400,
                 apply_spec_augment: bool = True):
        """
        Args:
            manifest_path: Path to JSONL manifest file
            text_processor: TextProcessor instance
            sample_rate: Sampling rate
            n_mels: Number of mel filter banks
            hop_length: STFT hop length
            win_length: STFT window length
            apply_spec_augment: Whether to apply SpecAugment
        """
        self.text_processor = text_processor
        self.apply_spec_augment = apply_spec_augment
        
        # Load data
        self.data = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        # Setup mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=0.0,
            f_max=sample_rate / 2,
            power=2.0,
            normalized=False
        )
        
        # SpecAugment transforms (data augmentation)
        if apply_spec_augment:
            self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
            self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)
        
        # Convert amplitude to decibels
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        print(f"Loaded {len(self.data)} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spectrogram: (n_mels, time_steps)
            text_ids: (text_length,)
        """
        item = self.data[idx]
        audio_path = item['audio_filepath']
        text = item['text']
        
        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            # Convert to mono (LibriSpeech is usually already mono)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return silence on error
            waveform = torch.zeros(1, 16000)  # 1 second of silence
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        # (1, n_mels, time) -> (n_mels, time)
        mel_spec = mel_spec.squeeze(0)
        
        # Convert to decibels
        mel_spec = self.amplitude_to_db(mel_spec)
        
        # Apply SpecAugment (training only)
        if self.apply_spec_augment:
            # Apply probabilistically
            if torch.rand(1).item() > 0.5:
                mel_spec = self.freq_mask(mel_spec)
            if torch.rand(1).item() > 0.5:
                mel_spec = self.time_mask(mel_spec)
        
        # Convert text to character IDs
        text_ids = torch.tensor(self.text_processor.text_to_ids(text), dtype=torch.long)
        
        return mel_spec, text_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader
    
    Args:
        batch: List of [(spectrogram, text_ids), ...]
    
    Returns:
        Dictionary format batch data
        - spectrograms: (batch_size, n_mels, max_time)
        - texts: (batch_size, max_text_len)
        - spec_lengths: (batch_size,) Original spectrogram lengths
        - text_lengths: (batch_size,) Original text lengths
    """
    spectrograms, texts = zip(*batch)
    
    # Record original lengths
    spec_lengths = torch.tensor([spec.shape[1] for spec in spectrograms], dtype=torch.long)
    text_lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)
    
    # Pad spectrograms
    # Transpose (n_mels, time) -> (time, n_mels) before padding
    spectrograms_transposed = [spec.transpose(0, 1) for spec in spectrograms]
    spectrograms_padded = pad_sequence(spectrograms_transposed, batch_first=True)
    # Transpose back (batch_size, max_time, n_mels) -> (batch_size, n_mels, max_time)
    spectrograms_padded = spectrograms_padded.transpose(1, 2)
    
    # Pad texts
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    
    return {
        'spectrograms': spectrograms_padded,
        'texts': texts_padded,
        'spec_lengths': spec_lengths,
        'text_lengths': text_lengths
    }


def create_data_loaders(train_manifest: str, val_manifest: str,
                       text_processor: TextProcessor,
                       batch_size: int = 8,
                       num_workers: int = 2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation DataLoaders
    
    Args:
        train_manifest: Training manifest file
        val_manifest: Validation manifest file
        text_processor: TextProcessor instance
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = LibriSpeechDataset(
        train_manifest, text_processor, 
        apply_spec_augment=True
    )
    val_dataset = LibriSpeechDataset(
        val_manifest, text_processor,
        apply_spec_augment=False  # No data augmentation for validation
    )
    
    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # CPU mode
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False  # CPU mode
    )
    
    return train_loader, val_loader


def collate_fn_transformer(batch):
    """
    Collate function for Transformer models with Teacher Forcing support
    
    Returns both source sequences (encoder input) and target sequences (decoder input/output)
    for Teacher Forcing training.
    """
    spectrograms = []
    src_texts = []  # Source text IDs (encoder)
    tgt_inputs = []  # Target inputs (decoder input with SOS)
    tgt_outputs = []  # Target outputs (decoder output with EOS)
    spec_lengths = []
    src_lengths = []
    tgt_lengths = []
    
    # batch is a list of tuples (spectrogram, text_ids)
    for spectrogram, text_ids in batch:
        spectrograms.append(spectrogram)
        spec_lengths.append(spectrogram.size(-1))  # Time dimension
        
        # Source sequence (for encoder)
        src_texts.append(text_ids)
        src_lengths.append(len(text_ids))
        
        # For teacher forcing, we need shifted sequences
        # tgt_input: [SOS, text...] (decoder input)
        # tgt_output: [text..., EOS] (expected decoder output)
        
        # Note: We'll add SOS/EOS in the model or training loop
        # based on text processor configuration
        tgt_inputs.append(text_ids)
        tgt_outputs.append(text_ids)
        tgt_lengths.append(len(text_ids))
    
    # Pad sequences
    # For spectrograms, we need to handle 2D tensors (n_mels, time)
    # First, find the max time dimension
    max_spec_len = max(spec.size(-1) for spec in spectrograms)
    n_mels = spectrograms[0].size(0)
    
    # Pad spectrograms manually
    padded_spectrograms = []
    for spec in spectrograms:
        # spec shape: (n_mels, time)
        pad_length = max_spec_len - spec.size(-1)
        if pad_length > 0:
            # Pad on the time dimension (right side)
            padded_spec = torch.nn.functional.pad(spec, (0, pad_length), value=0)
        else:
            padded_spec = spec
        padded_spectrograms.append(padded_spec)
    
    spectrograms = torch.stack(padded_spectrograms)  # (batch, n_mels, time)
    
    # Pad text sequences
    src_texts = nn.utils.rnn.pad_sequence(src_texts, batch_first=True)
    tgt_inputs = nn.utils.rnn.pad_sequence(tgt_inputs, batch_first=True)
    tgt_outputs = nn.utils.rnn.pad_sequence(tgt_outputs, batch_first=True)
    
    # Convert lengths to tensors
    spec_lengths = torch.tensor(spec_lengths, dtype=torch.long)
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)
    tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
    
    return {
        'spectrograms': spectrograms,
        'src_texts': src_texts,
        'tgt_inputs': tgt_inputs,
        'tgt_outputs': tgt_outputs,
        'spec_lengths': spec_lengths,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths
    }


def create_transformer_data_loaders(train_manifest: str, val_manifest: str, 
                                  text_processor, batch_size: int = 8,
                                  num_workers: int = 2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for Transformer model training
    
    Args:
        train_manifest: Training manifest file
        val_manifest: Validation manifest file
        text_processor: TransformerTextProcessor instance
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = LibriSpeechDataset(
        train_manifest, text_processor, 
        apply_spec_augment=True
    )
    val_dataset = LibriSpeechDataset(
        val_manifest, text_processor,
        apply_spec_augment=False
    )
    
    # Create DataLoaders with Transformer collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn_transformer,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_transformer,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader