#!/usr/bin/env python3
"""
Unified script to create train, validation, and test datasets from LibriSpeech

This script can:
1. Create training subset from train-clean-100 (without splitting)
2. Create validation subset from dev-clean
3. Create test manifest from test-clean

Usage examples:
    # Create 10-hour training subset
    python src/ctc/tools/create_dataset.py --mode train --duration 36000 --output generated/subset_10h
    
    # Create 1-hour validation subset from dev-clean
    python src/ctc/tools/create_dataset.py --mode val --duration 3600 --output generated/dev_subset
    
    # Create test manifest from entire test-clean
    python src/ctc/tools/create_dataset.py --mode test --output generated/test_dataset/test_manifest.jsonl
    
    # Create all datasets at once
    python src/ctc/tools/create_dataset.py --mode all
"""

import os
import random
import argparse
import json
from pathlib import Path
from typing import List, Tuple
import torchaudio
from tqdm import tqdm


def get_audio_duration(filepath: str) -> float:
    """Get audio file duration in seconds"""
    try:
        info = torchaudio.info(filepath)
        return info.num_frames / info.sample_rate
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0


def collect_audio_files(data_dir: str, include_file_id: bool = False) -> List[Tuple]:
    """
    Collect audio file and transcript pairs from data directory
    
    Returns: 
        - [(audio_path, transcript), ...] if include_file_id is False
        - [(audio_path, transcript, file_id), ...] if include_file_id is True
    """
    files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Scanning directory: {data_dir}")
    
    # Search for all .trans.txt files
    trans_files = list(data_path.rglob("*.trans.txt"))
    print(f"Found {len(trans_files)} transcript files")
    
    for trans_file in trans_files:
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Separate file ID and text
                parts = line.split(' ', 1)
                if len(parts) < 2:
                    continue
                
                file_id, transcript = parts
                # Build corresponding audio file path
                audio_file = trans_file.parent / f"{file_id}.flac"
                
                if audio_file.exists():
                    if include_file_id:
                        files.append((str(audio_file), transcript, file_id))
                    else:
                        files.append((str(audio_file), transcript))
                else:
                    print(f"Warning: Audio file not found: {audio_file}")
    
    return files


def create_subset(files: List[Tuple[str, str]], target_duration: int, 
                 output_dir: str, manifest_name: str = "train_manifest.jsonl") -> None:
    """
    Create subset with specified duration
    
    Args:
        files: List of [(audio_path, transcript), ...]
        target_duration: Target duration in seconds (0 means all files)
        output_dir: Output directory
        manifest_name: Name of the manifest file
    """
    # Random shuffle for reproducibility
    random.seed(42)
    random.shuffle(files)
    
    # Create subset while calculating audio file durations
    subset_files = []
    total_duration = 0.0
    
    print("Calculating audio durations and creating subset...")
    for i, (audio_path, transcript) in enumerate(tqdm(files, desc="Processing files")):
        duration = get_audio_duration(audio_path)
        if duration > 0:
            subset_files.append((audio_path, transcript, duration))
            total_duration += duration
            
            # If target_duration is 0, include all files
            if target_duration > 0 and total_duration >= target_duration:
                print(f"\nTarget duration reached: {total_duration:.1f}s with {len(subset_files)} files")
                break
    
    if target_duration > 0 and total_duration < target_duration:
        print(f"Warning: Could not reach target duration. Got {total_duration:.1f}s")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    manifest = []
    for audio_path, transcript, duration in subset_files:
        manifest.append({
            "audio_filepath": audio_path,
            "text": transcript,
            "duration": duration
        })
    
    manifest_file = output_path / manifest_name
    with open(manifest_file, 'w', encoding='utf-8') as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save statistics
    stats = {
        "total_files": len(subset_files),
        "total_duration_seconds": total_duration,
        "total_duration_hours": total_duration / 3600,
        "average_duration": total_duration / len(subset_files) if subset_files else 0,
        "target_duration": target_duration,
        "manifest_file": str(manifest_file)
    }
    
    stats_file = output_path / "subset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSubset created successfully!")
    print(f"Files: {len(subset_files)}")
    print(f"Duration: {total_duration/3600:.2f} hours ({total_duration:.1f} seconds)")
    print(f"Output: {output_dir}/{manifest_name}")


def create_test_manifest(data_dir: str, output_file: str) -> None:
    """Create manifest for entire test dataset"""
    # Collect all files with file IDs
    files = collect_audio_files(data_dir, include_file_id=True)
    print(f"Found {len(files)} audio-transcript pairs")
    
    if not files:
        print("Error: No audio files found!")
        return
    
    # Create manifest
    total_duration = 0
    valid_count = 0
    
    print(f"\nCreating manifest file: {output_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for audio_path, transcript, file_id in tqdm(files, desc="Processing files"):
            duration = get_audio_duration(audio_path)
            
            if duration > 0:
                entry = {
                    "audio_filepath": audio_path,
                    "text": transcript,
                    "duration": duration,
                    "file_id": file_id
                }
                f.write(json.dumps(entry) + '\n')
                total_duration += duration
                valid_count += 1
    
    # Print summary
    print("\n" + "="*50)
    print("TEST MANIFEST CREATION SUMMARY")
    print("="*50)
    print(f"Total files processed: {len(files)}")
    print(f"Valid files in manifest: {valid_count}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average duration: {total_duration/valid_count:.2f} seconds per file")
    print(f"Output file: {output_file}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Create LibriSpeech datasets")
    parser.add_argument("--mode", choices=['train', 'val', 'test', 'all'], required=True,
                       help="Dataset creation mode")
    parser.add_argument("--data_dir", type=str,
                       help="Path to data directory (default: auto-selected based on mode)")
    parser.add_argument("--duration", type=int, default=0,
                       help="Target duration in seconds (0 = all files)")
    parser.add_argument("--output", type=str,
                       help="Output directory/file (default: auto-generated based on mode)")
    
    args = parser.parse_args()
    
    # Handle 'all' mode
    if args.mode == 'all':
        print("Creating all datasets...")
        # Create 10-hour training subset
        print("\n1. Creating training subset (10 hours)...")
        train_files = collect_audio_files("data/train-clean-100")
        create_subset(train_files, 36000, "generated/subset_10h", "train_manifest.jsonl")
        
        # Create 1-hour validation subset
        print("\n2. Creating validation subset (1 hour)...")
        val_files = collect_audio_files("data/dev-clean")
        create_subset(val_files, 3600, "generated/dev_subset", "val_manifest.jsonl")
        
        # Create test manifest
        print("\n3. Creating test manifest...")
        os.makedirs("test_dataset", exist_ok=True)
        create_test_manifest("data/test-clean", "generated/test_dataset/test_manifest.jsonl")
        
        print("\nAll datasets created successfully!")
        return
    
    # Handle individual modes
    if args.mode == 'train':
        data_dir = args.data_dir or "data/train-clean-100"
        output_dir = args.output or "generated/subset_10h"
        duration = args.duration or 36000  # Default 10 hours
        
        print(f"Creating training subset from {data_dir}")
        print(f"Target duration: {duration/3600:.1f} hours")
        
        files = collect_audio_files(data_dir)
        create_subset(files, duration, output_dir, "train_manifest.jsonl")
        
    elif args.mode == 'val':
        data_dir = args.data_dir or "data/dev-clean"
        output_dir = args.output or "generated/dev_subset"
        duration = args.duration or 3600  # Default 1 hour
        
        print(f"Creating validation subset from {data_dir}")
        print(f"Target duration: {duration/3600:.1f} hours")
        
        files = collect_audio_files(data_dir)
        create_subset(files, duration, output_dir, "val_manifest.jsonl")
        
    elif args.mode == 'test':
        data_dir = args.data_dir or "data/test-clean"
        output_file = args.output or "generated/test_dataset/test_manifest.jsonl"
        
        # Create directory if needed
        output_path = Path(output_file)
        if output_path.parent.name != '.':
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating test manifest from {data_dir}")
        create_test_manifest(data_dir, output_file)


if __name__ == "__main__":
    main()