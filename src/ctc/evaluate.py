#!/usr/bin/env python3
"""
Evaluate trained CTC model on test dataset

Usage:
    python src/ctc/evaluate.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt
    python src/ctc/evaluate.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt --test-manifest generated/test_dataset/test_manifest.jsonl
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import jiwer
from collections import defaultdict
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ctc.core.dataset import create_data_loaders
from src.ctc.core.utils import setup_device


def evaluate_model(checkpoint_path, test_manifest, batch_size=8, num_workers=0, device=None):
    """Evaluate model on test dataset"""
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    # Load with weights_only=False to handle TextProcessor object
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model and text processor
    model_state = checkpoint['model_state_dict']
    text_processor = checkpoint['text_processor']
    
    # Setup device
    if device is None:
        device = setup_device(force_cpu=True)  # Default to CPU for compatibility
    print(f"Using device: {device}")
    
    # Recreate model architecture
    from src.ctc.model import create_ctc_model
    model = create_ctc_model(
        vocab_size=text_processor.vocab_size,
        hidden_dim=256,  # Should match training config
        num_layers=3,    # 3 layers based on checkpoint analysis
        dropout=0.0  # No dropout for evaluation
    )
    
    # Load weights
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create test data loader
    print(f"\nLoading test data from: {test_manifest}")
    
    # Create a dummy train loader (required by create_data_loaders)
    test_loader, _ = create_data_loaders(
        train_manifest=test_manifest,  # Use test manifest as train
        val_manifest=test_manifest,    # Dummy, not used
        text_processor=text_processor,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluation metrics
    all_predictions = []
    all_targets = []
    file_results = []
    
    # Performance tracking
    total_audio_duration = 0
    total_inference_time = 0
    
    print("\nEvaluating model...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
            # Move data to device
            spectrograms = batch['spectrograms'].to(device)
            texts = batch['texts']
            spec_lengths = batch['spec_lengths']
            text_lengths = batch['text_lengths']
            
            # Estimate audio duration based on spectrogram length
            # Assuming hop_length=512, sample_rate=16000
            batch_audio_duration = spec_lengths.float().sum().item() * 512 / 16000
            total_audio_duration += batch_audio_duration
            
            # Inference timing
            start_time = time.time()
            
            # Decode predictions
            predictions = model.decode_greedy(spectrograms, spec_lengths, text_processor)
            
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Get target texts
            targets = []
            for i in range(texts.shape[0]):
                target_ids = texts[i, :text_lengths[i]].cpu().tolist()
                target_text = text_processor.ids_to_text(target_ids)
                targets.append(target_text)
            
            # Store results
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                all_predictions.append(pred)
                all_targets.append(target)
                
                # Calculate per-file metrics
                file_cer = jiwer.cer(target, pred) if target else 0.0
                file_wer = jiwer.wer(target, pred) if target else 0.0
                
                # Estimate individual file duration
                file_duration = spec_lengths[i].item() * 512 / 16000
                
                file_results.append({
                    'prediction': pred,
                    'target': target,
                    'cer': file_cer,
                    'wer': file_wer,
                    'audio_duration': file_duration
                })
    
    # Calculate overall metrics
    overall_cer = jiwer.cer(all_targets, all_predictions)
    overall_wer = jiwer.wer(all_targets, all_predictions)
    
    # Performance metrics
    rtf = total_inference_time / total_audio_duration  # Real-time factor
    
    # Error analysis
    cer_distribution = defaultdict(int)
    wer_distribution = defaultdict(int)
    
    for result in file_results:
        # CER distribution (0-10%, 10-20%, etc.)
        cer_bucket = int(result['cer'] * 10) * 10
        cer_distribution[cer_bucket] += 1
        
        # WER distribution
        wer_bucket = int(result['wer'] * 10) * 10
        wer_distribution[wer_bucket] += 1
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test dataset: {test_manifest}")
    print(f"Total files: {len(all_predictions)}")
    print(f"Total audio duration: {total_audio_duration/3600:.2f} hours")
    print()
    print(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)")
    print(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)")
    print()
    print(f"Inference time: {total_inference_time:.2f} seconds")
    print(f"Real-time factor (RTF): {rtf:.4f}x")
    print(f"Processing speed: {total_audio_duration/total_inference_time:.2f}x real-time")
    
    # CER distribution
    print("\nCER Distribution:")
    for bucket in sorted(cer_distribution.keys()):
        count = cer_distribution[bucket]
        percentage = count / len(file_results) * 100
        print(f"  {bucket:3d}-{bucket+10:3d}%: {count:4d} files ({percentage:5.1f}%)")
    
    # WER distribution
    print("\nWER Distribution:")
    for bucket in sorted(wer_distribution.keys()):
        count = wer_distribution[bucket]
        percentage = count / len(file_results) * 100
        print(f"  {bucket:3d}-{bucket+10:3d}%: {count:4d} files ({percentage:5.1f}%)")
    
    # Find best and worst predictions
    sorted_results = sorted(file_results, key=lambda x: x['cer'])
    
    print("\nBest predictions (lowest CER):")
    for i in range(min(3, len(sorted_results))):
        result = sorted_results[i]
        print(f"  {i+1}. CER: {result['cer']:.3f}")
        print(f"     Pred: '{result['prediction']}'")
        print(f"     True: '{result['target']}'")
    
    print("\nWorst predictions (highest CER):")
    for i in range(min(3, len(sorted_results))):
        result = sorted_results[-(i+1)]
        print(f"  {i+1}. CER: {result['cer']:.3f}")
        print(f"     Pred: '{result['prediction']}'")
        print(f"     True: '{result['target']}'")
    
    # Save detailed results
    output_dir = Path(checkpoint_path).parent
    results_file = output_dir / "test_results.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Test Evaluation Results\n")
        f.write(f"="*60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test dataset: {test_manifest}\n")
        f.write(f"Total files: {len(all_predictions)}\n")
        f.write(f"Overall CER: {overall_cer:.4f} ({overall_cer*100:.2f}%)\n")
        f.write(f"Overall WER: {overall_wer:.4f} ({overall_wer*100:.2f}%)\n")
        f.write(f"RTF: {rtf:.4f}x\n")
        f.write(f"\nDetailed predictions:\n")
        f.write("-"*60 + "\n")
        
        for i, result in enumerate(sorted_results):
            f.write(f"\n[{i+1}] CER: {result['cer']:.3f}, WER: {result['wer']:.3f}\n")
            f.write(f"Pred: {result['prediction']}\n")
            f.write(f"True: {result['target']}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("="*60)
    
    return {
        'overall_cer': overall_cer,
        'overall_wer': overall_wer,
        'rtf': rtf,
        'total_files': len(all_predictions),
        'total_duration': total_audio_duration
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate CTC model on test dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-manifest', type=str, default='generated/test_dataset/test_manifest.jsonl',
                        help='Path to test manifest file (default: generated/test_dataset/test_manifest.jsonl)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for evaluation (default: 8)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    # Check if test manifest exists
    if not os.path.exists(args.test_manifest):
        print(f"Error: Test manifest not found: {args.test_manifest}")
        print("Please run: python create_dataset.py --mode test")
        return
    
    # Setup device
    if args.device:
        if args.device == 'cpu':
            device = setup_device(force_cpu=True)
        else:
            device = torch.device(args.device)
    else:
        device = None  # Auto-detect
    
    # Run evaluation
    evaluate_model(
        checkpoint_path=args.checkpoint,
        test_manifest=args.test_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )


if __name__ == "__main__":
    main()