#!/usr/bin/env python3
"""
Analyze TensorBoard event logs and display training metrics.

Usage:
    python src/common/analyze_tb.py [log_path]
    
Examples:
    python src/common/analyze_tb.py  # Find latest log in checkpoints
    python src/common/analyze_tb.py generated/checkpoints/ctc_20240101_120000/logs/
    python src/common/analyze_tb.py generated/checkpoints/attention_20240101_120000/logs/events.out.tfevents.xxx
    
Default: Searches for the latest log file in checkpoints directory
"""

import os
import sys
import argparse
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def find_all_runs(base_dir='generated/checkpoints'):
    """Find all training runs in the checkpoints directory."""
    runs = []
    
    # Look for directories with logs subdirectory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            logs_dir = os.path.join(item_path, 'logs')
            if os.path.exists(logs_dir):
                event_files = glob.glob(os.path.join(logs_dir, 'events.out.tfevents.*'))
                if event_files:
                    # Get the latest event file in this run
                    latest_event = max(event_files, key=os.path.getmtime)
                    runs.append({
                        'run_name': item,
                        'event_file': latest_event,
                        'timestamp': os.path.getmtime(latest_event)
                    })
    
    # Sort by timestamp
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs


def find_latest_event_file(base_dir='generated/checkpoints'):
    """Find the latest TensorBoard event file across all runs."""
    runs = find_all_runs(base_dir)
    if runs:
        return runs[0]['event_file']
    return None


def detect_model_type_from_metrics(ea):
    """Detect model type from available metrics."""
    scalar_tags = ea.Tags()['scalars']
    
    # Check for model-specific metrics
    if any('CTC' in tag for tag in scalar_tags):
        return 'CTC'
    if any('Transformer' in tag for tag in scalar_tags):
        return 'Transformer'
    if any('Attention' in tag for tag in scalar_tags):
        return 'Transformer'
    
    # Default detection based on common patterns
    if 'Metrics/CER' in scalar_tags:
        return 'Speech Recognition Model'
    
    return 'Unknown'


def analyze_tensorboard_logs(log_path=None, plot=False, save_plot=False):
    """Analyze TensorBoard event logs and print detailed metrics."""
    checkpoint_dir = None  # Track checkpoint directory for saving plots
    
    # If no specific file provided, find the latest
    if log_path is None:
        log_path = find_latest_event_file()
        if log_path is None:
            print("Error: No TensorBoard event files found in checkpoints directory")
            return
        print(f"Found latest log file: {log_path}")
    elif os.path.isdir(log_path):
        # If directory provided, find event file in it
        event_files = glob.glob(os.path.join(log_path, 'events.out.tfevents.*'))
        if not event_files:
            print(f"Error: No TensorBoard event files found in {log_path}")
            return
        log_path = max(event_files, key=os.path.getmtime)
    
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        return
    
    print(f"Loading TensorBoard logs from: {log_path}")
    
    # Extract checkpoint directory from log path
    if 'checkpoints' in log_path:
        # Convert to absolute path and split
        abs_log_path = os.path.abspath(log_path)
        path_parts = abs_log_path.split(os.sep)
        
        # Find the checkpoint directory (parent of logs if exists)
        checkpoint_dir = None
        for i, part in enumerate(path_parts):
            if part == 'checkpoints' and i + 1 < len(path_parts):
                # Check if the path contains a 'logs' subdirectory
                if i + 2 < len(path_parts) and path_parts[i + 2] == 'logs':
                    # Parent of logs: generated/checkpoints/ctc_YYYYMMDD_HHMMSS
                    checkpoint_dir = os.sep.join(path_parts[:i+2])
                else:
                    # Direct checkpoint directory
                    checkpoint_dir = os.sep.join(path_parts[:i+2])
                break
    
    # Load TensorBoard logs
    ea = EventAccumulator(log_path)
    ea.Reload()
    
    # Detect model type
    model_type = detect_model_type_from_metrics(ea)
    
    print('=' * 50)
    print('TENSORBOARD EVENT ANALYSIS')
    print('=' * 50)
    print(f'\nModel type: {model_type}')
    print(f'Available tags: {ea.Tags()["scalars"]}')
    print()

    # Extract scalar data
    scalars_dict = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars_dict[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events]
        }

    # Analyze each metric
    for tag, data in scalars_dict.items():
        values = data['values']
        steps = data['steps']
        
        print(f'\n=== {tag} ===')
        print(f'Total steps: {len(steps)}')
        if values:
            print(f'Initial value: {values[0]:.4f}')
            print(f'Final value: {values[-1]:.4f}')
            
            if 'loss' in tag.lower():
                print(f'Best (min) value: {min(values):.4f} at step {steps[values.index(min(values))]}')
                improvement = (values[0] - values[-1]) / values[0] * 100 if values[0] != 0 else 0
                print(f'Loss reduction: {improvement:.1f}%')
            elif 'cer' in tag.lower() or 'wer' in tag.lower():
                print(f'Best (min) value: {min(values):.4f} at step {steps[values.index(min(values))]}')
                if values[0] > 0:
                    improvement = (values[0] - values[-1]) / values[0] * 100
                    print(f'Error reduction: {improvement:.1f}%')
            elif 'lr' in tag.lower() or 'learning_rate' in tag.lower():
                print(f'Min LR: {min(values):.6f}')
                print(f'Max LR: {max(values):.6f}')
            
            print(f'Average: {np.mean(values):.4f}')
            print(f'Std dev: {np.std(values):.4f}')

    # Training time analysis
    if scalars_dict:
        first_tag = list(scalars_dict.keys())[0]
        wall_times = scalars_dict[first_tag]['wall_times']
        if len(wall_times) > 1:
            total_time = wall_times[-1] - wall_times[0]
            print(f'\n=== Training Time ===')
            print(f'Total training time: {total_time/3600:.2f} hours')
            print(f'Average time per epoch: {total_time/len(wall_times)/60:.2f} minutes')
            
            # Show convergence analysis
            if 'Validation/loss' in scalars_dict or 'Loss/Val' in scalars_dict:
                val_key = 'Validation/loss' if 'Validation/loss' in scalars_dict else 'Loss/Val'
                val_losses = scalars_dict[val_key]['values']
                print(f'\n=== Convergence Analysis ===')
                print(f'Epochs to reach <1.5 loss: {next((i for i, v in enumerate(val_losses) if v < 1.5), "Not reached")}')
                print(f'Epochs to reach <1.0 loss: {next((i for i, v in enumerate(val_losses) if v < 1.0), "Not reached")}')
                
                # Check for early stopping
                if len(val_losses) > 5:
                    recent_improvement = (val_losses[-5] - val_losses[-1]) / val_losses[-5] * 100
                    print(f'Improvement in last 5 epochs: {recent_improvement:.2f}%')
    
    # Epoch-based analysis
    print('\n=== Epoch-based Metrics ===')
    
    # Debug: Show available LR tags
    lr_tags = [tag for tag in scalars_dict.keys() if 'lr' in tag.lower() or 'learning' in tag.lower()]
    if lr_tags:
        print(f"Found Learning Rate tags: {lr_tags}")
        for tag in lr_tags:
            print(f"  {tag}: {len(scalars_dict[tag]['values'])} values")
    
    epoch_metrics = extract_epoch_metrics(scalars_dict)
    display_epoch_progression(epoch_metrics)
    
    # Plot if requested
    if plot:
        plot_epoch_metrics(epoch_metrics, save_plot=save_plot, checkpoint_dir=checkpoint_dir)


def extract_epoch_metrics(scalars_dict):
    """Extract metrics grouped by epoch."""
    epoch_metrics = defaultdict(lambda: defaultdict(list))
    
    # First, determine the number of epochs from epoch-based metrics
    num_epochs = 0
    for tag in ['Loss/Train', 'Loss/Val', 'Metrics/CER', 'Metrics/WER']:
        if tag in scalars_dict:
            num_epochs = max(num_epochs, len(scalars_dict[tag]['values']))
    
    # Process epoch-based metrics
    for tag in scalars_dict.keys():
        data = scalars_dict[tag]
        
        # Check if this is an epoch-based metric (recorded once per epoch)
        if tag in ['Loss/Train', 'Loss/Val', 'Metrics/CER', 'Metrics/WER', 'Learning_Rate']:
            for i, value in enumerate(data['values']):
                epoch_metrics[i][tag] = value
        
        # Handle per-batch metrics (like Train/LR) - take the last value of each epoch
        elif 'lr' in tag.lower() and num_epochs > 0:
            # Calculate approximate steps per epoch
            total_steps = len(data['values'])
            steps_per_epoch = total_steps // num_epochs if num_epochs > 0 else total_steps
            
            # Extract the last LR value for each epoch
            for epoch in range(num_epochs):
                # Get the index of the last step in this epoch
                if steps_per_epoch > 0:
                    last_step_idx = min((epoch + 1) * steps_per_epoch - 1, total_steps - 1)
                    if last_step_idx < len(data['values']):
                        epoch_metrics[epoch]['Learning_Rate'] = data['values'][last_step_idx]
    
    return epoch_metrics


def display_epoch_progression(epoch_metrics):
    """Display metrics progression by epoch."""
    if not epoch_metrics:
        print("No epoch-based metrics found")
        return
    
    # Create header
    headers = ['Epoch', 'Train Loss', 'Val Loss', 'CER', 'WER', 'LR']
    print('\n' + '\t'.join(headers))
    print('-' * 80)
    
    # Display each epoch
    for epoch in sorted(epoch_metrics.keys()):
        metrics = epoch_metrics[epoch]
        row = [f"{epoch+1:3d}"]
        
        # Add metrics in order
        for tag, header in [('Loss/Train', 'train_loss'), ('Loss/Val', 'val_loss'), 
                           ('Metrics/CER', 'cer'), ('Metrics/WER', 'wer'), 
                           ('Learning_Rate', 'lr')]:
            if tag in metrics:
                value = metrics[tag]
                if 'Loss' in tag:
                    row.append(f"{value:8.4f}")
                elif 'CER' in tag or 'WER' in tag:
                    row.append(f"{value*100:7.2f}%")
                elif 'LR' in tag or 'Learning_Rate' in tag:
                    row.append(f"{value:8.6f}")
            else:
                row.append("     -    ")
        
        print('\t'.join(row))
    
    # Summary statistics
    print('\n--- Summary ---')
    if 'Loss/Val' in epoch_metrics[0]:
        val_losses = [epoch_metrics[e].get('Loss/Val', float('inf')) for e in sorted(epoch_metrics.keys())]
        best_epoch = np.argmin(val_losses)
        print(f"Best validation loss: {val_losses[best_epoch]:.4f} at epoch {best_epoch+1}")
    
    if 'Metrics/CER' in epoch_metrics[0]:
        cers = [epoch_metrics[e].get('Metrics/CER', 1.0) for e in sorted(epoch_metrics.keys())]
        best_epoch = np.argmin(cers)
        print(f"Best CER: {cers[best_epoch]*100:.2f}% at epoch {best_epoch+1}")


def plot_epoch_metrics(epoch_metrics, save_plot=False, checkpoint_dir=None):
    """Plot metrics over epochs."""
    if not epoch_metrics:
        print("No epoch-based metrics to plot")
        return
    
    epochs = sorted(epoch_metrics.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16)
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if any('Loss/Train' in epoch_metrics[e] for e in epochs):
        train_losses = [epoch_metrics[e].get('Loss/Train', np.nan) for e in epochs]
        ax.plot([e+1 for e in epochs], train_losses, 'b-', label='Train Loss', linewidth=2)
    
    if any('Loss/Val' in epoch_metrics[e] for e in epochs):
        val_losses = [epoch_metrics[e].get('Loss/Val', np.nan) for e in epochs]
        ax.plot([e+1 for e in epochs], val_losses, 'r-', label='Val Loss', linewidth=2)
        
        # Mark best validation
        best_idx = np.nanargmin(val_losses)
        ax.plot(best_idx+1, val_losses[best_idx], 'ro', markersize=10, 
                label=f'Best Val: {val_losses[best_idx]:.4f}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Error rates
    ax = axes[0, 1]
    if any('Metrics/CER' in epoch_metrics[e] for e in epochs):
        cers = [epoch_metrics[e].get('Metrics/CER', np.nan) * 100 for e in epochs]
        ax.plot([e+1 for e in epochs], cers, 'g-', label='CER', linewidth=2)
        
        # Mark best CER
        best_idx = np.nanargmin(cers)
        ax.plot(best_idx+1, cers[best_idx], 'go', markersize=10, 
                label=f'Best CER: {cers[best_idx]:.2f}%')
    
    if any('Metrics/WER' in epoch_metrics[e] for e in epochs):
        wers = [epoch_metrics[e].get('Metrics/WER', np.nan) * 100 for e in epochs]
        ax.plot([e+1 for e in epochs], wers, 'm-', label='WER', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Rates')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning rate
    ax = axes[1, 0]
    if any('Learning_Rate' in epoch_metrics[e] for e in epochs):
        lrs = [epoch_metrics[e].get('Learning_Rate', np.nan) for e in epochs]
        ax.plot([e+1 for e in epochs], lrs, 'c-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Combined normalized view
    ax = axes[1, 1]
    if any('Loss/Val' in epoch_metrics[e] for e in epochs):
        val_losses = np.array([epoch_metrics[e].get('Loss/Val', np.nan) for e in epochs])
        val_losses_norm = (val_losses - np.nanmin(val_losses)) / (np.nanmax(val_losses) - np.nanmin(val_losses))
        ax.plot([e+1 for e in epochs], val_losses_norm, 'r-', label='Val Loss (norm)', linewidth=2)
    
    if any('Metrics/CER' in epoch_metrics[e] for e in epochs):
        cers = np.array([epoch_metrics[e].get('Metrics/CER', np.nan) for e in epochs])
        ax.plot([e+1 for e in epochs], cers, 'g-', label='CER', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Normalized Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_plot:
        if checkpoint_dir:
            plot_filename = os.path.join(checkpoint_dir, 'training_metrics.png')
        else:
            plot_filename = 'training_metrics.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\\nPlot saved to: {plot_filename}")
    else:
        plt.show()


def list_all_runs(base_dir='generated/checkpoints'):
    """List all available training runs."""
    runs = find_all_runs(base_dir)
    
    if not runs:
        print(f"No training runs found in {base_dir}")
        return
    
    print(f"Available training runs in {base_dir}:")
    print("-" * 60)
    for i, run in enumerate(runs):
        print(f"{i+1}. {run['run_name']}")
        print(f"   Last updated: {pd.Timestamp.fromtimestamp(run['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze TensorBoard event logs')
    parser.add_argument('log_path', nargs='?', default=None,
                        help='Path to event file or log directory (default: searches for latest)')
    parser.add_argument('--list', action='store_true',
                        help='List all available training runs')
    parser.add_argument('--dir', default='generated/checkpoints',
                        help='Base directory for checkpoints (default: checkpoints)')
    parser.add_argument('--plot', action='store_true',
                        help='Plot metrics over epochs')
    parser.add_argument('--save-plot', action='store_true',
                        help='Save plot to file instead of displaying')
    
    args = parser.parse_args()
    
    if args.list:
        list_all_runs(args.dir)
    else:
        analyze_tensorboard_logs(args.log_path, plot=args.plot, save_plot=args.save_plot)


if __name__ == '__main__':
    main()