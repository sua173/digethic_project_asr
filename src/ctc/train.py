#!/usr/bin/env python3
"""
CTC model training script

Usage:
    python src/ctc/train.py
    python src/ctc/train.py --epochs 30 --batch_size 8 --lr 1e-3
    python src/ctc/train.py --train_manifest path/train.jsonl --val_manifest path/val.jsonl
"""

import os
import sys
import time
from datetime import datetime

# Set environment variable for MPS fallback
# This allows PyTorch to automatically fall back to CPU for operations not supported on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import jiwer
from tqdm import tqdm
from src.ctc.core.utils import TextProcessor, setup_device, print_model_info, init_weights
from src.ctc.core.dataset import create_data_loaders
from src.ctc.model import create_ctc_model
from src.ctc.core.early_stopping import EarlyStopping
from src.ctc.config import get_full_config, update_config


def calculate_cer_wer(predictions: list, targets: list) -> tuple:
    """Calculate CER and WER"""
    if not predictions or not targets:
        return 1.0, 1.0

    # CER (Character Error Rate)
    cer = jiwer.cer(targets, predictions)

    # WER (Word Error Rate)
    wer = jiwer.wer(targets, predictions)

    return cer, wer


def evaluate_model(model, val_loader, criterion, text_processor, device):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Only move spectrograms to MPS/GPU, keep texts and lengths on CPU for CTCLoss
            spectrograms = batch["spectrograms"].to(device)
            texts = batch["texts"]  # Keep on CPU
            spec_lengths = batch["spec_lengths"]  # Keep on CPU for CTCLoss
            text_lengths = batch["text_lengths"]  # Keep on CPU for CTCLoss

            # Forward pass
            log_probs, output_lengths = model(spectrograms, spec_lengths)

            # Move model output to CPU for stable CTC loss computation
            if str(device) != "cpu":
                log_probs = log_probs.to("cpu")
                output_lengths = output_lengths.to("cpu")

            # CTC Loss calculation
            # CTC expects: (time, batch, vocab_size)
            log_probs_ctc = log_probs.transpose(0, 1)
            loss = criterion(log_probs_ctc, texts, output_lengths, text_lengths)
            total_loss += loss.item()

            # Decode predictions
            predictions = model.decode_greedy(
                spectrograms, spec_lengths, text_processor
            )

            # Get target texts
            targets = []
            for i in range(texts.shape[0]):
                target_ids = texts[i, : text_lengths[i]].cpu().tolist()
                target_text = text_processor.ids_to_text(target_ids)
                targets.append(target_text)

            all_predictions.extend(predictions)
            all_targets.extend(targets)

    avg_loss = total_loss / len(val_loader)
    cer, wer = calculate_cer_wer(all_predictions, all_targets)

    return avg_loss, cer, wer, all_predictions[:3], all_targets[:3]


def train_model(config=None):
    """Train the model with configuration"""

    # Get configuration
    if config is None:
        config = get_full_config()
    else:
        config = update_config(get_full_config(), config)

    # Setup
    if config["device"] == "cpu":
        # Force CPU usage when explicitly set
        device = setup_device(force_cpu=True)
    else:
        # Let setup_device choose the best available device (cuda, mps, or cpu)
        device = setup_device(force_cpu=False)
    text_processor = TextProcessor()

    print(f"=== CTC Training Configuration ===")
    print(f"Device: {device}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print(f"LSTM layers: {config['num_layers']}")
    print(f"Bidirectional: {config['bidirectional']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print(f"Vocabulary size: {text_processor.vocab_size}")

    # Data loaders
    train_loader, val_loader = create_data_loaders(
        config["train_manifest"],
        config["val_manifest"],
        text_processor,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Model
    model = create_ctc_model(
        vocab_size=text_processor.vocab_size,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    # Apply weight initialization
    model.apply(init_weights)

    print_model_info(model)

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=text_processor.blank_id, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Scheduler
    if config["scheduler_type"] == "reduce_on_plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config["mode"],
            factor=config["factor"],
            patience=config["scheduler_patience"],
        )
    else:
        # Default to ReduceLROnPlateau if not specified
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.7,
            patience=config.get("scheduler_patience", 3),
        )

    # Learning rate warmup configuration
    warmup_epochs = config.get("warmup_epochs", 0)
    warmup_steps = warmup_epochs * len(train_loader)
    initial_lr = config["lr"]

    # Create timestamped directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ctc_{timestamp}"
    run_dir = os.path.join(config["save_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    print(f"\n💾 Saving checkpoints and logs to: {run_dir}")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        mode="min",
        min_delta=0.0001,
        verbose=True,
        restore_best_weights=True,
    )

    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)

    # Training loop
    best_cer = float("inf")
    best_val_loss = float("inf")

    print(f"\n=== Starting Training ===")

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        train_loss = 0
        start_time = time.time()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(progress_bar):
            # Only move spectrograms to MPS/GPU, keep texts and lengths on CPU for CTCLoss
            spectrograms = batch["spectrograms"].to(device)
            texts = batch["texts"]  # Keep on CPU
            spec_lengths = batch["spec_lengths"]  # Keep on CPU for CTCLoss
            text_lengths = batch["text_lengths"]  # Keep on CPU for CTCLoss

            optimizer.zero_grad()

            # Forward pass
            log_probs, output_lengths = model(spectrograms, spec_lengths)

            # Debug: Check for NaN/Inf in model output
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                print(
                    f"\n🛑 Error at epoch {epoch + 1}, batch {batch_idx}: Model output contains NaN or Inf"
                )
                print(
                    f"Log probs min: {log_probs.min():.4f}, max: {log_probs.max():.4f}"
                )
                break

            # Move model output to CPU for stable CTC loss computation
            if str(device) != "cpu":
                log_probs = log_probs.to("cpu")
                output_lengths = output_lengths.to("cpu")

            # Debug: Check sequence lengths
            if (output_lengths < text_lengths).any():
                print(
                    f"\n🛑 Error at epoch {epoch + 1}, batch {batch_idx}: Invalid lengths for CTCLoss"
                )
                print(f"Output lengths: {output_lengths.cpu().numpy()}")
                print(f"Text lengths: {text_lengths.cpu().numpy()}")
                break

            # CTC Loss
            log_probs_ctc = log_probs.transpose(0, 1)  # (time, batch, vocab_size)
            loss = criterion(log_probs_ctc, texts, output_lengths, text_lengths)

            # Debug: Check loss value
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"\n🛑 Error at epoch {epoch + 1}, batch {batch_idx}: Loss became NaN or Inf"
                )
                print(f"Loss value: {loss.item()}")
                break

            # Backward pass
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=config["gradient_clip"]
            )

            # Debug: Monitor gradient norm
            # if batch_idx % 100 == 0:  # Log every 100 batches
            #     print(f"\nBatch {batch_idx}: Grad norm = {grad_norm:.4f}, Loss = {loss.item():.4f}")

            # Learning rate warmup
            if warmup_steps > 0:
                global_step = epoch * len(train_loader) + batch_idx
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = initial_lr * lr_scale

            optimizer.step()
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Logging to TensorBoard
            if batch_idx % config["log_interval"] == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar(
                    "Train/LR", optimizer.param_groups[0]["lr"], global_step
                )

        # Validation
        val_loss, cer, wer, sample_preds, sample_targets = evaluate_model(
            model, val_loader, criterion, text_processor, device
        )

        # Scheduler step
        scheduler.step(val_loss)

        # Logging
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}/{config['epochs']} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  CER: {cer:.4f} ({cer * 100:.2f}%)")
        print(f"  WER: {wer:.4f} ({wer * 100:.2f}%)")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Sample predictions
        print(f"\nSample Predictions:")
        for i, (pred, target) in enumerate(zip(sample_preds, sample_targets)):
            print(f"  {i + 1}. Pred: '{pred}'")
            print(f"     True: '{target}'")

        # Tensorboard logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Metrics/CER", cer, epoch)
        writer.add_scalar("Metrics/WER", wer, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        # Always save checkpoint (for resuming)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cer": cer,
            "wer": wer,
            "text_processor": text_processor,
        }

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_cer = cer  # Track CER at best val_loss
            torch.save(checkpoint, os.path.join(run_dir, "best_ctc_model.pt"))
            print(f"  ✅ New best model saved! Val Loss: {val_loss:.4f}, CER: {cer:.4f}")

        # Save latest checkpoint (for resuming interrupted training)
        torch.save(checkpoint, os.path.join(run_dir, "latest_checkpoint.pt"))

        # Save periodic checkpoint (skip epoch 0)
        if epoch > 0 and epoch % config["save_interval"] == 0:
            torch.save(
                checkpoint, os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt")
            )

        # Check early stopping
        if early_stopping(val_loss, model, epoch):
            print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
            break

        print("-" * 60)

    writer.close()

    # Print final best validation loss for hyperparameter optimization
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"\n🎉 Training completed! Best Val Loss: {best_val_loss:.4f} (CER: {best_cer:.4f})")
    print(f"Model saved in: {run_dir}")
    print(f"Best model: {os.path.join(run_dir, 'best_ctc_model.pt')}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train CTC Speech Recognition Model")
    parser.add_argument(
        "--train_manifest", type=str, help="Override training manifest file"
    )
    parser.add_argument(
        "--val_manifest", type=str, help="Override validation manifest file"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument(
        "--save_dir", type=str, help="Override directory to save checkpoints"
    )
    parser.add_argument("--hidden_dim", type=int, help="Override hidden dimension")
    parser.add_argument("--num_layers", type=int, help="Override number of LSTM layers")
    parser.add_argument("--dropout", type=float, help="Override dropout rate")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        help="Override device (cuda, mps, or cpu)",
    )
    parser.add_argument("--gradient_clip", type=float, help="Override gradient clipping value")
    parser.add_argument("--warmup_epochs", type=int, help="Override number of warmup epochs")
    parser.add_argument("--early_stopping_patience", type=int, help="Override early stopping patience")

    args = parser.parse_args()

    # Build config overrides
    overrides = {}
    if args.train_manifest:
        overrides["train_manifest"] = args.train_manifest
    if args.val_manifest:
        overrides["val_manifest"] = args.val_manifest
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.lr:
        overrides["lr"] = args.lr
    if args.save_dir:
        overrides["save_dir"] = args.save_dir
    if args.hidden_dim:
        overrides["hidden_dim"] = args.hidden_dim
    if args.num_layers:
        overrides["num_layers"] = args.num_layers
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.device:
        overrides["device"] = args.device
    if args.gradient_clip:
        overrides["gradient_clip"] = args.gradient_clip
    if args.warmup_epochs:
        overrides["warmup_epochs"] = args.warmup_epochs
    if args.early_stopping_patience:
        overrides["early_stopping_patience"] = args.early_stopping_patience

    # Train model
    train_model(overrides if overrides else None)


if __name__ == "__main__":
    main()
