# Train RNN Attention Encoder-Decoder model

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import json
import shutil
import numpy as np
import argparse
import multiprocessing
import jiwer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import optim

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (no GUI)
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from src.attention.core import levenshtein
from src.attention.core.dataset import SequenceDataset
from src.attention.model.attention_model import AttentionModel
from src.attention.config import (
    get_default_config,
    update_config,
    save_config_summary,
    setup_device,
)


def calculate_cer_wer(hypothesis_tokens: list, reference_tokens: list) -> tuple:
    """
    Calculate CER and WER from character-level token lists

    Args:
        hypothesis_tokens: List of hypothesis character tokens
        reference_tokens: List of reference character tokens

    Returns:
        tuple: (cer, wer, ter)
    """
    # Remove special tokens
    special_tokens = ["<blank>", "<eos>", "<sos>"]
    hypothesis_clean = [t for t in hypothesis_tokens if t not in special_tokens]
    reference_clean = [t for t in reference_tokens if t not in special_tokens]

    # Calculate TER (Token Error Rate) - character level
    from src.attention.core import levenshtein

    if len(reference_clean) == 0:
        ter = 1.0 if len(hypothesis_clean) > 0 else 0.0
    else:
        (error, _, _, _, ref_length) = levenshtein.calculate_error(
            hypothesis_clean, reference_clean
        )
        ter = error / ref_length

    # Convert token lists to strings (preserve spaces for WER calculation)
    hypothesis_str = "".join(
        [
            " " if t == "<SPACE>" else t
            for t in hypothesis_tokens
            if t not in ["<blank>", "<eos>", "<sos>"]
        ]
    )
    reference_str = "".join(
        [
            " " if t == "<SPACE>" else t
            for t in reference_tokens
            if t not in ["<blank>", "<eos>", "<sos>"]
        ]
    )

    # Calculate CER and WER using jiwer
    if reference_str == "":
        cer = 1.0 if hypothesis_str != "" else 0.0
        wer = 1.0 if hypothesis_str != "" else 0.0
    else:
        cer = jiwer.cer(reference_str, hypothesis_str)
        wer = jiwer.wer(reference_str, hypothesis_str)

    return cer, wer, ter


def setup_directories(config):
    """Create output directories"""
    # Attention weight matrix output directory
    out_att_dir = os.path.join(config["output_dir"], "att_matrix")
    os.makedirs(out_att_dir, exist_ok=True)

    # TensorBoard log directory
    log_dir = os.path.join(config["output_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Save configuration
    conf_file = os.path.join(config["output_dir"], "config.json")
    with open(conf_file, mode="w") as f:
        json.dump(save_config_summary(config), f, indent=4)

    return out_att_dir, log_dir


def load_data_info(config):
    """Load mean/std and token list"""
    # Feature list files
    feat_scp_train = os.path.join(config["feat_dir_train"], "feats.scp")
    feat_scp_dev = os.path.join(config["feat_dir_dev"], "feats.scp")

    # Label files
    train_set_name = os.path.basename(config["feat_dir_train"])
    dev_set_name = os.path.basename(config["feat_dir_dev"])
    label_train = os.path.join(
        config["token_dir"], train_set_name, "label_" + train_set_name
    )
    label_dev = os.path.join(config["token_dir"], dev_set_name, "label_" + dev_set_name)

    # Mean/standard deviation file
    mean_std_file = os.path.join(config["feat_dir_train"], "mean_std.txt")

    # Token list
    token_list_path = os.path.join(config["token_dir"], "token_list")

    # Load mean/std
    with open(mean_std_file, mode="r") as f:
        lines = f.readlines()
        # LibriSpeech format only
        mean_line = lines[0]
        std_line = lines[1]
        feat_mean = np.array(mean_line.split(), dtype=np.float32)
        feat_std = np.array(std_line.split(), dtype=np.float32)

    # Copy mean/std file to output directory
    shutil.copyfile(mean_std_file, os.path.join(config["output_dir"], "mean_std.txt"))

    # Load token list
    token_list = {0: "<blank>"}
    with open(token_list_path, mode="r") as f:
        for line in f:
            parts = line.split()
            token_list[int(parts[1])] = parts[0]

    # Add <eos> token
    eos_id = len(token_list)
    token_list[eos_id] = "<eos>"
    sos_id = eos_id

    return {
        "feat_scp_train": feat_scp_train,
        "feat_scp_dev": feat_scp_dev,
        "label_train": label_train,
        "label_dev": label_dev,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "token_list": token_list,
        "sos_id": sos_id,
        "eos_id": eos_id,
        "feat_dim": feat_mean.shape[0],
        "num_tokens": len(token_list),
    }


def create_model(config, feat_dim, num_tokens, sos_id):
    """Create the E2E model"""
    model = AttentionModel(
        dim_in=feat_dim,
        dim_enc_hid=config["enc_hidden_dim"],
        dim_enc_proj=config["enc_projection_dim"],
        dim_dec_hid=config["dec_hidden_dim"],
        dim_out=num_tokens,
        dim_att=config["att_hidden_dim"],
        att_filter_size=config["att_filter_size"],
        att_filter_num=config["att_filter_num"],
        sos_id=sos_id,
        att_temperature=config["att_temperature"],
        enc_num_layers=config["enc_num_layers"],
        dec_num_layers=config["dec_num_layers"],
        enc_bidirectional=config["enc_bidirectional"],
        enc_sub_sample=config["enc_sub_sample"],
        enc_rnn_type=config["enc_rnn_type"],
        enc_dropout_rate=config.get("enc_dropout_rate", 0.2),
        dec_dropout_rate=config.get("dec_dropout_rate", 0.2),
    )
    return model


def create_data_loaders(data_info, config):
    """Create training and development data loaders"""
    train_dataset = SequenceDataset(
        data_info["feat_scp_train"],
        data_info["label_train"],
        data_info["feat_mean"],
        data_info["feat_std"],
    )

    dev_dataset = SequenceDataset(
        data_info["feat_scp_dev"],
        data_info["label_dev"],
        data_info["feat_mean"],
        data_info["feat_std"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )

    dev_loader = DataLoader(
        dev_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    return train_loader, dev_loader


def save_attention_matrix(model, indices, utt_ids, epoch, out_att_dir):
    """Save attention weight matrix for visualization"""
    # Save attention matrix for the first utterance
    # Use index 0 since attention matrices are stored in sorted order
    idx = torch.nonzero(indices == 0, as_tuple=False).view(-1)[0]
    out_name = os.path.join(out_att_dir, "%s_ep%d.png" % (utt_ids[0], epoch + 1))
    model.save_att_matrix(idx, out_name)


def train_one_epoch(
    model,
    dataset_loader,
    optimizer,
    criterion,
    device,
    data_info,
    config,
    epoch,
    loss_history,
    error_history,
    cer_history,
    wer_history,
    log_file,
    out_att_dir,
    writer,
):
    """Process one full epoch (both train and validation phases)"""

    # Process both phases
    for phase in ["train", "validation"]:
        # Accumulated values for this phase
        total_loss = 0
        total_utt = 0
        total_error = 0
        total_token_length = 0
        total_cer = 0
        total_wer = 0
        total_cer_samples = 0
        total_wer_samples = 0

        # Progress bar
        total_batches = len(dataset_loader[phase])
        pbar = tqdm(
            dataset_loader[phase],
            total=total_batches,
            desc=f"  {phase}",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        # Process each batch
        for features, labels, feat_lens, label_lens, utt_ids in pbar:
            # Sort by frame length (required for PackedSequence)
            sorted_lens, indices = torch.sort(
                feat_lens.view(-1), dim=0, descending=True
            )
            features = features[indices]
            labels = labels[indices]
            feat_lens = sorted_lens
            label_lens = label_lens[indices]

            # Add <eos> to labels
            labels = torch.cat(
                (labels, torch.zeros(labels.size()[0], 1, dtype=torch.long)), dim=1
            )
            for m, length in enumerate(label_lens):
                labels[m][length] = data_info["eos_id"]
            label_lens += 1

            # Truncate labels to batch max length
            labels = labels[:, : torch.max(label_lens)]

            # Move to device
            features, labels = features.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs, _ = model(features, feat_lens, labels)

            # Calculate loss
            b_size, t_size, _ = outputs.size()
            loss = criterion(
                outputs.view(b_size * t_size, data_info["num_tokens"]),
                labels.reshape(-1),
            )
            # deactivate
            # loss *= np.mean(label_lens.numpy())

            # Backward pass and update (training only)
            if phase == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["clip_grad_threshold"]
                )
                optimizer.step()

            # Calculate error if needed
            if config["evaluate_error"][phase]:
                for n in range(outputs.size(0)):
                    _, hyp_per_step = torch.max(outputs[n], 1)
                    hyp_per_step = hyp_per_step.cpu().numpy()
                    hypothesis = []
                    for m in hyp_per_step[: label_lens[n]]:
                        hypothesis.append(data_info["token_list"][m])
                    reference = []
                    for m in labels[n][: label_lens[n]].cpu().numpy():
                        reference.append(data_info["token_list"][m])

                    # Calculate CER, WER, and TER
                    cer, wer, ter = calculate_cer_wer(hypothesis, reference)

                    # Accumulate CER and WER
                    total_cer += cer
                    total_wer += wer
                    total_cer_samples += 1
                    total_wer_samples += 1

                    # For backward compatibility, still calculate TER using levenshtein
                    (error, substitute, delete, insert, ref_length) = (
                        levenshtein.calculate_error(hypothesis, reference)
                    )
                    total_error += error
                    total_token_length += ref_length

            # Save attention matrix for validation (first batch of each epoch)
            if phase == "validation" and total_loss == 0:
                save_attention_matrix(model, indices, utt_ids, epoch, out_att_dir)

            # Accumulate loss
            total_loss += loss.item()
            total_utt += outputs.size(0)

            # Update progress bar
            current_avg_loss = total_loss / total_utt
            if config["evaluate_error"][phase] and total_token_length > 0:
                current_error = 100.0 * total_error / total_token_length
                pbar.set_postfix(
                    loss=f"{current_avg_loss:.4f}", error=f"{current_error:.2f}%"
                )
            else:
                pbar.set_postfix(loss=f"{current_avg_loss:.4f}")

        # Calculate epoch metrics for this phase
        epoch_loss = total_loss / total_utt
        print("    %s loss: %f" % (phase, epoch_loss))
        log_file.write("%.6f\t" % (epoch_loss))
        loss_history[phase].append(epoch_loss)

        if config["evaluate_error"][phase]:
            epoch_error = 100.0 * total_error / total_token_length
            epoch_cer = (
                100.0 * total_cer / total_cer_samples if total_cer_samples > 0 else 0
            )
            epoch_wer = (
                100.0 * total_wer / total_wer_samples if total_wer_samples > 0 else 0
            )

            print("    %s token error rate: %f %%" % (phase, epoch_error))
            print("    %s CER: %.2f %%, WER: %.2f %%" % (phase, epoch_cer, epoch_wer))
            log_file.write("%.6f\t" % (epoch_error))
            error_history[phase].append(epoch_error)
            cer_history[phase].append(epoch_cer)
            wer_history[phase].append(epoch_wer)
        else:
            log_file.write("     ---     \t")
            epoch_error = None

        # Store validation loss for early stopping
        if phase == "validation":
            validation_loss = epoch_loss

    # TensorBoard logging (after both phases complete)
    writer.add_scalar("Loss/Train", loss_history["train"][-1], epoch)
    writer.add_scalar("Loss/Val", loss_history["validation"][-1], epoch)

    if config["evaluate_error"]["train"] and len(error_history["train"]) > 0:
        writer.add_scalar("Metrics/TER_Train", error_history["train"][-1], epoch)
    if config["evaluate_error"]["validation"] and len(error_history["validation"]) > 0:
        writer.add_scalar("Metrics/TER_Val", error_history["validation"][-1], epoch)

    # Only log validation CER/WER (same as CTC model)
    # Note: cer_history and wer_history store percentage values (0-100),
    # but TensorBoard expects decimal values (0-1) like CTC model
    if len(cer_history["validation"]) > 0:
        writer.add_scalar("Metrics/CER", cer_history["validation"][-1] / 100.0, epoch)
    if len(wer_history["validation"]) > 0:
        writer.add_scalar("Metrics/WER", wer_history["validation"][-1] / 100.0, epoch)

    # Log learning rate
    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

    return validation_loss


def plot_results(loss_history, error_history, output_dir, evaluate_error):
    """Plot and save training curves"""
    # Loss curve
    fig1 = plt.figure()
    for phase in ["train", "validation"]:
        plt.plot(loss_history[phase], label=phase + " loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig1.legend()
    fig1.savefig(os.path.join(output_dir, "loss.png"))
    plt.close(fig1)

    # Error curve
    fig2 = plt.figure()
    for phase in ["train", "validation"]:
        if evaluate_error[phase]:
            plt.plot(error_history[phase], label=phase + " error")
    plt.xlabel("Epoch")
    plt.ylabel("Error [%]")
    fig2.legend()
    fig2.savefig(os.path.join(output_dir, "error.png"))
    plt.close(fig2)


def cleanup(model, train_loader, dev_loader):
    """Clean up resources"""
    # Flush output buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Clean up PyTorch resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    # Properly shutdown DataLoader workers
    for loader in [train_loader, dev_loader]:
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            loader._iterator._shutdown_workers()

    # Delete model and loaders to free memory
    del model
    del train_loader
    del dev_loader

    # Run garbage collection
    import gc

    gc.collect()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train attention-based ASR model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config.py)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config.py)",
    )
    parser.add_argument(
        "--train-set",
        type=str,
        default=None,
        help="Training set name (default: use config.py setting)",
    )
    parser.add_argument(
        "--dev-set",
        type=str,
        default=None,
        help="Development set name (default: use config.py setting)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=None,
        help="Test set name (for config reference, not used in training)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Initial learning rate (default: from config.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: auto-generated)",
    )
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()

    # Get default configuration
    config = get_default_config()

    # Update configuration with command line arguments
    config = update_config(config, args)

    # Setup directories
    out_att_dir, log_dir = setup_directories(config)

    # Load data information
    data_info = load_data_info(config)

    # Create model
    model = create_model(
        config, data_info["feat_dim"], data_info["num_tokens"], data_info["sos_id"]
    )

    # Create optimizer
    # optimizer = optim.Adadelta(
    #     model.parameters(),
    #     lr=config["initial_learning_rate"],
    #     rho=0.95,
    #     eps=1e-6,
    #     weight_decay=0.0,
    # )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Setup device
    if config["device"] == "cpu":
        device = setup_device(force_cpu=True)
    else:
        device = setup_device(force_cpu=False)

    # Set multiprocessing start method for MPS (Apple Silicon)
    if device.type == "mps":
        multiprocessing.set_start_method("fork", force=True)

    # Create data loaders (after setting multiprocessing if needed)
    train_loader, dev_loader = create_data_loaders(data_info, config)

    # Loss function
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    model = model.to(device)

    # Set model to training mode
    model.train()

    # Data loaders dictionary
    dataset_loader = {"train": train_loader, "validation": dev_loader}

    # History tracking
    loss_history = {"train": [], "validation": []}
    error_history = {"train": [], "validation": []}
    cer_history = {"train": [], "validation": []}
    wer_history = {"train": [], "validation": []}

    # Best model tracking
    best_loss = -1
    best_epoch = 0
    early_stop_flag = False
    counter_for_early_stop = 0

    # Open log file
    log_file = open(os.path.join(config["output_dir"], "log.txt"), mode="w")
    log_file.write("epoch\ttrain loss\ttrain err\tvalid loss\tvalid err")

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop
    for epoch in range(config["max_num_epoch"]):
        if early_stop_flag:
            print(
                "    Early stopping."
                " (early_stop_threshold = %d)" % (config["early_stop_threshold"])
            )
            log_file.write(
                "\n    Early stopping."
                " (early_stop_threshold = %d)" % (config["early_stop_threshold"])
            )
            break

        print("\nepoch %d/%d:" % (epoch + 1, config["max_num_epoch"]))
        log_file.write("\n%d\t" % (epoch + 1))

        # Process one full epoch (train + validation)
        validation_loss = train_one_epoch(
            model,
            dataset_loader,
            optimizer,
            criterion,
            device,
            data_info,
            config,
            epoch,
            loss_history,
            error_history,
            cer_history,
            wer_history,
            log_file,
            out_att_dir,
            writer,
        )

        # Model saving and early stopping logic
        improved = False
        if epoch == 0 or best_loss > validation_loss:
            improved = True
            best_loss = validation_loss
            torch.save(
                model.state_dict(),
                os.path.join(config["output_dir"], "best_model.pt"),
            )
            best_epoch = epoch
            counter_for_early_stop = 0
        else:
            if epoch + 1 >= config["lr_decay_start_epoch"]:
                if counter_for_early_stop + 1 >= config["early_stop_threshold"]:
                    early_stop_flag = True
                else:
                    # Learning rate decay
                    if config["lr_decay_factor"] < 1.0:
                        for i, param_group in enumerate(optimizer.param_groups):
                            if i == 0:
                                lr = param_group["lr"]
                                dlr = config["lr_decay_factor"] * lr
                                print("    (Decay learning rate: %f -> %f)" % (lr, dlr))
                                log_file.write(
                                    "(Decay learning rate: %f -> %f)" % (lr, dlr)
                                )
                            param_group["lr"] = dlr
                    counter_for_early_stop += 1

        # Print early stopping status
        if epoch + 1 >= config["lr_decay_start_epoch"]:
            if improved:
                print(
                    "    Early Stopping: [Reset] - Val loss improved! (Best: %.4f)"
                    % validation_loss
                )
            elif early_stop_flag:
                print(
                    "    Early Stopping: [Triggered] - Stopping after %d epochs without improvement"
                    % config["early_stop_threshold"]
                )
            elif counter_for_early_stop > 0:
                if config["lr_decay_factor"] < 1.0 and counter_for_early_stop == 1:
                    print(
                        "    Early Stopping: [LR Decay] - No improvement for %d/%d epochs (LR: %.1f → %.1f)"
                        % (
                            counter_for_early_stop,
                            config["early_stop_threshold"],
                            optimizer.param_groups[0]["lr"] / config["lr_decay_factor"],
                            optimizer.param_groups[0]["lr"],
                        )
                    )
                else:
                    print(
                        "    Early Stopping: [Waiting] - No improvement for %d/%d epochs"
                        % (counter_for_early_stop, config["early_stop_threshold"])
                    )
        else:
            print(
                "    Early Stopping: Not active until epoch %d"
                % config["lr_decay_start_epoch"]
            )

    # Save final model
    print("---------------Summary------------------")
    log_file.write("\n---------------Summary------------------\n")

    torch.save(model.state_dict(), os.path.join(config["output_dir"], "final_model.pt"))
    print(
        "Final epoch model -> %s"
        % (os.path.join(config["output_dir"], "final_model.pt"))
    )
    log_file.write(
        "Final epoch model -> %s\n"
        % (os.path.join(config["output_dir"], "final_model.pt"))
    )

    # Print final and best epoch results
    for phase in ["train", "validation"]:
        print("    %s loss: %f" % (phase, loss_history[phase][-1]))
        log_file.write("    %s loss: %f\n" % (phase, loss_history[phase][-1]))
        if config["evaluate_error"][phase]:
            print("    %s token error rate: %f %%" % (phase, error_history[phase][-1]))
            if len(cer_history[phase]) > 0:
                print(
                    "    %s CER: %.2f %%, WER: %.2f %%"
                    % (phase, cer_history[phase][-1], wer_history[phase][-1])
                )
            log_file.write(
                "    %s token error rate: %f %%\n" % (phase, error_history[phase][-1])
            )
        else:
            print("    %s token error rate: (not evaluated)" % (phase))
            log_file.write("    %s token error rate: (not evaluated)\n" % (phase))

    print(
        "Best epoch model (%d-th epoch)"
        " -> %s" % (best_epoch + 1, os.path.join(config["output_dir"], "best_model.pt"))
    )
    log_file.write(
        "Best epoch model (%d-th epoch)"
        " -> %s\n"
        % (best_epoch + 1, os.path.join(config["output_dir"], "best_model.pt"))
    )

    for phase in ["train", "validation"]:
        print("    %s loss: %f" % (phase, loss_history[phase][best_epoch]))
        log_file.write("    %s loss: %f\n" % (phase, loss_history[phase][best_epoch]))
        if config["evaluate_error"][phase]:
            print(
                "    %s token error rate: %f %%"
                % (phase, error_history[phase][best_epoch])
            )
            if len(cer_history[phase]) > best_epoch:
                print(
                    "    %s CER: %.2f %%, WER: %.2f %%"
                    % (
                        phase,
                        cer_history[phase][best_epoch],
                        wer_history[phase][best_epoch],
                    )
                )
            log_file.write(
                "    %s token error rate: %f %%\n"
                % (phase, error_history[phase][best_epoch])
            )
        else:
            print("    %s token error rate: (not evaluated)" % (phase))
            log_file.write("    %s token error rate: (not evaluated)\n" % (phase))

    # Plot results - commented out as requested
    # plot_results(
    #     loss_history, error_history, config["output_dir"], config["evaluate_error"]
    # )

    # Close log file
    log_file.close()

    # Close TensorBoard writer
    writer.close()

    print("\nTraining completed successfully!")
    print(f"All results saved to: {config['output_dir']}")

    # Cleanup
    cleanup(model, train_loader, dev_loader)

    # Exit normally
    sys.exit(0)


if __name__ == "__main__":
    main()
