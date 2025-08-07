# Evaluate trained RNN Attention Encoder-Decoder model
# This script combines decoding and scoring functionalities

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
import numpy as np
import json
import glob
import argparse
import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attention.core.dataset import SequenceDataset
from src.attention.config import setup_device, get_default_config
from src.attention.model.attention_model import AttentionModel
from src.attention.core import levenshtein
import jiwer


def find_checkpoint_directory(checkpoint_path=None):
    """Find checkpoint directory"""
    if checkpoint_path:
        # Use specified checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Specified checkpoint directory not found: {checkpoint_path}")
            sys.exit(1)
        print(f"Using specified checkpoint: {checkpoint_path}")
        return checkpoint_path
    else:
        # Find the latest checkpoint directory
        checkpoint_dirs = glob.glob("generated/checkpoints/attention_202*")
        if not checkpoint_dirs:
            print("Error: No checkpoint directories found in generated/checkpoints/")
            sys.exit(1)

        # Sort by timestamp (directory name) and get the latest
        latest_checkpoint = sorted(checkpoint_dirs)[-1]
        print(f"Using latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint


def load_model_and_config(checkpoint_dir, device):
    """Load trained model and configuration"""
    # Load configuration
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, mode="r") as f:
        config = json.load(f)

    # Load mean/std
    mean_std_file = os.path.join(checkpoint_dir, "mean_std.txt")
    with open(mean_std_file, mode="r") as f:
        lines = f.readlines()
        mean_line = lines[0] if len(lines) < 4 else lines[1]
        std_line = lines[1] if len(lines) < 4 else lines[3]
        feat_mean = np.array(mean_line.split(), dtype=np.float32)
        feat_std = np.array(std_line.split(), dtype=np.float32)

    # Get base configuration to access paths
    base_config = get_default_config()

    # Load token list
    token_list_path = os.path.join(base_config["token_dir"], "token_list")

    token_list = {0: "<blank>"}
    with open(token_list_path, mode="r") as f:
        for line in f:
            parts = line.split()
            token_list[int(parts[1])] = parts[0]

    # Add eos and sos
    eos_id = len(token_list)
    token_list[eos_id] = "<eos>"
    sos_id = eos_id

    # Create model
    model = AttentionModel(
        dim_in=feat_mean.shape[0],
        dim_enc_hid=config["enc_hidden_dim"],
        dim_enc_proj=config["enc_projection_dim"],
        dim_dec_hid=config["dec_hidden_dim"],
        dim_out=len(token_list),
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

    # Load model weights
    model_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(checkpoint_dir, "final_model.pt")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model, token_list, feat_mean, feat_std, config


def create_test_loader(test_set, feat_mean, feat_std, batch_size=5):
    """Create test data loader"""
    # Get base configuration to access paths
    base_config = get_default_config()

    # Feature and label paths
    feat_dir = os.path.join(os.path.dirname(base_config["feat_dir_train"]), test_set)
    feat_scp_test = os.path.join(feat_dir, "feats.scp")

    label_dir = os.path.join(base_config["token_dir"], test_set)
    label_test = os.path.join(label_dir, f"label_{test_set}")

    # Create dataset
    test_dataset = SequenceDataset(feat_scp_test, label_test, feat_mean, feat_std)

    # Create dataloader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return test_loader


def decode_batch(
    model, features, feat_lens, labels, label_lens, token_list, eos_id, device
):
    """Decode a batch of utterances using greedy decoding"""
    # Sort by frame length
    sorted_lens, indices = torch.sort(feat_lens.view(-1), dim=0, descending=True)
    features = features[indices]
    labels = labels[indices]
    feat_lens = sorted_lens
    label_lens = label_lens[indices]

    # Move to device
    features = features.to(device)

    # Forward pass
    outputs, out_lens = model(features, feat_lens)

    # Decode each utterance
    hypotheses = []
    references = []

    for n in range(outputs.size(0)):
        # Get the original index
        original_idx = torch.nonzero(indices == n, as_tuple=False).view(-1)[0]

        # Get hypothesis by taking argmax at each step
        _, hyp_per_step = torch.max(outputs[original_idx], 1)
        hyp_per_step = hyp_per_step.cpu().numpy()

        # Convert to tokens
        hypothesis = []
        for m in hyp_per_step[: out_lens[original_idx]]:
            if m == eos_id:
                break
            if m in token_list:
                hypothesis.append(token_list[m])
        hypotheses.append(hypothesis)

        # Get reference
        reference = []
        for m in labels[n][: label_lens[n]].cpu().numpy():
            if m in token_list:
                reference.append(token_list[m])
        references.append(reference)

    return hypotheses, references


def evaluate_model(
    model, test_loader, token_list, eos_id, device, output_dir, checkpoint_dir, test_set
):
    """Evaluate model on test data"""
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open output files
    hyp_file = open(os.path.join(output_dir, "hypothesis.txt"), mode="w")
    ref_file = open(os.path.join(output_dir, "reference.txt"), mode="w")
    result_file = open(os.path.join(output_dir, "result.txt"), mode="w")

    # Lists for CTC-style output
    all_predictions = []

    # Evaluation metrics
    total_err = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_length = 0
    total_files = 0

    # For CER/WER calculation
    all_hyp_texts = []
    all_ref_texts = []

    # Progress bar
    pbar = tqdm(test_loader, desc="Evaluating")

    for features, labels, feat_lens, label_lens, utt_ids in pbar:
        # Decode batch
        hypotheses, references = decode_batch(
            model, features, feat_lens, labels, label_lens, token_list, eos_id, device
        )

        # Process each utterance
        for i, (hyp, ref, utt_id) in enumerate(zip(hypotheses, references, utt_ids)):
            # Calculate errors
            (error, substitute, delete, insert, ref_length) = (
                levenshtein.calculate_error(hyp, ref)
            )

            # Accumulate errors
            total_err += error
            total_sub += substitute
            total_del += delete
            total_ins += insert
            total_length += ref_length

            # Write results
            hyp_file.write(f"{utt_id} {' '.join(hyp)}\n")
            ref_file.write(f"{utt_id} {' '.join(ref)}\n")

            # Convert to text for CER/WER
            # Join characters without spaces, only convert <SPACE> to actual space
            hyp_text = "".join([" " if token == "<SPACE>" else token for token in hyp])
            ref_text = "".join([" " if token == "<SPACE>" else token for token in ref])

            all_hyp_texts.append(hyp_text)
            all_ref_texts.append(ref_text)

            # Calculate individual CER/WER
            individual_cer = jiwer.cer(ref_text, hyp_text) if ref_text else 0.0
            individual_wer = jiwer.wer(ref_text, hyp_text) if ref_text else 0.0

            # Store for CTC-style output
            all_predictions.append(
                {
                    "utt_id": utt_id,
                    "hyp": hyp_text,
                    "ref": ref_text,
                    "cer": individual_cer,
                    "wer": individual_wer,
                }
            )

            total_files += 1

            # Write detailed results (attention format)
            result_file.write(f"ID: {utt_id}\n")
            result_file.write(
                f"#ERROR (#SUB #DEL #INS): {error} ({substitute} {delete} {insert})\n"
            )
            result_file.write(f"REF: {' '.join(ref)}\n")
            result_file.write(f"HYP: {' '.join(hyp)}\n")
            result_file.write("\n")

        # Update progress bar
        if total_length > 0:
            current_error = 100.0 * total_err / total_length
            pbar.set_postfix(error=f"{current_error:.2f}%")

    # Calculate final error rates
    if total_length > 0:
        err_rate = 100.0 * total_err / total_length
        sub_rate = 100.0 * total_sub / total_length
        del_rate = 100.0 * total_del / total_length
        ins_rate = 100.0 * total_ins / total_length
    else:
        err_rate = sub_rate = del_rate = ins_rate = 0.0

    # Write summary
    result_file.write("-" * 80 + "\n")
    result_file.write(
        f"#TOKEN: {total_length}, #ERROR: {total_err} "
        f"(#SUB: {total_sub}, #DEL: {total_del}, #INS: {total_ins})\n"
    )
    result_file.write(
        f"TER: {err_rate:.2f}% (SUB: {sub_rate:.2f}, "
        f"DEL: {del_rate:.2f}, INS: {ins_rate:.2f})\n"
    )
    result_file.write("-" * 80 + "\n")

    # Calculate overall CER/WER
    overall_cer = jiwer.cer(all_ref_texts, all_hyp_texts) if all_ref_texts else 0.0
    overall_wer = jiwer.wer(all_ref_texts, all_hyp_texts) if all_ref_texts else 0.0

    # Close files
    hyp_file.close()
    ref_file.close()
    result_file.close()

    # Write CTC-style test_results.txt in checkpoint directory
    ctc_result_path = os.path.join(checkpoint_dir, "test_results.txt")
    with open(ctc_result_path, "w") as ctc_file:
        # Write header
        ctc_file.write("Test Evaluation Results\n")
        ctc_file.write("=" * 60 + "\n")
        ctc_file.write(f"Checkpoint: {checkpoint_dir}/best_model.pt\n")
        ctc_file.write(f"Test dataset: {test_set}\n")
        ctc_file.write(f"Total files: {total_files}\n")
        ctc_file.write(f"Overall CER: {overall_cer:.4f} ({overall_cer * 100:.2f}%)\n")
        ctc_file.write(f"Overall WER: {overall_wer:.4f} ({overall_wer * 100:.2f}%)\n")
        ctc_file.write(f"RTF: N/A (attention model)\n")
        ctc_file.write("\n")
        ctc_file.write("Detailed predictions:\n")
        ctc_file.write("-" * 60 + "\n")
        ctc_file.write("\n")

        # Write all predictions
        for i, pred in enumerate(all_predictions):
            ctc_file.write(
                f"[{i + 1}] CER: {pred['cer']:.3f}, WER: {pred['wer']:.3f}\n"
            )
            ctc_file.write(f"Pred: {pred['hyp']}\n")
            ctc_file.write(f"True: {pred['ref']}\n")
            ctc_file.write("\n")

    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"Total tokens: {total_length}")
    print(f"Total errors: {total_err}")
    print(f"Token Error Rate (TER): {err_rate:.2f}%")
    print(f"  - Substitutions: {sub_rate:.2f}%")
    print(f"  - Deletions: {del_rate:.2f}%")
    print(f"  - Insertions: {ins_rate:.2f}%")
    print(f"Character Error Rate (CER): {overall_cer * 100:.2f}%")
    print(f"Word Error Rate (WER): {overall_wer * 100:.2f}%")
    print("=" * 50)
    print(f"\nOutput saved to: {output_dir}")
    print(f"CTC-style results: {ctc_result_path}")

    return err_rate


def parse_arguments():
    """Parse command line arguments"""
    # Get default config to use for default values
    default_config = get_default_config()
    default_test_set = os.path.basename(
        default_config.get("feat_dir_test", "test_clean")
    )

    parser = argparse.ArgumentParser(
        description="Evaluate trained attention-based ASR model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: use latest checkpoint)",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default=default_test_set,
        help=f"Test set name (default: {default_test_set})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for decoding (default: 5)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint_dir/evaluate_test)",
    )
    return parser.parse_args()


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_arguments()

    # Find checkpoint directory
    checkpoint_dir = find_checkpoint_directory(args.checkpoint)

    # Setup device
    device = setup_device(force_cpu=False)

    # Set multiprocessing start method for MPS
    if device.type == "mps":
        multiprocessing.set_start_method("fork", force=True)

    # Load model and configuration
    model, token_list, feat_mean, feat_std, config = load_model_and_config(
        checkpoint_dir, device
    )

    # Get eos_id from token_list
    eos_id = len(token_list) - 1  # <eos> is the last token

    # Create test data loader
    test_loader = create_test_loader(
        args.test_set, feat_mean, feat_std, args.batch_size
    )

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(checkpoint_dir, "evaluate_test")

    print(f"\nEvaluating model from: {checkpoint_dir}")
    print(f"Test set: {args.test_set}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}\n")

    # Evaluate model
    error_rate = evaluate_model(
        model,
        test_loader,
        token_list,
        eos_id,
        device,
        args.output_dir,
        checkpoint_dir,
        args.test_set,
    )

    print(f"\nResults saved to: {args.output_dir}")
    print(f"Final Token Error Rate: {error_rate:.2f}%")

    # Cleanup
    sys.stdout.flush()
    sys.stderr.flush()

    # Clean up PyTorch resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()

    # Delete model to free memory
    del model

    # Run garbage collection
    import gc

    gc.collect()

    # Exit normally
    sys.exit(0)


if __name__ == "__main__":
    main()
