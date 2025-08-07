# Unified Label Conversion Script for Subsets
# Uses existing token_list to generate label files for any subset.

# Usage:
#     # For train_clean_100_1hr
#     python src/tools/convert_labels.py --dataset train_clean_100_1hr

#     # For dev_clean_1hr (character-level only)
#     python src/tools/convert_labels.py --dataset dev_clean_1hr

#     # With custom paths
#     python src/tools/convert_labels.py --dataset test_clean_1hr --base_dir generated/attention

import argparse
import os
import shutil


def convert_labels(dataset_name, base_dir="generated/attention"):
    """Generate label files for specified dataset (character-level only)"""

    # Set directory paths
    token_dir = os.path.join(base_dir, "tokens")
    label_dir = os.path.join(base_dir, "labels", dataset_name)
    output_dir = os.path.join(base_dir, "tokens", dataset_name)

    # Required file paths
    token_list_path = os.path.join(token_dir, "token_list")
    label_file = os.path.join(label_dir, "text")
    output_label = os.path.join(output_dir, f"label_{dataset_name}")

    # Check file existence
    if not os.path.exists(token_list_path):
        raise FileNotFoundError(f"Token list not found: {token_list_path}")
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load token list
    token_to_id = {}
    with open(token_list_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                token = parts[0]
                token_id = int(parts[1])
                token_to_id[token] = token_id

    print(f"Loaded {len(token_to_id)} tokens from {token_list_path}")

    # Convert label file
    converted_count = 0
    with open(label_file, "r") as fin, open(output_label, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            utt_id = parts[0]
            tokens = parts[1:]

            # Convert tokens to IDs
            token_ids = []
            for token in tokens:
                if token in token_to_id:
                    token_ids.append(str(token_to_id[token]))
                else:
                    print(f"Warning: Unknown token '{token}' in utterance {utt_id}")

            if token_ids:
                fout.write(f"{utt_id} {' '.join(token_ids)}\n")
                converted_count += 1

    print(f"Converted {converted_count} utterances")
    print(f"Output saved to: {output_label}")

    return output_label


def main():
    parser = argparse.ArgumentParser(
        description="Convert labels for LibriSpeech subsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., train_clean_100_1hr, dev_clean_1hr)",
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="generated/attention",
        help="Base directory for input/output (default: generated/attention)",
    )

    args = parser.parse_args()

    try:
        convert_labels(args.dataset, args.base_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure to run prepare_data.py first to generate token_list")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
