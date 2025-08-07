# LibriSpeech Data Preparation Pipeline

# This script integrates the following processes:
# 1. Generate label files from LibriSpeech data (prepare)
# 2. Create subsets (subset) - optional
# 3. Create token list and convert labels to numerical format (token)

# Usage:
#     # Prepare full dataset
#     python src/attention/tools/prepare_data.py --task all --dataset train-clean-100

#     # Prepare with subset creation
#     python src/attention/tools/prepare_data.py --task all --dataset train-clean-100 --subset --duration 3600

#     # Token processing only
#     python src/attention/tools/prepare_data.py --task token

import argparse
import os
import sys
from pathlib import Path
import random

try:
    import soundfile as sf

    def get_audio_duration(file_path):
        """Get audio duration using soundfile"""
        info = sf.info(str(file_path))
        return info.duration
except ImportError:

    def get_audio_duration(file_path):
        """Get audio duration (approximate)"""
        # For FLAC files, this is just an approximation
        return 10.0


def read_transcripts(trans_file):
    """Read transcript file and return dictionary"""
    transcripts = {}
    with open(trans_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                utt_id = parts[0]
                text = parts[1].upper()  # Convert to uppercase
                transcripts[utt_id] = text
    return transcripts


def text_to_chars(text):
    """Convert text to character sequence with <SPACE> tokens"""
    chars = []
    for char in text:
        if char == " ":
            chars.append("<SPACE>")
        else:
            chars.append(char)
    return chars


def process_speaker(speaker_dir, transcripts):
    """Process all utterances for a speaker"""
    wav_scp_entries = []
    text_char_entries = []

    # Get all FLAC files in speaker directory
    flac_files = sorted(list(speaker_dir.glob("*.flac")))

    for flac_file in flac_files:
        utt_id = flac_file.stem

        if utt_id in transcripts:
            # wav.scp entry (using absolute path)
            wav_scp_entries.append(f"{utt_id} {flac_file.absolute()}")

            # Character-based text
            text = transcripts[utt_id]
            chars = text_to_chars(text)
            text_char_entries.append(f"{utt_id} {' '.join(chars)}")

    return wav_scp_entries, text_char_entries


def prepare_librispeech(librispeech_dir, dataset, output_dir):
    """Prepare LibriSpeech data"""
    # Convert dataset name (train-clean-100 -> train_clean_100)
    dataset_name = dataset.replace("-", "_")

    # Setup paths
    librispeech_path = Path(librispeech_dir)
    dataset_path = librispeech_path / dataset

    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return False

    # Create output directory
    output_dataset_dir = os.path.join(output_dir, "labels", dataset_name)
    os.makedirs(output_dataset_dir, exist_ok=True)

    # Initialize lists for all entries
    all_wav_scp = []
    all_text_char = []

    # Process each chapter
    for speaker_dir in sorted(dataset_path.iterdir()):
        if not speaker_dir.is_dir():
            continue

        print(f"Processing speaker {speaker_dir.name}")

        for chapter_dir in sorted(speaker_dir.iterdir()):
            if not chapter_dir.is_dir():
                continue

            # Read transcript file
            trans_file = (
                chapter_dir / f"{speaker_dir.name}-{chapter_dir.name}.trans.txt"
            )
            if not trans_file.exists():
                print(f"Warning: Transcript file not found: {trans_file}")
                continue

            transcripts = read_transcripts(trans_file)

            # Process all utterances in this chapter
            wav_scp, text_char = process_speaker(chapter_dir, transcripts)

            all_wav_scp.extend(wav_scp)
            all_text_char.extend(text_char)

    # Sort all entries by utterance ID
    all_wav_scp.sort()
    all_text_char.sort()

    # Write output files
    with open(os.path.join(output_dataset_dir, "wav.scp"), "w") as f:
        for entry in all_wav_scp:
            f.write(entry + "\n")

    with open(os.path.join(output_dataset_dir, "text"), "w") as f:
        for entry in all_text_char:
            f.write(entry + "\n")

    # Phone and kana files are not created as they are not supported

    print(f"\nProcessed {len(all_wav_scp)} utterances")
    print(f"Output files written to: {output_dataset_dir}")

    return True


def extract_subset(label_dir, output_dir, target_duration, subset_name):
    """Extract subset of utterances up to target duration"""

    # Read wav.scp to get audio file paths
    wav_scp_path = os.path.join(label_dir, "wav.scp")
    if not os.path.exists(wav_scp_path):
        print(f"Error: wav.scp not found at {wav_scp_path}")
        return False

    # Read all utterances
    utterances = []
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt_id = parts[0]
                audio_path = parts[1]
                utterances.append((utt_id, audio_path))

    # Shuffle for random selection
    random.seed(42)  # For reproducibility
    random.shuffle(utterances)

    # Select utterances up to target duration
    selected_utterances = []
    total_duration = 0.0

    for utt_id, audio_path in utterances:
        if os.path.exists(audio_path):
            duration = get_audio_duration(audio_path)
            if total_duration + duration <= target_duration:
                selected_utterances.append(utt_id)
                total_duration += duration
            else:
                break

    print(f"Selected {len(selected_utterances)} utterances")
    print(
        f"Total duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)"
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subset files
    selected_set = set(selected_utterances)

    # Copy files while filtering by selected utterances
    for filename in ["wav.scp", "text"]:
        input_file = os.path.join(label_dir, filename)
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(input_file):
            with open(input_file, "r") as fin, open(output_file, "w") as fout:
                for line in fin:
                    utt_id = line.strip().split()[0] if line.strip() else ""
                    if utt_id in selected_set:
                        fout.write(line)

            # Sort the output file
            with open(output_file, "r") as f:
                lines = f.readlines()
            lines.sort()
            with open(output_file, "w") as f:
                f.writelines(lines)

    print(f"Subset files created in: {output_dir}")
    return True


def token_to_int(label_file_str, label_file_int, unknown_list_file, token_list):
    """Convert text labels to numerical labels using token list"""

    with (
        open(label_file_str, mode="r") as label_in,
        open(label_file_int, mode="w") as label_out,
        open(unknown_list_file, mode="w") as unk_list,
    ):
        for line in label_in:
            text = line.split()

            if len(text) < 2:
                continue

            # Write utterance ID
            label_out.write("%s" % text[0])

            # Convert each token to number
            for u in text[1:]:
                if u in token_list:
                    label_out.write(" %d" % token_list[u])
                else:
                    # Unknown token
                    label_out.write(" %d" % token_list["<UNK>"])
                    unk_list.write("%s\n" % u)

            label_out.write("\n")


def create_tokens(output_dir):
    """Create token list and convert labels"""

    # Input label directories
    label_base_dir = os.path.join(output_dir, "labels")

    # Output directory
    token_dir = os.path.join(output_dir, "tokens")
    os.makedirs(token_dir, exist_ok=True)

    # Initialize token set
    token_set = set()

    # Collect tokens from all available datasets
    for dataset_name in os.listdir(label_base_dir):
        label_dir = os.path.join(label_base_dir, dataset_name)
        if not os.path.isdir(label_dir):
            continue

        label_file = os.path.join(label_dir, "text")

        if not os.path.exists(label_file):
            continue

        print(f"Collecting tokens from {dataset_name}")

        with open(label_file, mode="r") as f:
            for line in f:
                text = line.split()
                if len(text) > 1:
                    for u in text[1:]:
                        token_set.add(u)

    # Sort tokens
    token_sorted = sorted(list(token_set))

    # Create token list with IDs
    # 0 is reserved for blank
    token_list = {"<BLANK>": 0}
    token_list["<UNK>"] = 1

    for i, token in enumerate(token_sorted):
        token_list[token] = i + 2

    # Write token list
    token_list_path = os.path.join(token_dir, "token_list")
    with open(token_list_path, mode="w") as f:
        for token, idx in sorted(token_list.items(), key=lambda x: x[1]):
            if token != "<BLANK>":  # Skip blank token
                f.write(f"{token} {idx}\n")

    print(f"Token list written to: {token_list_path}")
    print(f"Number of tokens: {len(token_list) - 1}")  # Excluding <BLANK>

    # Convert labels for each dataset
    for dataset_name in os.listdir(label_base_dir):
        label_dir = os.path.join(label_base_dir, dataset_name)
        if not os.path.isdir(label_dir):
            continue

        label_file_str = os.path.join(label_dir, "text")

        if not os.path.exists(label_file_str):
            continue

        # Create output directory for this dataset
        dataset_token_dir = os.path.join(token_dir, dataset_name)
        os.makedirs(dataset_token_dir, exist_ok=True)

        label_file_int = os.path.join(dataset_token_dir, f"label_{dataset_name}")
        unknown_list_file = os.path.join(
            dataset_token_dir, f"unknown_{dataset_name}.txt"
        )

        print(f"Converting labels for {dataset_name}...")
        token_to_int(
            label_file_str,
            label_file_int,
            unknown_list_file,
            token_list,
        )

        # Sort the output file
        with open(label_file_int, "r") as f:
            lines = f.readlines()
        lines.sort()
        with open(label_file_int, "w") as f:
            f.writelines(lines)

    print(f"Token processing completed. Output directory: {token_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="LibriSpeech data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["prepare", "subset", "token", "all"],
        default="all",
        help="Task to perform (default: all)",
    )

    # Prepare task arguments
    parser.add_argument(
        "--librispeech_dir",
        type=str,
        default="/path/to/LibriSpeech",
        help="Path to LibriSpeech root directory",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["train-clean-100", "dev-clean", "test-clean"],
        help="Dataset to process",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated/attention",
        help="Output directory (default: generated/attention)",
    )

    # Subset task arguments
    parser.add_argument(
        "--subset", action="store_true", help="Create subset after preparing data"
    )

    parser.add_argument("--duration", type=int, help="Duration of subset in seconds")

    args = parser.parse_args()

    # Execute tasks
    if args.task in ["prepare", "all"]:
        if not args.dataset:
            parser.error("--dataset is required for prepare task")

        print(f"\n=== Preparing {args.dataset} ===")
        success = prepare_librispeech(
            args.librispeech_dir, args.dataset, args.output_dir
        )
        if not success:
            sys.exit(1)

    if args.subset and args.task in ["subset", "all"]:
        if not args.duration:
            parser.error("--duration is required when --subset is specified")

        print(f"\n=== Creating {args.duration}s subset ===")

        # Setup paths
        dataset_name = args.dataset.replace("-", "_")
        label_dir = os.path.join(args.output_dir, "labels", dataset_name)

        # Create subset name
        hours = args.duration // 3600
        if hours >= 1:
            subset_suffix = f"{hours}hr"
        else:
            minutes = args.duration // 60
            subset_suffix = f"{minutes}min"

        subset_name = f"{dataset_name}_{subset_suffix}"
        output_subset_dir = os.path.join(args.output_dir, "labels", subset_name)

        success = extract_subset(
            label_dir, output_subset_dir, args.duration, subset_name
        )
        if not success:
            sys.exit(1)

    if args.task in ["token", "all"]:
        print("\n=== Creating token list (character-level) ===")
        success = create_tokens(args.output_dir)
        if not success:
            sys.exit(1)

    print("\n=== Data preparation completed ===")


if __name__ == "__main__":
    main()
