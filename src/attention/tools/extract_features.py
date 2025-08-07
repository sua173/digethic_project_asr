# Feature extraction script for LibriSpeech
# Extracts Fbank features from wav files and saves in Kaldi format

import os
import numpy as np
import argparse
import struct
import librosa
import scipy.signal
from pathlib import Path


def compute_fbank(
    wav_file,
    sample_frequency=16000,
    num_mel_bins=40,
    frame_length=25,
    frame_shift=10,
    use_log_fbank=True,
    use_power=False,
):
    """
    Extract log-mel spectrogram features from wav/flac file

    Args:
        wav_file: Path to audio file
        sample_frequency: Sampling frequency
        num_mel_bins: Number of mel filter banks
        frame_length: Frame length in milliseconds
        frame_shift: Frame shift in milliseconds
        use_log_fbank: Whether to take logarithm
        use_power: Whether to use power spectrum

    Returns:
        fbank_features: Fbank features (time, num_mel_bins)
    """
    # Load audio file
    wav, sr = librosa.load(wav_file, sr=sample_frequency)

    # Convert frame length and shift to number of samples
    n_fft = int(sample_frequency * frame_length / 1000)
    hop_length = int(sample_frequency * frame_shift / 1000)

    # Compute mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_frequency,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=num_mel_bins,
        power=2.0 if use_power else 1.0,
    )

    # Transpose to (time, num_mel_bins) shape
    mel_spectrogram = mel_spectrogram.T

    # Take logarithm
    if use_log_fbank:
        mel_spectrogram = np.log(mel_spectrogram + 1e-10)

    return mel_spectrogram.astype(np.float32)


def write_kaldi_matrix(file_path, matrix):
    """
    Save features as Kaldi format binary file

    Args:
        file_path: Output file path
        matrix: Feature matrix (time, dim)
    """
    with open(file_path, "wb") as f:
        # Kaldi binary header
        f.write(b"\x00\x00B")  # Binary flag

        # Matrix size information
        rows, cols = matrix.shape
        f.write(b"\x04")  # Size of int
        f.write(struct.pack("<i", rows))
        f.write(b"\x04")  # Size of int
        f.write(struct.pack("<i", cols))

        # Write data
        matrix.astype(np.float32).tofile(f)


def extract_features(wav_scp, output_dir, compute_mean_std=False):
    """
    Extract features from audio files listed in wav.scp

    Args:
        wav_scp: Path to wav.scp file
        output_dir: Output directory for feature files
        compute_mean_std: Whether to compute mean and standard deviation
    """
    os.makedirs(output_dir, exist_ok=True)
    feat_scp_path = os.path.join(output_dir, "feats.scp")

    all_features = []

    with open(wav_scp, "r") as scp_in, open(feat_scp_path, "w") as scp_out:
        lines = scp_in.readlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            utterance_id = parts[0]
            wav_path = parts[1]

            if i % 100 == 0:
                print(f"Processing {i}/{len(lines)} utterances...")

            # Extract features
            try:
                features = compute_fbank(wav_path)

                # Save features to file
                feat_path = os.path.join(output_dir, f"{utterance_id}.ark")
                write_kaldi_matrix(feat_path, features)

                # Write to feats.scp
                scp_out.write(f"{utterance_id} {os.path.abspath(feat_path)}\n")

                # Save features for mean/std calculation
                if compute_mean_std:
                    all_features.append(features)

            except Exception as e:
                print(f"Error processing {utterance_id}: {e}")
                continue

    # Calculate and save mean and standard deviation
    if compute_mean_std and all_features:
        all_features = np.vstack(all_features)
        mean = np.mean(all_features, axis=0)
        std = np.std(all_features, axis=0)

        mean_std_path = os.path.join(output_dir, "mean_std.txt")
        with open(mean_std_path, "w") as f:
            # Write mean values
            f.write(" ".join(map(str, mean)) + "\n")
            # Write standard deviation
            f.write(" ".join(map(str, std)) + "\n")

        print(f"Mean and std saved to: {mean_std_path}")

    # LibriSpeech support: Sort feats.scp
    # To maintain consistency with label files
    print(f"Sorting {feat_scp_path}...")
    with open(feat_scp_path, "r") as f:
        lines = f.readlines()
    lines.sort()
    with open(feat_scp_path, "w") as f:
        f.writelines(lines)

    print(f"Feature extraction completed. feats.scp saved to: {feat_scp_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract features from LibriSpeech")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to process (e.g., train_clean_100, dev_clean, test_clean, train_clean_100_1hr)",
    )
    parser.add_argument(
        "--librispeech_dir",
        type=str,
        default="generated/attention",
        help="LibriSpeech data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated/attention/features",
        help="Output directory for features",
    )
    parser.add_argument(
        "--compute_mean_std",
        action="store_true",
        help="Compute mean and std for training data",
    )

    args = parser.parse_args()

    # wav.scp file path
    wav_scp = os.path.join(args.librispeech_dir, "labels", args.dataset, "wav.scp")

    if not os.path.exists(wav_scp):
        raise FileNotFoundError(f"wav.scp not found: {wav_scp}")

    # Output directory
    output_dataset_dir = os.path.join(args.output_dir, args.dataset)

    print(f"Extracting features for {args.dataset}...")
    print(f"Input wav.scp: {wav_scp}")
    print(f"Output directory: {output_dataset_dir}")

    # Run feature extraction
    # Set compute_mean_std to True for training data
    # (Dataset names starting with train_ are considered training data)
    is_train_data = args.dataset.startswith("train_")
    extract_features(
        wav_scp,
        output_dataset_dir,
        compute_mean_std=(is_train_data and args.compute_mean_std),
    )


if __name__ == "__main__":
    main()
