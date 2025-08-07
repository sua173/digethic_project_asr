#!/usr/bin/env python3
"""
Plot waveform from FLAC audio files.

This script loads FLAC audio files and visualizes their waveforms along with
useful information such as sample rate, duration, and basic statistics.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path


def load_flac_file(filepath):
    """
    Load a FLAC audio file and return audio data with metadata.

    Args:
        filepath (str): Path to the FLAC file

    Returns:
        tuple: (audio_data, sample_rate, duration, channels)
    """
    # Load audio using soundfile for accurate FLAC handling
    audio_data, sample_rate = sf.read(filepath)

    # Handle multi-channel audio
    if audio_data.ndim > 1:
        channels = audio_data.shape[1]
        # Convert to mono for visualization if stereo
        audio_mono = np.mean(audio_data, axis=1)
    else:
        channels = 1
        audio_mono = audio_data

    duration = len(audio_mono) / sample_rate

    return audio_mono, sample_rate, duration, channels


def extract_audio_info(audio_data, sample_rate):
    """
    Extract useful information from audio data.

    Args:
        audio_data (np.array): Audio signal
        sample_rate (int): Sample rate in Hz

    Returns:
        dict: Dictionary containing audio statistics
    """
    info = {
        "sample_rate": sample_rate,
        "duration": len(audio_data) / sample_rate,
        "num_samples": len(audio_data),
        "max_amplitude": np.max(np.abs(audio_data)),
        "mean_amplitude": np.mean(np.abs(audio_data)),
        "rms": np.sqrt(np.mean(audio_data**2)),
        "dynamic_range_db": 20
        * np.log10(
            np.max(np.abs(audio_data)) / (np.sqrt(np.mean(audio_data**2)) + 1e-10)
        ),
    }

    # Calculate zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
    info["zero_crossing_rate"] = zero_crossings / len(audio_data)

    return info


def plot_waveform(audio_data, sample_rate, filepath, info_dict, save_path=None):
    """
    Plot the waveform with multiple views and save as separate files.

    Args:
        audio_data (np.array): Audio signal
        sample_rate (int): Sample rate in Hz
        filepath (str): Path to the audio file (for title)
        info_dict (dict): Dictionary containing audio statistics
        save_path (str): Base path for saving images (without extension)
    
    Returns:
        list: List of matplotlib figures (4 separate figures)
    """
    # Create time axis
    time = np.arange(len(audio_data)) / sample_rate
    
    # Prepare data for all plots
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)
    
    figures = []

    # Figure 1: Waveform (Time domain)
    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(time, audio_data, linewidth=0.5, color="blue", alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="r", linestyle="-", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    figures.append(fig1)
    if save_path:
        fig1.savefig(f"{save_path}_waveform.png", dpi=150)

    # Figure 2: Power Spectrum (Frequency domain)
    fig2 = plt.figure(figsize=(12, 6))
    power_spectrum = np.mean(magnitude**2, axis=1)
    power_spectrum_db = 10 * np.log10(power_spectrum + 1e-10)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    plt.semilogx(frequencies[1:], power_spectrum_db[1:], linewidth=1.5, color="red")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.grid(True, alpha=0.3, which="both")
    plt.xlim([20, min(8000, sample_rate / 2)])
    plt.tight_layout()
    figures.append(fig2)
    if save_path:
        fig2.savefig(f"{save_path}_power_spectrum.png", dpi=150)

    # Figure 3: Spectrogram (Time-Frequency domain)
    fig3 = plt.figure(figsize=(12, 6))
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    plt.imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, info_dict["duration"], 0, sample_rate / 2],
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim([0, min(8000, sample_rate / 2)])  # Focus on speech frequencies
    plt.tight_layout()
    figures.append(fig3)
    if save_path:
        fig3.savefig(f"{save_path}_spectrogram.png", dpi=150)

    # Figure 4: Mel Spectrogram (Perceptual scale)
    fig4 = plt.figure(figsize=(12, 6))
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=80
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.imshow(
        mel_spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, info_dict["duration"], 0, 80],
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mel bin")
    plt.tight_layout()
    figures.append(fig4)
    if save_path:
        fig4.savefig(f"{save_path}_mel_spectrogram.png", dpi=150)

    return figures


def main():
    """Main function to handle command-line arguments and execute plotting."""
    parser = argparse.ArgumentParser(
        description="Plot waveform and analyze FLAC audio files"
    )
    parser.add_argument("audio_file", type=str, help="Path to the FLAC audio file")
    parser.add_argument(
        "--save", type=str, default=None, help="Base name for saving plots (e.g., output will create output_waveform.png, etc.)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the plot (useful when only saving)",
    )

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: File '{args.audio_file}' not found!")
        sys.exit(1)

    # Check if file is FLAC
    if not args.audio_file.lower().endswith(".flac"):
        print(f"Warning: File may not be a FLAC file. Attempting to load anyway...")

    try:
        # Load audio file
        print(f"Loading audio file: {args.audio_file}")
        audio_data, sample_rate, duration, channels = load_flac_file(args.audio_file)

        print(f"Successfully loaded:")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Duration: {duration:.2f} seconds")
        print(f"  - Channels: {channels}")

        # Extract audio information
        info = extract_audio_info(audio_data, sample_rate)

        # Plot waveform
        print("Generating plots...")
        figures = plot_waveform(audio_data, sample_rate, args.audio_file, info, args.save)

        # Save plot if requested
        if args.save:
            print(f"Plots saved with base name: {args.save}")
            print(f"  - {args.save}_waveform.png")
            print(f"  - {args.save}_power_spectrum.png")
            print(f"  - {args.save}_spectrogram.png")
            print(f"  - {args.save}_mel_spectrogram.png")

        # Display plot unless --no-display is set
        if not args.no_display:
            print("Displaying plots...")
            plt.show()
        else:
            for fig in figures:
                plt.close(fig)

        print("Done!")

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
