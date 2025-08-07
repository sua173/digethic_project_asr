#!/usr/bin/env python3
"""
Dataloader test script

Usage:
    python src/ctc/tests/test_dataloader.py
"""

import sys
import os

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

import torch
from src.ctc.core.utils import TextProcessor, setup_device
from src.ctc.core.dataset import create_data_loaders


def test_dataloader():
    """Test dataloader functionality"""

    print("=== Data Loader Test ===")

    # Device setup
    device = setup_device()

    # Initialize text processor
    text_processor = TextProcessor()
    print(f"Vocabulary size: {text_processor.vocab_size}")
    print(f"Characters: {text_processor.chars}")
    print(f"Blank ID: {text_processor.blank_id}")

    # Text conversion test
    test_text = "HELLO WORLD"
    ids = text_processor.text_to_ids(test_text)
    decoded = text_processor.ids_to_text(ids)
    print(f"\nText conversion test:")
    print(f"Original: '{test_text}'")
    print(f"IDs: {ids}")
    print(f"Decoded: '{decoded}'")

    # CTC decode test
    ctc_output = [
        8,
        8,
        5,
        12,
        12,
        15,
        27,
        27,
        23,
        15,
        18,
        12,
        4,
    ]  # "HELLO WORLD"のCTC出力例
    ctc_decoded = text_processor.decode_ctc(ctc_output)
    print(f"CTC decode test: {ctc_output} -> '{ctc_decoded}'")

    # Create data loaders
    print(f"\n=== Creating Data Loaders ===")
    try:
        train_loader, val_loader = create_data_loaders(
            "generated/subset_10h/train_manifest.jsonl",
            "generated/dev_subset/val_manifest.jsonl",
            text_processor,
            batch_size=4,
            num_workers=0,  # For testing, set to 0 to avoid multiprocessing issues
        )

        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")

        # Test first batch
        print("\n=== Testing First Batch ===")
        batch = next(iter(train_loader))

        print(f"Batch keys: {list(batch.keys())}")
        print(f"Spectrograms shape: {batch['spectrograms'].shape}")
        print(f"Texts shape: {batch['texts'].shape}")
        print(f"Spec lengths: {batch['spec_lengths']}")
        print(f"Text lengths: {batch['text_lengths']}")

        # Move to device
        spectrograms = batch["spectrograms"].to(device)
        texts = batch["texts"].to(device)
        print(f"Successfully moved to {device}")

        # Print sample spectrogram shape
        print(f"\n=== Sample Text Decoding ===")
        for i in range(min(2, batch["texts"].shape[0])):
            text_ids = batch["texts"][i][: batch["text_lengths"][i]]
            decoded_text = text_processor.ids_to_text(text_ids.tolist())
            print(f"Sample {i}: '{decoded_text}'")

        print(f"\n✅ Data loader test passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_dataloader()
