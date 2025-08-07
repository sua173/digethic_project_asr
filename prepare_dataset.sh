#!/bin/bash

# Prepare dataset for CTC and Attention models
# This script:
# 1. Downloads LibriSpeech datasets if not present
# 2. Creates subsets for CTC model training
# 3. Prepares data for Attention model training

set -e  # Exit on error

echo "="
echo "Dataset Preparation for Speech Recognition Models"
echo "="
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

# Step 1: Download LibriSpeech datasets if not already present
echo "Step 1: Checking and downloading LibriSpeech datasets..."
echo "-"

# Check for each dataset and download if missing
if [ ! -d "data/dev-clean" ]; then
    echo "Downloading dev-clean dataset..."
    python src/common/download_librispeech.py --dev
else
    echo "dev-clean already exists, skipping download"
fi

if [ ! -d "data/test-clean" ]; then
    echo "Downloading test-clean dataset..."
    python src/common/download_librispeech.py --test
else
    echo "test-clean already exists, skipping download"
fi

if [ ! -d "data/train-clean-100" ]; then
    echo "Downloading train-clean-100 dataset..."
    python src/common/download_librispeech.py --train
else
    echo "train-clean-100 already exists, skipping download"
fi

echo ""
echo "LibriSpeech datasets ready!"
echo ""

# Step 2: Create datasets for CTC model
echo "Step 2: Creating datasets for CTC model..."
echo "-"

# Create all CTC datasets (train 10h, dev 1h, test full)
if [ ! -d "generated/subset_10h" ] || [ ! -d "generated/dev_subset" ] || [ ! -d "generated/test_dataset" ]; then
    echo "Creating CTC training datasets..."
    python src/ctc/tools/create_dataset.py --mode all
    echo "CTC datasets created:"
    echo "  - Training: generated/subset_10h/train_manifest.jsonl (10 hours)"
    echo "  - Validation: generated/dev_subset/val_manifest.jsonl (1 hour)"
    echo "  - Test: generated/test_dataset/test_manifest.jsonl (full test-clean)"
else
    echo "CTC datasets already exist, skipping creation"
fi

echo ""

# Step 3: Data Preparation for Attention model (train 10h, dev 32min, test full)
echo "Step 3: Preparing data for Attention model..."
echo "-"
echo "Creating Attention model data subsets..."
python src/attention/tools/prepare_data.py --task all --dataset train-clean-100 --librispeech_dir data --output_dir generated/attention --subset --duration 36000
python src/attention/tools/prepare_data.py --task all --dataset dev-clean --librispeech_dir data --output_dir generated/attention --subset --duration 1920
python src/attention/tools/prepare_data.py --task all --dataset test-clean --librispeech_dir data --output_dir generated/attention

echo ""

# Step 4: Feature Extraction for Attention model
echo "Step 4: Extracting features for Attention model..."
echo "-"
python src/attention/tools/extract_features.py --dataset train_clean_100_10hr --librispeech_dir generated/attention --output_dir generated/attention/features --compute_mean_std
python src/attention/tools/extract_features.py --dataset dev_clean_32min --librispeech_dir generated/attention --output_dir generated/attention/features
python src/attention/tools/extract_features.py --dataset test_clean --librispeech_dir generated/attention --output_dir generated/attention/features

echo ""
echo "="
echo "Dataset preparation complete!"
echo "="
echo ""
echo "Summary:"
echo "  CTC Model:"
echo "    - Training: generated/subset_10h/train_manifest.jsonl (10 hours)"
echo "    - Validation: generated/dev_subset/val_manifest.jsonl (1 hour)"
echo "    - Test: generated/test_dataset/test_manifest.jsonl (full test-clean)"
echo ""
echo "  Attention Model:"
echo "    - Training: generated/attention/train_clean_100_10hr/ (10 hours)"
echo "    - Validation: generated/attention/dev_clean_32min/ (32 minutes)"
echo "    - Test: generated/attention/test_clean/ (full test-clean)"
echo "    - Features: generated/attention/features/"
echo ""
echo "You can now train the models:"
echo "  CTC: python src/ctc/train.py"
echo "  Attention: python src/attention/train.py"
