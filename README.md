# LibriSpeech Speech Recognition

A research project implementing end-to-end speech recognition models using the LibriSpeech dataset. This project demonstrates two distinct approaches: CTC-based and Attention-based models.

## Project Overview

This project aims to implement and compare different end-to-end ASR architectures


## Architecture

### 1. CTC Model (Connectionist Temporal Classification)
- **Encoder**: 2-layer CNN (feature extraction) + 4-layer BiLSTM
- **Decoder**: CTC loss with greedy decoding
- **Features**: Mel-spectrogram (80 filters, 16kHz)
- **Training**: Adam optimizer with ReduceLROnPlateau
- **Status**: Complete and tested

### 2. Attention-Based Encoder-Decoder Model
- **Encoder**: 3-layer Bidirectional GRU with subsampling
- **Decoder**: LSTM with Location-Aware Attention
- **Features**: 40-dimensional Fbank features (16kHz, Kaldi format)
- **Training**: Adam optimizer (lr=1e-4) with manual learning rate decay
- **Status**: In development

## Quick Start

### Prerequisites
- Python 3.13.5+
- 64GB RAM recommended (for full dataset training)
- Mac/Linux/Windows (with WSL)
- ~10GB disk space for LibriSpeech data

### Installation
```bash
# Clone repository
git clone <repository-url>
cd digethic_project_asr

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Quick setup: Download data and prepare all datasets (recommended)
chmod +x prepare_dataset.sh  # Make script executable (first time only)
./prepare_dataset.sh         # Download LibriSpeech and prepare datasets
```

## Detailed Usage Guide

### 1. Data Preparation

#### Quick Setup (Recommended)

Use the all-in-one preparation script to download LibriSpeech and create all necessary datasets:

```bash
# Download LibriSpeech and prepare all datasets for both models
./prepare_dataset.sh
```

This script automatically:
1. Downloads LibriSpeech datasets (dev-clean, test-clean, train-clean-100) if not present
2. Creates CTC model datasets (10h train, 1h validation, full test)
3. Prepares Attention model data (10h train, 32min validation, full test)
4. Extracts features for Attention model

Total download size: ~7GB (only downloads missing datasets)
Estimated time: 15-30 minutes (depending on internet speed)

#### Manual Setup

If you prefer to set up datasets manually or need custom configurations:

##### Download LibriSpeech Datasets
```bash
# Download all datasets at once
python src/common/download_librispeech.py

# Or download individually:
python src/common/download_librispeech.py --dev   # dev-clean (337MB)
python src/common/download_librispeech.py --test  # test-clean (346MB)
python src/common/download_librispeech.py --train # train-clean-100 (6.3GB)
```

##### Create Training Subsets (CTC)
```bash
# Create all datasets at once (recommended)
python src/ctc/tools/create_dataset.py --mode all

# Or create individual datasets:
# Create 10-hour training subset
python src/ctc/tools/create_dataset.py --mode train --duration 36000 --output generated/subset_10h

# Create 1-hour validation subset
python src/ctc/tools/create_dataset.py --mode val --duration 3600 --output generated/dev_subset

# Create test manifest from entire test-clean
python src/ctc/tools/create_dataset.py --mode test --output generated/test_dataset
```

**Parameters:**
- `--mode`: Dataset creation mode: train, val, test, or all
- `--data_dir`: Path to LibriSpeech data directory (auto-selected based on mode)
- `--duration`: Target duration in seconds (0 = all files)
- `--output`: Output directory/file path

##### Prepare Data for Attention Model
```bash
# Step 1: Create data subsets
python src/attention/tools/prepare_data.py --task all --dataset train-clean-100 --librispeech_dir data --output_dir generated/attention --subset --duration 36000
python src/attention/tools/prepare_data.py --task all --dataset dev-clean --librispeech_dir data --output_dir generated/attention --subset --duration 1920
python src/attention/tools/prepare_data.py --task all --dataset test-clean --librispeech_dir data --output_dir generated/attention

# Step 2: Extract features (Fbank)
python src/attention/tools/extract_features.py --dataset train_clean_100_10hr --librispeech_dir generated/attention --output_dir generated/attention/features --compute_mean_std
python src/attention/tools/extract_features.py --dataset dev_clean_32min --librispeech_dir generated/attention --output_dir generated/attention/features
python src/attention/tools/extract_features.py --dataset test_clean --librispeech_dir generated/attention --output_dir generated/attention/features
```

**Note**: These steps are automatically done if you run `./prepare_dataset.sh`

### 2. Training Models

#### Train CTC Model
```bash
# Default configuration (10-hour subset, 60 epochs)
python src/ctc/train.py

# Custom parameters
python src/ctc/train.py --epochs 50 --batch_size 8 --lr 1e-3

# Specify device
python src/ctc/train.py --device cpu  # Force CPU (recommended for CTC)
python src/ctc/train.py --device cuda  # Force GPU
python src/ctc/train.py --device mps  # Force MPS (may have issues)

# Full dataset training
python src/ctc/train.py --train_manifest data/train-clean-100/train_manifest.jsonl \
                        --val_manifest data/dev-clean/dev_manifest.jsonl
```

**Parameters:**
- `--train_manifest`: Path to training manifest file
- `--val_manifest`: Path to validation manifest file
- `--epochs`: Number of training epochs (default: 30)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 3e-4)
- `--hidden_dim`: LSTM hidden dimension (default: 256)
- `--num_layers`: Number of LSTM layers (default: 3)
- `--save_dir`: Checkpoint directory (default: generated/checkpoints)
- `--device`: Device to use: cuda, mps, or cpu (default: auto-detect)

#### Train Attention-Based Model
```bash
# Default configuration (10-hour subset, 60 epochs)
python src/attention/train.py

# Custom parameters
python src/attention/train.py --epochs 100 --batch-size 16
```

**Note**: Make sure data preparation is complete (run `./prepare_dataset.sh` or see Data Preparation section)

### 3. Model Evaluation

#### Evaluate CTC Model
```bash
# Evaluate on test dataset (uses generated/test_dataset/test_manifest.jsonl)
python src/ctc/evaluate.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt

# Specify custom test manifest
python src/ctc/evaluate.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt \
                           --test-manifest custom_test_manifest.jsonl

# Adjust batch size for performance
python src/ctc/evaluate.py --checkpoint generated/checkpoints/ctc_*/best_ctc_model.pt \
                           --batch-size 16 --device cpu
```

**Parameters:**
- `--checkpoint`: Path to model checkpoint (required)
- `--test-manifest`: Path to test manifest file (default: generated/test_dataset/test_manifest.jsonl)
- `--batch-size`: Batch size for evaluation (default: 8)
- `--num-workers`: Number of data loading workers (default: 0)
- `--device`: Device to use: cuda, mps, or cpu (default: auto-detect)

**Output:**
- Overall CER/WER metrics
- Performance statistics (RTF, processing speed)
- Error distribution histograms
- Best/worst predictions
- Detailed results saved to checkpoint directory

#### Evaluate Attention Model
```bash
# Evaluate on test set
python src/attention/evaluate.py --checkpoint generated/checkpoints/attention_*/best_model.pt
```

### 4. Analysis Tools

#### Analyze Model Checkpoints
```bash
# Analyze latest checkpoint (auto-detect)
python src/common/analyze_model.py

# Analyze specific checkpoint
python src/common/analyze_model.py generated/checkpoints/ctc_20240101_120000/best_ctc_model.pt

# Search in different directory
python src/common/analyze_model.py --dir custom_checkpoints
```

**Parameters:**
- Positional argument: Path to checkpoint file (optional, searches for latest)
- `--dir`: Base directory to search for checkpoints (default: generated/checkpoints)

#### Analyze Training Logs
```bash
# List all training runs
python src/common/analyze_tb.py --list

# Analyze latest run
python src/common/analyze_tb.py

# Analyze specific run
python src/common/analyze_tb.py generated/checkpoints/ctc_20240101_120000/logs/

# Plot training metrics
python src/common/analyze_tb.py --plot

# Save plots to file
python src/common/analyze_tb.py --plot --save-plot

# Search in different directory
python src/common/analyze_tb.py --dir custom_checkpoints
```

**Parameters:**
- Positional argument: Path to event file or log directory (optional, searches for latest)
- `--list`: List all available training runs
- `--dir`: Base directory for checkpoints (default: generated/checkpoints)
- `--plot`: Plot metrics over epochs
- `--save-plot`: Save plot to file instead of displaying

**Features:**
- Epoch-by-epoch metric progression
- Loss curves and error rates
- Learning rate schedule visualization
- Convergence analysis
- Training time statistics

### 5. Testing and Debugging

#### Test Data Pipeline
```bash
# Test data loading and augmentation
python src/ctc/tests/test_dataloader.py
```

#### Test Model Architecture
```bash
# Test CTC model architecture
python src/ctc/model.py
```

#### Test Inference Components
```bash
# Test all inference components
python src/ctc/tests/test_inference.py

# Skip audio processing tests
python src/ctc/tests/test_inference.py --skip-audio

# Test specific checkpoint
python src/ctc/tests/test_inference.py --checkpoint path/to/checkpoint.pt
```

## Project Structure

```
digethic_project_asr/
├── src/                      # Source code
│   ├── ctc/                 # CTC model implementation
│   │   ├── model.py         # Model architecture
│   │   ├── train.py         # Training script
│   │   ├── inference.py     # Inference script (for single file transcription)
│   │   ├── evaluate.py      # Evaluation script
│   │   ├── config.py        # Configuration
│   │   ├── core/            # Core components
│   │   │   ├── dataset.py   # Data loading
│   │   │   ├── utils.py     # Text processing
│   │   │   └── early_stopping.py
│   │   ├── tools/           # Utility tools
│   │   │   ├── create_dataset.py
│   │   │   └── optimize_hyperparams.py
│   │   └── tests/           # Test scripts
│   │       ├── test_dataloader.py
│   │       └── test_inference.py
│   │
│   ├── attention/           # Attention model
│   │   ├── model/           # Model components
│   │   │   ├── encoder.py   # BiGRU encoder
│   │   │   ├── decoder.py   # LSTM decoder
│   │   │   ├── attention.py # Location-aware attention mechanism
│   │   │   ├── attention_model.py # End-to-end model
│   │   │   └── initialize.py # Weight initialization
│   │   ├── core/            # Core utilities
│   │   │   ├── dataset.py   # Data loading for attention model
│   │   │   └── levenshtein.py # Error calculation
│   │   ├── tools/           # Data preparation
│   │   │   ├── prepare_data.py # Main data preparation
│   │   │   ├── extract_features.py # Feature extraction (Fbank)
│   │   │   └── convert_labels.py # Label conversion
│   │   ├── train.py         # Training script
│   │   ├── evaluate.py      # Evaluation script
│   │   └── config.py        # Configuration
│   │
│   └── common/              # Shared utilities
│       ├── analyze_model.py # Model checkpoint analysis
│       ├── analyze_tb.py    # TensorBoard log analysis
│       ├── download_librispeech.py # Dataset download script
│       └── plot_waveform.py # Audio visualization
│
├── data/                    # LibriSpeech data (gitignored)
│   ├── dev-clean/
│   ├── test-clean/
│   └── train-clean-100/
│
├── generated/               # Generated outputs (gitignored)
│   ├── subset_10h/          # CTC training subset
│   ├── dev_subset/          # CTC validation subset
│   ├── test_dataset/        # CTC test dataset
│   ├── attention/           # Attention model data
│   │   ├── features/        # Extracted features (.ark files)
│   │   ├── labels/          # Label files
│   │   └── tokens/          # Token list
│   └── checkpoints/         # Model checkpoints
│       ├── ctc_*/           # CTC model checkpoints
│       └── attention_*/     # Attention model checkpoints
│
├── venv/                    # Virtual environment (gitignored)
├── prepare_dataset.sh       # All-in-one data preparation script
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```


## Troubleshooting

### Common Issues

1. **CUDA/MPS Issues with CTC**
   ```bash
   # Force CPU usage
   python src/ctc/train.py --device cpu
   ```

2. **Out of Memory**
   ```bash
   # Reduce batch size
   python src/ctc/train.py --batch_size 2
   ```

3. **Slow Data Loading**
   ```bash
   # Increase workers (careful on Mac)
   python src/ctc/train.py --num_workers 2
   ```

4. **NaN Loss**
   - Reduce learning rate
   - Enable gradient clipping
   - Check for data corruption


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LibriSpeech Dataset](https://www.openslr.org/12/) by Vassil Panayotov et al.
- PyTorch and TorchAudio teams for excellent frameworks
- [python_asr](https://github.com/ry-takashima/python_asr) - Reference implementation for attention-based model
- Research papers that inspired this implementation:
  - [CTC: Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
  - [Location-Aware Attention](https://arxiv.org/abs/1506.07503)

