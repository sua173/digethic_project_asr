"""
Configuration for CTC-based Speech Recognition Model
"""

# Model architecture
MODEL_CONFIG = {
    "hidden_dim": 256,  # LSTM hidden dimension
    "num_layers": 3,  # Number of LSTM layers
    "bidirectional": True,  # Use bidirectional LSTM
    "dropout": 0.4,  # Dropout rate
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,  # Batch size (increased for gradient stability)
    "epochs": 60,  # Number of epochs
    "lr": 0.000476,  # Learning rate (reduced for MPS stability)
    "gradient_clip": 4.89,  # Gradient clipping value (reduced for MPS stability)
    "warmup_epochs": 2,  # Learning rate warmup epochs
    "early_stopping_patience": 5,  # Early stopping patience
}

# Scheduler configuration
SCHEDULER_CONFIG = {
    "scheduler_type": "reduce_on_plateau",  # Type of scheduler
    "mode": "min",  # Minimize loss
    "factor": 0.7,  # Reduction factor
    "scheduler_patience": 3,  # Scheduler patience
}

# Data configuration
DATA_CONFIG = {
    "train_manifest": "generated/subset_10h/train_manifest.jsonl",
    "val_manifest": "generated/dev_subset/val_manifest.jsonl",
    "num_workers": 0,  # DataLoader workers
    "apply_spec_augment": True,  # Apply SpecAugment
}

# Inference configuration
INFERENCE_CONFIG = {
    "beam_size": 1,  # CTC uses greedy decoding
    "blank_threshold": 0.95,  # Threshold for blank token
}

# System configuration
SYSTEM_CONFIG = {
    "save_dir": "generated/checkpoints",  # Directory to save models
    "log_interval": 5,  # Steps between logging
    "save_interval": 1000,  # Epochs between saving
    "device": "cuda",  # Device (CTC loss not supported on MPS)
}


def get_full_config():
    """Get complete configuration dictionary"""
    config = {}
    config.update(MODEL_CONFIG)
    config.update(TRAINING_CONFIG)
    config.update(SCHEDULER_CONFIG)
    config.update(DATA_CONFIG)
    config.update(INFERENCE_CONFIG)
    config.update(SYSTEM_CONFIG)
    return config


def update_config(config_dict, updates):
    """Update configuration with new values"""
    config = config_dict.copy()
    config.update(updates)
    return config
