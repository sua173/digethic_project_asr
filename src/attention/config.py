# Configuration for attention-based ASR model

from datetime import datetime
import os


def get_default_config():
    """Get default training configuration as dictionary"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = {
        # Data directories
        "feat_dir_train": "generated/attention/features/train_clean_100_10hr",
        "feat_dir_dev": "generated/attention/features/dev_clean_32min",
        "feat_dir_test": "generated/attention/features/test_clean",
        "label_dir": "generated/attention/labels",
        "token_dir": "generated/attention/tokens",
        # Output directory
        "timestamp": timestamp,
        "output_dir": os.path.join(
            "generated", "checkpoints", f"attention_{timestamp}"
        ),
        # Training parameters
        "batch_size": 16,
        "max_num_epoch": 60,
        "initial_learning_rate": 0.5,
        "clip_grad_threshold": 2.0,
        "lr_decay_start_epoch": 7,
        "lr_decay_factor": 0.5,
        "early_stop_threshold": 3,
        # Encoder settings
        "enc_num_layers": 3,
        "enc_sub_sample": [1, 2, 2, 1, 1],
        "enc_rnn_type": "GRU",
        # "enc_rnn_type": "LSTM",
        "enc_hidden_dim": 320,
        "enc_projection_dim": 320,
        "enc_bidirectional": True,
        "enc_dropout_rate": 0.2,  # Dropout rate for encoder
        # Decoder settings
        "dec_num_layers": 1,
        "dec_hidden_dim": 300,
        "dec_dropout_rate": 0.2,  # Dropout rate for decoder
        # Attention settings
        "att_hidden_dim": 320,
        "att_filter_size": 100,
        "att_filter_num": 10,
        "att_temperature": 1.0,
        # Evaluation settings
        "evaluate_error": {"train": False, "validation": True},
        # Device settings
        "device": "cuda",
    }

    return config


def update_config(config, args):
    """Update configuration with command line arguments"""
    if hasattr(args, "epochs") and args.epochs is not None:
        config["max_num_epoch"] = args.epochs

    if hasattr(args, "batch_size") and args.batch_size is not None:
        config["batch_size"] = args.batch_size

    if hasattr(args, "train_set") and args.train_set is not None:
        config["feat_dir_train"] = f"generated/attention/features/{args.train_set}"

    if hasattr(args, "dev_set") and args.dev_set is not None:
        config["feat_dir_dev"] = f"generated/attention/features/{args.dev_set}"

    if hasattr(args, "test_set") and args.test_set is not None:
        config["feat_dir_test"] = f"generated/attention/features/{args.test_set}"

    if hasattr(args, "learning_rate") and args.learning_rate is not None:
        config["initial_learning_rate"] = args.learning_rate

    if hasattr(args, "output_dir") and args.output_dir is not None:
        config["output_dir"] = args.output_dir
    else:
        # Update output_dir if train_set changed
        config["output_dir"] = os.path.join(
            "generated", "checkpoints", f"attention_{config['timestamp']}"
        )

    return config


def get_model_config(config):
    """Extract model-specific configuration"""
    return {
        "enc_rnn_type": config["enc_rnn_type"],
        "enc_num_layers": config["enc_num_layers"],
        "enc_sub_sample": config["enc_sub_sample"],
        "enc_hidden_dim": config["enc_hidden_dim"],
        "enc_projection_dim": config["enc_projection_dim"],
        "enc_bidirectional": config["enc_bidirectional"],
        "enc_dropout_rate": config["enc_dropout_rate"],
        "dec_num_layers": config["dec_num_layers"],
        "dec_hidden_dim": config["dec_hidden_dim"],
        "dec_dropout_rate": config["dec_dropout_rate"],
        "att_hidden_dim": config["att_hidden_dim"],
        "att_filter_size": config["att_filter_size"],
        "att_filter_num": config["att_filter_num"],
        "att_temperature": config["att_temperature"],
    }


def get_training_config(config):
    """Extract training-specific configuration"""
    return {
        "batch_size": config["batch_size"],
        "max_num_epoch": config["max_num_epoch"],
        "clip_grad_threshold": config["clip_grad_threshold"],
        "initial_learning_rate": config["initial_learning_rate"],
        "lr_decay_start_epoch": config["lr_decay_start_epoch"],
        "lr_decay_factor": config["lr_decay_factor"],
        "early_stop_threshold": config["early_stop_threshold"],
        "evaluate_error": config["evaluate_error"],
    }


def save_config_summary(config):
    """Get configuration summary for saving"""
    return {
        # Data info
        "train_set": os.path.basename(config["feat_dir_train"]),
        "dev_set": os.path.basename(config["feat_dir_dev"]),
        "test_set": os.path.basename(config.get("feat_dir_test", "test_clean")),
        # Model config
        **get_model_config(config),
        # Training config
        **get_training_config(config),
        # Timestamp
        "timestamp": config["timestamp"],
    }


def setup_device(force_cpu: bool = False):
    """Setup optimal device"""
    import torch

    if force_cpu:
        # Force CPU usage (e.g., for CTC loss)
        device = torch.device("cpu")
        print("Using CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        # MPS is available for Attention models
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
