# Definition of Dataset for PyTorch

from torch.utils.data import Dataset
import numpy as np
import sys
import struct


def read_kaldi_matrix(file_path):
    """
    Read feature matrix from Kaldi-format binary file

    Args:
        file_path: Path to ark file

    Returns:
        matrix: Feature matrix (time, dim)
    """
    with open(file_path, "rb") as f:
        # Read header
        header = f.read(3)
        if header != b"\x00\x00B":
            raise ValueError(f"Invalid Kaldi binary header: {header}")

        # Read number of rows
        size_marker = f.read(1)
        if size_marker != b"\x04":
            raise ValueError(f"Invalid size marker for rows: {size_marker}")
        rows = struct.unpack("<i", f.read(4))[0]

        # Read number of columns
        size_marker = f.read(1)
        if size_marker != b"\x04":
            raise ValueError(f"Invalid size marker for cols: {size_marker}")
        cols = struct.unpack("<i", f.read(4))[0]

        # Read data
        data = np.frombuffer(f.read(rows * cols * 4), dtype=np.float32)
        matrix = data.reshape(rows, cols)

    return matrix


class SequenceDataset(Dataset):
    """Class for creating mini-batch data
        Inherits from torch.utils.data.Dataset and defines
        the following functions:
        __len__: Function to output total number of samples
        __getitem__: Function to output data for one sample
    feat_scp:  Feature list file
    label_scp: Label file
    feat_mean: Feature mean vector
    feat_std:  Feature standard deviation vector per dimension
    pad_index: Integer value for padding to match frame
               numbers during batching
    splice:    Concatenate features from surrounding frames
               splice=1 concatenates 1 frame before and after,
               resulting in 3x the dimension.
               splice=0 does nothing
    """

    def __init__(self, feat_scp, label_scp, feat_mean, feat_std, pad_index=0, splice=0):
        # Number of utterances
        self.num_utts = 0
        # ID list for each utterance
        self.id_list = []
        # List of feature file paths for each utterance
        self.feat_list = []
        # List of feature frame counts for each utterance
        self.feat_len_list = []
        # Feature mean vector
        self.feat_mean = feat_mean
        # Feature standard deviation vector
        self.feat_std = feat_std
        # Floor standard deviation
        # (to prevent division by zero)
        self.feat_std[self.feat_std < 1e-10] = 1e-10
        # Feature dimension
        self.feat_dim = np.size(self.feat_mean)
        # Labels for each utterance
        self.label_list = []
        # List of label lengths for each utterance
        self.label_len_list = []
        # Maximum number of frames
        self.max_feat_len = 0
        # Maximum label length
        self.max_label_len = 0
        # Integer value used for frame padding
        self.pad_index = pad_index
        # splice: Concatenate features from n frames before and after
        self.splice = splice

        # Read feature list and labels line by line
        # to obtain information
        with open(feat_scp, mode="r") as file_f, open(label_scp, mode="r") as file_l:
            for line_feats, line_label in zip(file_f, file_l):
                # Split each line by spaces
                # and convert to list
                parts_feats = line_feats.split()
                parts_label = line_label.split()

                # Error if utterance ID (0th element of parts)
                # doesn't match between features and labels
                if parts_feats[0] != parts_label[0]:
                    sys.stderr.write("IDs of feat and label do not match.\n")
                    exit(1)

                # Add utterance ID to list
                self.id_list.append(parts_feats[0])
                # Add feature file path to list
                self.feat_list.append(parts_feats[1])
                # Add frame count to list
                # LibriSpeech format: Frame count not provided,
                # so read the feature file to check
                features = read_kaldi_matrix(parts_feats[1])
                feat_len = features.shape[0]  # Number of rows is frame count
                self.feat_len_list.append(feat_len)

                # Convert labels (written as numbers)
                # to int numpy array
                label = np.int64(parts_label[1:])
                # Add to label list
                self.label_list.append(label)
                # Add label length
                self.label_len_list.append(len(label))

                # Count utterances
                self.num_utts += 1

        # Get maximum frame count
        self.max_feat_len = np.max(self.feat_len_list)
        # Get maximum label length
        self.max_label_len = np.max(self.label_len_list)

        # Pad label data to match maximum frame length
        # using pad_index value
        for n in range(self.num_utts):
            # Number of frames to pad
            # = maximum frames - own frames
            pad_len = self.max_label_len - self.label_len_list[n]
            # Pad with pad_index value
            self.label_list[n] = np.pad(
                self.label_list[n],
                [0, pad_len],
                mode="constant",
                constant_values=self.pad_index,
            )

    def __len__(self):
        """Function to return total number of training samples
        Since this implementation creates batches per utterance,
        total samples = number of utterances.
        """
        return self.num_utts

    def __getitem__(self, idx):
        """Function to return sample data
        Since this implementation creates batches per utterance,
        idx = utterance number.
        """
        # Number of frames in feature sequence
        feat_len = self.feat_len_list[idx]
        # Label length
        label_len = self.label_len_list[idx]

        # Read feature data from feature file
        # LibriSpeech format (Kaldi binary) only
        feat = read_kaldi_matrix(self.feat_list[idx])

        # Normalize using mean and standard deviation
        feat = (feat - self.feat_mean) / self.feat_std

        # splicing: Concatenate features from n frames before and after
        org_feat = feat.copy()
        for n in range(-self.splice, self.splice + 1):
            # Shift original features by n frames
            tmp = np.roll(org_feat, n, axis=0)
            if n < 0:
                # If shifted forward,
                # set last n frames to 0
                tmp[n:] = 0
            elif n > 0:
                # If shifted backward,
                # set first n frames to 0
                tmp[:n] = 0
            else:
                continue
            # Concatenate shifted features
            # in dimension direction
            feat = np.hstack([feat, tmp])

        # Pad feature data with zeros to match
        # maximum frame count
        pad_len = self.max_feat_len - feat_len
        feat = np.pad(feat, [(0, pad_len), (0, 0)], mode="constant", constant_values=0)

        # Label
        label = self.label_list[idx]

        # Utterance ID
        utt_id = self.id_list[idx]

        # Return features, label, frame count,
        # label length, and utterance ID
        return (feat, label, feat_len, label_len, utt_id)
