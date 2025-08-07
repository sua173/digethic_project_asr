# Implementation of RNN encoder.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class Encoder(nn.Module):
    """Encoder
    dim_in:        Input feature dimension
    dim_hidden:    Hidden layer dimension (when bidirectional=True,
                   actual dimension is dim_hidden * 2)
    dim_proj:      Projection layer dimension
                   (this becomes encoder output dimension)
    num_layers:    Number of RNN layers (and Projection layers)
    bidirectional: If True, use bidirectional RNN
    sub_sample:    Frame subsampling rate for each layer
                   When num_layers=4, sub_sample=[1,2,3,1] means
                   subsample by 1/2 at 2nd layer, by 1/3 at 3rd layer
                   (output frame count becomes 1/6)
    rnn_type:      Select 'LSTM' or 'GRU'
    dropout_rate:  Dropout rate applied to the output of each layer
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_proj,
        num_layers=2,
        bidirectional=True,
        sub_sample=None,
        rnn_type="LSTM",
        ## dropout
        dropout_rate=0.2,
    ):
        super(Encoder, self).__init__()
        # Number of RNN layers
        self.num_layers = num_layers

        # Define RNN layers one by one and create list
        rnn = []
        for n in range(self.num_layers):
            # Input dimension to RNN is
            # dim_in for first layer only, dim_proj for others
            input_size = dim_in if n == 0 else dim_proj
            # Use GRU if rnn_type is GRU, otherwise use LSTM
            if rnn_type == "GRU":
                rnn.append(
                    nn.GRU(
                        input_size=input_size,
                        hidden_size=dim_hidden,
                        num_layers=1,
                        bidirectional=bidirectional,
                        batch_first=True,
                    )
                )
            else:
                rnn.append(
                    nn.LSTM(
                        input_size=input_size,
                        hidden_size=dim_hidden,
                        num_layers=1,
                        bidirectional=bidirectional,
                        batch_first=True,
                    )
                )
        # Convert to ModuleList since standard list
        # cannot be handled by PyTorch
        self.rnn = nn.ModuleList(rnn)

        # Define sub_sample
        if sub_sample is None:
            # If not defined, do not perform frame subsampling
            # (set sub_sample as list with all elements 1)
            self.sub_sample = [1 for i in range(num_layers)]
        else:
            # If defined, use it
            self.sub_sample = sub_sample

        # Define Projection layers one by one, similar to RNN layers
        proj = []
        for n in range(self.num_layers):
            # Projection layer input dimension = RNN layer output dimension.
            # Dimension doubles when bidirectional=True
            input_size = dim_hidden * (2 if bidirectional else 1)
            proj.append(nn.Linear(in_features=input_size, out_features=dim_proj))
        # Convert to ModuleList, similar to RNN layers
        self.proj = nn.ModuleList(proj)

        ## dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, sequence, lengths):
        """Network computation (forward processing) function
        sequence: Input sequence for each utterance [B x T x D]
        lengths:  Sequence length (frame count) for each utterance [B]
          [] indicates tensor dimensions
          B: Number of utterances in minibatch (minibatch size)
          T: Input tensor sequence length (including zero padding)
          D: Input dimension (dim_in)
        """
        # Initialize output and its length information with input
        output = sequence
        output_lengths = lengths

        # Input alternately to RNN and Projection layers for num_layers times
        for n in range(self.num_layers):
            # Convert input to PackedSequence data
            # for input to RNN
            rnn_input = nn.utils.rnn.pack_padded_sequence(
                output, output_lengths, batch_first=True
            )

            # When using GPU and cuDNN,
            # adding the following line speeds up processing
            # (reset parameter data pointers)
            self.rnn[n].flatten_parameters()

            # Input to RNN layer
            output, (h, c) = self.rnn[n](rnn_input)

            # Convert back from PackedSequence data to tensor
            # for input to Projection layer
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )

            # Execute sub sampling
            # Get subsampling rate for this layer
            sub = self.sub_sample[n]
            if sub > 1:
                # Execute subsampling
                output = output[:, ::sub]
                # Update frame count
                # Updated frame count = (previous frame count + 1) // sub
                output_lengths = torch.div(
                    (output_lengths + 1), sub, rounding_mode="floor"
                )
            # Input to Projection layer
            output = torch.tanh(self.proj[n](output))

            ## dropout
            output = self.dropout(output)

        # Since frame count changes when sub sampling is executed,
        # also output the frame count information
        return output, output_lengths
