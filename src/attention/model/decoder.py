# Implementation of Attention RNN decoder.
# References
#   - D. Bahdanau, et al.,
#     ``End-to-end attention-based large vocabulary speech
#       recognition,''
#     in Proc. ICASSP, 2016.
#   - J. Chorowski, et al.,
#     ``Attention-based models for speech recognition,''
#     in Proc. NIPS , 2015.


import torch
import torch.nn as nn
from .attention import LocationAwareAttention
import numpy as np
import matplotlib.pyplot as plt


class Decoder(nn.Module):
    """Decoder
    dim_in:          Input dimension (encoder output dimension)
    dim_hidden:      Decoder RNN hidden dimension
    dim_out:         Output dimension (total tokens including sos and eos)
    dim_att:         Attention mechanism dimension
    att_filter_size: Filter size for LocationAwareAttention
    att_filter_num:  Number of filters for LocationAwareAttention
    sos_id:          <sos> token ID
    att_temperature: Attention temperature parameter
    num_layers:      Number of decoder RNN layers
    dropout_rate:    Dropout rate applied to the output of each layer
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        dim_att,
        att_filter_size,
        att_filter_num,
        sos_id,
        att_temperature=1.0,
        num_layers=1,
        ## dropout
        dropout_rate=0.2,
    ):
        super(Decoder, self).__init__()

        # Set <sos> token ID
        self.sos_id = sos_id

        # Input and output dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out

        # Embedding layer for previous step output
        # (transforms from dim_out dimensional vector
        #  to dim_hidden dimensional vector)
        self.embedding = nn.Embedding(dim_out, dim_hidden)

        # Location aware attention
        self.attention = LocationAwareAttention(
            dim_in,
            dim_hidden,
            dim_att,
            att_filter_size,
            att_filter_num,
            att_temperature,
        )

        # RNN layer
        # RNN receives previous output (after Embedding) and
        # encoder output (after Attention).
        # Therefore RNN input dimension is
        # dim_hidden (Embedding output dimension) \
        #   + dim_in (encoder output dimension)
        self.rnn = nn.LSTM(
            input_size=dim_hidden + dim_in,
            hidden_size=dim_hidden,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True,
        )

        # dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Output layer
        self.out = nn.Linear(in_features=dim_hidden, out_features=dim_out)

        # Attention weight matrix (for visualization)
        self.att_matrix = None

    def forward(self, enc_sequence, enc_lengths, label_sequence=None):
        """Network forward computation
        enc_sequence:   Encoder output sequence for each utterance
                        [B x Tenc x Denc]
        enc_lengths:    Sequence length of encoder RNN output for each utterance [B]
        label_sequence: Ground truth label sequence for each utterance (used during training)
                        [B x Tout]
          [] indicates tensor dimensions
          B:    Number of utterances in minibatch (batch size)
          Tenc: Encoder RNN output sequence length (including zero padding)
          Denc: Encoder RNN output dimension (dim_in)
          Tout: Ground truth label sequence length (including zero padding)
        label_sequence is provided only during training
        """
        # Get input information (batch size, device (cpu or cuda))
        batch_size = enc_sequence.size()[0]
        device = enc_sequence.device

        #
        # Determine maximum decoder steps
        #
        if label_sequence is not None:
            # Training:
            #   = When label information is provided,
            #     use label sequence length
            max_step = label_sequence.size()[1]
        else:
            # Evaluation:
            #   = When label information is not provided,
            #     use encoder output sequence length
            max_step = enc_sequence.size()[1]

        #
        # Initialize internal parameters
        #
        # Previous step token. Initialize with <sos>
        prev_token = torch.ones(batch_size, 1, dtype=torch.long) * self.sos_id
        # Place on device (CPU/GPU)
        prev_token = prev_token.to(device=device, dtype=torch.long)
        # Initialize previous RNN output and Attention weights with None
        prev_rnnout = None
        prev_att = None
        # Initialize previous RNN internal parameters (h, c) with None
        prev_h_c = None
        # Reset Attention internal parameters
        self.attention.reset()

        # Prepare output tensor [batch_size x max_step x dim_out]
        output = torch.zeros(batch_size, max_step, self.dim_out)
        # Place on device (CPU/GPU)
        output = output.to(device=device, dtype=enc_sequence.dtype)

        # Initialize Attention weight matrix for visualization
        self.att_matrix = torch.zeros(batch_size, max_step, enc_sequence.size(1))

        #
        # Run decoder for maximum number of steps
        #
        for i in range(max_step):
            #
            # 1. Calculate Attention and get context vector
            #    (weighted sum of encoder outputs) and
            #    Attention weights
            #
            context, att_weight = self.attention(
                enc_sequence, enc_lengths, prev_rnnout, prev_att
            )
            #
            # 2. Run RNN for one step
            #
            # Pass previous token through Embedding layer
            prev_token_emb = self.embedding(prev_token)
            # Concatenate prev_token_emb and context vector,
            # then input to RNN. RNN input tensor size is
            # (batch_size, sequence_length(=1), dim_in), so
            # use view on context to match dimensions before concatenation
            context = context.view(batch_size, 1, self.dim_in)
            rnn_input = torch.cat((prev_token_emb, context), dim=2)
            # Pass through RNN
            rnnout, h_c = self.rnn(rnn_input, prev_h_c)

            ## dropout
            # Apply dropout to RNN output
            rnnout_with_dropout = self.dropout(rnnout)

            #
            # 3. Pass RNN output through linear layer
            #

            ## dropout
            # out = self.out(rnnout)
            out = self.out(rnnout_with_dropout)

            # Store out in output tensor
            output[:, i, :] = out.view(batch_size, self.dim_out)

            #
            # 4. Update previous RNN output, RNN internal parameters,
            #    Attention weights, and token.
            #
            prev_rnnout = rnnout
            prev_h_c = h_c
            prev_att = att_weight
            # Update token
            if label_sequence is not None:
                # Training:
                #  = Use ground truth label when provided
                prev_token = label_sequence[:, i].view(batch_size, 1)
            else:
                # Evaluation:
                #  = Use predicted value when ground truth
                #    is not provided
                _, prev_token = torch.max(out, 2)

            # Attention weight matrix for visualization
            self.att_matrix[:, i, :] = att_weight

        return output

    def save_att_matrix(self, utt, filename):
        """Save Attention Matrix
        utt:      Utterance index in batch to output
        filename: Output filename
        """
        att_mat = self.att_matrix[utt].detach().numpy()
        plt.figure(figsize=(10, 8))

        # Normalize attention weights (to 0-1 range)
        att_mat = att_mat - np.min(att_mat)
        att_mat = att_mat / (np.max(att_mat) + 1e-8)  # Avoid division by zero

        # Plot
        im = plt.imshow(
            att_mat,
            cmap="hot",
            interpolation="nearest",
            aspect="auto",
        )

        plt.colorbar(im, label="Attention Weight")
        plt.xlabel("Encoder Steps")
        plt.ylabel("Decoder Steps")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
