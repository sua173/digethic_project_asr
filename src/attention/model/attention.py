# Implementation of Attention (Location aware attention).
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
import torch.nn.functional as F


class LocationAwareAttention(nn.Module):
    """Location aware attention
    dim_encoder:   Encoder RNN output dimension
    dim_decoder:   Decoder RNN output dimension
    dim_attention: Attention mechanism dimension
    filter_size:   Location filter size (filter convolved
                   with previous attention weights)
    filter_num:    Number of location filters
    temperature:   Temperature parameter for attention weight calculation
    """

    def __init__(
        self,
        dim_encoder,
        dim_decoder,
        dim_attention,
        filter_size,
        filter_num,
        temperature=1.0,
    ):
        super(LocationAwareAttention, self).__init__()

        # F: Convolutional layer applied to previous attention weights
        self.loc_conv = nn.Conv1d(
            in_channels=1,
            out_channels=filter_num,
            kernel_size=2 * filter_size + 1,
            stride=1,
            padding=filter_size,
            bias=False,
        )
        # Among the following three layers, only one has bias=True, others have bias=False
        # W: Projection layer for previous decoder RNN output
        self.dec_proj = nn.Linear(
            in_features=dim_decoder, out_features=dim_attention, bias=False
        )
        # V: Projection layer for encoder RNN output
        self.enc_proj = nn.Linear(
            in_features=dim_encoder, out_features=dim_attention, bias=False
        )
        # U: Projection layer for convolved attention weights
        self.att_proj = nn.Linear(
            in_features=filter_num, out_features=dim_attention, bias=True
        )
        # w: Linear layer for Ws + Vh + Uf + b
        self.out = nn.Linear(in_features=dim_attention, out_features=1)

        # Dimensions
        self.dim_encoder = dim_encoder
        self.dim_decoder = dim_decoder
        self.dim_attention = dim_attention

        # Temperature parameter
        self.temperature = temperature

        # Encoder RNN output (h) and its projection (Vh)
        # Since these have the same value at every decode step,
        # compute only once and keep the results
        self.input_enc = None
        self.projected_enc = None
        # Sequence length of encoder RNN output for each utterance
        self.enc_lengths = None
        # Maximum sequence length of encoder RNN output
        # (= zero-padded encoder RNN output sequence length)
        self.max_enc_length = None
        # Attention mask
        # Mask to zero out weights after encoder sequence length
        # (zero-padded portion)
        self.mask = None

    def reset(self):
        """Reset internal parameters
        This function must be called at the beginning
        of processing each batch
        """
        self.input_enc = None
        self.projected_enc = None
        self.enc_lengths = None
        self.max_enc_length = None
        self.mask = None

    def forward(self, input_enc, enc_lengths, input_dec=None, prev_att=None):
        """Network forward computation
        input_enc:   Encoder RNN output [B x Tenc x Denc]
        enc_lengths: Sequence length of encoder RNN output for each utterance in batch [B]
        input_dec:   Decoder RNN output from previous step [B x Ddec]
        prev_att:    Attention weights from previous step [B x Tenc]
          [] indicates tensor dimensions
          B:    Number of utterances in minibatch (batch size)
          Tenc: Encoder RNN output sequence length (including zero padding)
          Denc: Encoder RNN output dimension (dim_encoder)
          Ddec: Decoder RNN output dimension (dim_decoder)
        """
        # Get batch size (number of utterances)
        batch_size = input_enc.size()[0]

        #
        # Compute encoder RNN output and its projection only once
        #
        if self.input_enc is None:
            # Encoder RNN output (h)
            self.input_enc = input_enc
            # Sequence length for each utterance
            self.enc_lengths = enc_lengths
            # Maximum sequence length
            self.max_enc_length = input_enc.size()[1]
            # Perform projection (compute Vh)
            self.projected_enc = self.enc_proj(self.input_enc)

        #
        # Project decoder RNN output from previous step (compute Ws)
        #
        # Use zero matrix as initial value if no previous decoder RNN output
        if input_dec is None:
            input_dec = torch.zeros(batch_size, self.dim_decoder)
            # Place created tensor on the same device (GPU/CPU)
            # as encoder RNN output
            input_dec = input_dec.to(
                device=self.input_enc.device, dtype=self.input_enc.dtype
            )
        # Project previous decoder RNN output
        projected_dec = self.dec_proj(input_dec)

        #
        # Project attention weight information from previous step
        # (compute Uf+b)
        #
        # Create attention mask
        if self.mask is None:
            self.mask = torch.zeros(batch_size, self.max_enc_length, dtype=torch.bool)
            # For each utterance in batch, set elements beyond
            # sequence length (i.e., zero-padded portion) to
            # 1 (= masking target)
            for i, length in enumerate(self.enc_lengths):
                length = length.item()
                self.mask[i, length:] = 1
            # Place created tensor on the same device (GPU/CPU)
            # as encoder RNN output
            self.mask = self.mask.to(device=self.input_enc.device)

        # If there are no previous attention weights,
        # provide uniform weights as initial values
        if prev_att is None:
            # Create tensor with all elements set to 1
            prev_att = torch.ones(batch_size, self.max_enc_length)
            # Divide by sequence length for each utterance
            # Since prev_att is a 2D tensor and
            # enc_lengths is a 1D tensor,
            # use view(batch_size, 1) to reshape enc_lengths
            # to 2D tensor before division
            prev_att = prev_att / self.enc_lengths.view(batch_size, 1)
            # Place created tensor on the same device (GPU/CPU)
            # as encoder RNN output
            prev_att = prev_att.to(
                device=self.input_enc.device, dtype=self.input_enc.dtype
            )
            # Apply masking to zero out weights beyond utterance length
            prev_att.masked_fill_(self.mask, 0)

        # Compute convolution of attention weights {f} = F*a
        # Conv1D expects input size of
        # (batch_size, in_channels, self.max_enc_length)
        # (in_channels is the number of input channels,
        # which is 1 in this program)
        # Use view to match the size
        convolved_att = self.loc_conv(prev_att.view(batch_size, 1, self.max_enc_length))

        # Size of convolved_att is
        # (batch_size, filter_num, self.max_enc_length)
        # Linear layer expects input size of
        # (batch_size, self.max_enc_length, filter_num), so
        # transpose dimensions 1 and 2 before passing to att_proj
        projected_att = self.att_proj(convolved_att.transpose(1, 2))

        #
        # Calculate attention weights
        #
        # Tensor sizes at this point:
        # self.projected_enc: (batch_size, self.max_enc_length,
        #                      self.dim_attention)
        # projected_dec: (batch_size, self.dim_attention)
        # projected_att: (batch_size, self.max_enc_length, self.dim_attention)
        # Use view to match dimensions of projected_dec tensor
        projected_dec = projected_dec.view(batch_size, 1, self.dim_attention)

        # To calculate score, add projection tensors,
        # apply tanh, then apply projection
        # w tanh(Ws + Vh + Uf + b)
        score = self.out(torch.tanh(projected_dec + self.projected_enc + projected_att))

        # Current score tensor size is
        # (batch_size, self.max_enc_length, 1)
        # Use view to restore original attention size
        score = score.view(batch_size, self.max_enc_length)

        # Apply masking
        # (zero out weights in zero-padded portion
        # of encoder RNN output)
        # However, since exp(score) computed in softmax
        # must become zero, fill with -inf (log of 0)
        # instead of 0 at the score stage
        score.masked_fill_(self.mask, -float("inf"))

        # Calculate attention weights using temperature-scaled softmax
        att_weight = F.softmax(self.temperature * score, dim=1)

        # Use att_weight to calculate weighted sum of encoder RNN output
        # to obtain context vector
        # (using view to match tensor sizes of input_enc and attention_weight)
        context = torch.sum(
            self.input_enc * att_weight.view(batch_size, self.max_enc_length, 1), dim=1
        )

        # Output context vector and attention weights
        return context, att_weight
