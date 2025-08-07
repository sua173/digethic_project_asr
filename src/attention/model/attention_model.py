# Define model structure


import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder
from .initialize import lecun_initialization
import numpy as np


class AttentionModel(nn.Module):
    """Definition of End-to-End model using Attention RNN
    dim_in:            Input dimension
    dim_enc_hid:       Encoder hidden layer dimension
    dim_enc_proj:      Encoder Projection layer dimension
                       (this becomes encoder output dimension)
    dim_dec_hid:       Decoder RNN dimension
    dim_out:           Output dimension (total tokens including sos and eos)
    dim_att:           Attention mechanism dimension
    att_filter_size:   LocationAwareAttention filter size
    att_filter_num:    LocationAwareAttention filter count
    sos_id:            <sos> token ID
    enc_bidirectional: If True, use bidirectional RNN for encoder
    enc_sub_sample:    Frame subsampling rate for each layer in encoder
    enc_rnn_type:      Encoder RNN type. Select 'LSTM' or 'GRU'
    att_temperature:   Attention temperature parameter
    enc_num_layers:    Number of encoder RNN layers
    dec_num_layers:    Number of decoder RNN layers
    enc_dropout_rate:  Dropout rate applied to the output of each encoder layer
    dec_dropout_rate:  Dropout rate applied to the output of each decoder layer
    """

    def __init__(
        self,
        dim_in,
        dim_enc_hid,
        dim_enc_proj,
        dim_dec_hid,
        dim_out,
        dim_att,
        att_filter_size,
        att_filter_num,
        sos_id,
        att_temperature=1.0,
        enc_num_layers=2,
        dec_num_layers=2,
        enc_bidirectional=True,
        enc_sub_sample=None,
        enc_rnn_type="LSTM",
        enc_dropout_rate=0.2,
        dec_dropout_rate=0.2,
    ):
        super(AttentionModel, self).__init__()

        # Create encoder
        self.encoder = Encoder(
            dim_in=dim_in,
            dim_hidden=dim_enc_hid,
            dim_proj=dim_enc_proj,
            num_layers=enc_num_layers,
            bidirectional=enc_bidirectional,
            sub_sample=enc_sub_sample,
            rnn_type=enc_rnn_type,
            dropout_rate=enc_dropout_rate,
        )

        # Create decoder
        self.decoder = Decoder(
            dim_in=dim_enc_proj,
            dim_hidden=dim_dec_hid,
            dim_out=dim_out,
            dim_att=dim_att,
            att_filter_size=att_filter_size,
            att_filter_num=att_filter_num,
            sos_id=sos_id,
            att_temperature=att_temperature,
            num_layers=dec_num_layers,
            dropout_rate=dec_dropout_rate,
        )

        # Execute LeCun parameter initialization
        lecun_initialization(self)

    def forward(self, input_sequence, input_lengths, label_sequence=None):
        """Network computation (forward processing) function
        input_sequence: Input sequence for each utterance [B x Tin x D]
        input_lengths:  Sequence length (frame count) for each utterance [B]
        label_sequence: Ground truth label sequence for each utterance (used during training) [B x Tout]
          [] indicates tensor dimensions
          B:    Number of utterances in minibatch (minibatch size)
          Tin:  Input tensor sequence length (including zero padding)
          D:    Input dimension (dim_in)
          Tout: Ground truth label sequence length (including zero padding)
        """
        # Input to encoder
        enc_out, enc_lengths = self.encoder(input_sequence, input_lengths)

        # Input to decoder
        dec_out = self.decoder(enc_out, enc_lengths, label_sequence)

        # Output decoder output and encoder output sequence length
        return dec_out, enc_lengths

    def save_att_matrix(self, utt, filename):
        """Save Attention matrix as image
        utt:      Utterance index within batch to output
        filename: Output filename
        """
        # Execute decoder's save_att_matrix
        self.decoder.save_att_matrix(utt, filename)
