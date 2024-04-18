import torch
import torch.nn as nn
import torch.nn.functional as F
from transfomers.layers.Encoder import Encoder
from transfomers.layers.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self,N, inp_vocab_size, trg_vocab_size, head_size, hidden_size, d_ff, dropout_rate, eps = 1e-6, ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N = N, head_size= head_size, vocab_size= inp_vocab_size, hidden_size= hidden_size, d_ff= d_ff , dropout_rate= dropout_rate, eps = eps)
        self.decoder = Decoder(N = N, head_size= head_size, vocab_size= trg_vocab_size, hidden_size= hidden_size, d_ff= d_ff, dropout_rate=dropout_rate, eps = eps)
        self.linear = nn.Linear(hidden_size, trg_vocab_size)

    def forward(self, encoder_in, decoder_in , encoder_padding_mask = None, decoder_look_ahead_mask = None, decoder_padding_mask =None):
        encoder_out = self.encoder(encoder_in, encoder_padding_mask)
        decoder_out, attention_weights = self.decoder(decoder_in, encoder_out,decoder_look_ahead_mask, decoder_padding_mask)
        return self.linear(decoder_out)