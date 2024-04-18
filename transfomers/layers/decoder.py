import torch
import torch.nn as nn
from transfomers.layers.decoderLayer import DecoderLayer
from transfomers.layers.positional import generate_positional_encoding
class Decoder(nn.Module):
    def __init__(self, N, vocab_size, hidden_size , head_size, d_ff, dropout_rate = 0.1 ,eps=1e-6):
        super(Decoder, self).__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoderLayers = [DecoderLayer(head_size, hidden_size, d_ff, dropout_rate, eps) for _ in range(N)]
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_seq, encoder_out, look_ahead_mask=None, padding_mask=None):
        input_length = input_seq.size()[1]
        embedded_input = self.word_embedding(input_seq)
        embedded_input *= torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))

        positional_embedding = generate_positional_encoding(input_length, self.hidden_size)
        decoder_out = self.dropout(embedded_input + positional_embedding)
        attention_weights = {}
        for i,decoder_layer in enumerate(self.decoderLayers):
            decoder_out, self_attn_weights, global_attn_weights = decoder_layer(decoder_out, encoder_out, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer_{}_self_attn_weights'.format(i)] = self_attn_weights
            attention_weights['decoder_layer_{}_global_attn_weights'.format(i)] = global_attn_weights

        return decoder_out, attention_weights



