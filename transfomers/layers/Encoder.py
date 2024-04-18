import torch
import torch.nn as nn
import torch.nn.functional as F
from transfomers.layers.encoderLayers import *
from transfomers.layers.positional import *
class Encoder(nn.Module):
    def __init__(self, N , head_size, vocab_size,  hidden_size, d_ff, dropout_rate=0.1, eps=1e-6):
        super(Encoder, self).__init__()
        self.N = N
        self.encoder_layers = [EncoderLayer(head_size, hidden_size, d_ff, dropout_rate=0.1, eps=eps) for _ in range(N)]
        self.hidden_size = hidden_size
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, input, mask=None):
        q_length = input.size()[1]


        # embedded_q shape: (batch_size, q_length, d_model)

        # Embedding input and applying positional encoding
        embedded_input = self.word_embedding(input)
        embedded_input *= torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))


        positional_embedding = generate_positional_encoding(q_length, self.hidden_size)
        encoder_out = self.dropout(embedded_input + positional_embedding)

        # Applying encoder layers
        for encoder_layer in self.encoder_layers:
            encoder_out = encoder_layer(encoder_out, mask)  # Applying masking if necessary

        return encoder_out





