import torch
import torch.nn as nn
import torch.nn.functional as F
from transfomers.layers.positional import *
from transfomers.layers.feedforward import *
from transfomers.layers.multiheadattention import *

class EncoderLayer(nn.Module):
    def __init__(self, head_size, hidden_size, d_ff, dropout_rate=0.1, eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.mtha = MultiheadAttention(hidden_size, head_size)
        self.feed_forward = ffn(d_ff, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mtha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.feed_forward(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


