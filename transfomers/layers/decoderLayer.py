import torch.nn as nn
from transfomers.layers.multiheadattention import MultiheadAttention
from transfomers.layers.feedforward import ffn


class DecoderLayer(nn.Module):
    def __init__(self, head_size, hidden_size, d_ff, dropout_rate=0.1, eps=1e-6):
        super(DecoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.self_attention = MultiheadAttention(hidden_size, head_size)
        self.enc_dec_attention = MultiheadAttention(hidden_size, head_size)
        self.feed_forward = ffn(d_ff, hidden_size)
        self.layernorm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.layernorm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.layernorm3 = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask = None , padding_mask = None):
        # Sublayer 1: Masked multi-head self-attention
        att_output1, self_attention_weights = self.self_attention(x, x, x, look_ahead_mask)
        att_output1 = self.dropout1(att_output1)
        att_output1 = self.layernorm1(att_output1 + x)

        # Sublayer 2: Multi-head attention over encoder output
        att_output2, global_attention_weights = self.enc_dec_attention(att_output1, enc_output, enc_output, padding_mask)
        att_output2 = self.dropout2(att_output2)
        att_output2 = self.layernorm2(att_output2 + att_output1)

        # Sublayer 3: Feed-forward network
        ff_output = self.feed_forward(att_output2)
        ff_output = self.dropout3(ff_output)
        ff_output = self.layernorm3(ff_output + att_output2)

        return ff_output, self_attention_weights, global_attention_weights
