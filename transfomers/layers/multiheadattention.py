import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size = 512, head_size = 8):
        super(MultiheadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def selfAttention(self, q, k ,v ,mask= None):
        """
                    Calculate Attention score
                    Parameters
                    ----------
                    q: tensor
                        query
                        shape: (..., q_length, d_k)
                    k: tensor
                        key
                        shape: (..., k_lengh, d_k)
                    v: tensor
                        value
                        shape: (..., v_length, d_v)
                    k_lengh = v_length

                    Returns
                    ----------
                    attention_weights: tensor
                        Attention Scores between Query and Key
                        shape: (..., q_length, k_lengh)
                    out: tensor
                        Attention Weights on Value
                        shape: (..., q_length, k_lengh)
        """
        dk = k.size(-1)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=torch.float32))
        if mask != None:
            attention_scores += (mask * -1e30)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def splitting_head(self, x):
        batch_size, length, _ = x.size()
        hd_v = self.hidden_size // self.head_size
        x = x.view(batch_size, length, self.head_size, hd_v)
        xs = x.transpose(1, 2)
        return xs

    def forward(self, q,k,v, mask = None):
        qw = self.wq(q)
        kw = self.wk(k)
        vw = self.wv(v)
        heads_qw = self.splitting_head(qw)
        heads_kw = self.splitting_head(kw)
        heads_vw = self.splitting_head(vw)
        output, attention_weights = self.selfAttention(heads_qw, heads_kw, heads_vw, mask)
        output = output.transpose(1,2)
        batch_size, _, _, hd_v = output.size()
        output = output.reshape(batch_size, -1, self.hidden_size)
        final = self.wo(output)
        return final, attention_weights







