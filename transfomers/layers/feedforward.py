import torch
import torch.nn as nn

def ffn(dff=2048, d_model=512):
    layers = [
        nn.Linear(d_model, dff),
        nn.ReLU(), # Assuming ReLU as the default activation
        nn.Linear(dff, d_model),
    ]
    return nn.Sequential(*layers)

