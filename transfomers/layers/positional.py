import numpy as np
import torch


def create_angle_rates(d_model):
    # Create angle rates
    angles = np.arange(d_model)
    angles[1::2] = angles[0::2]
    angles = 1 / (10000 ** (angles / d_model))
    angles = np.expand_dims(angles, axis=0)
    return angles


def generate_positional_encoding(pos, d_model):
    # Generate positional encoding
    angles = create_angle_rates(d_model)
    pos = np.expand_dims(np.arange(pos), axis=1)
    pos_angles = np.matmul(pos, angles)
    pos_angles[:, 0::2] = np.sin(pos_angles[:, 0::2])
    pos_angles[:, 1::2] = np.cos(pos_angles[:, 1::2])
    pos_angles = np.expand_dims(pos_angles, axis=0)

    return torch.tensor(pos_angles, dtype=torch.float32)

