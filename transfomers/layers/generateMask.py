import torch


def generate_padding_mask(inp):
    """
    Generate a padding mask for the input sequence.

    Parameters:
        inp (torch.Tensor): Input sequence tensor.

    Returns:
        torch.Tensor: Padding mask tensor.
    """
    mask = (inp == 0).unsqueeze(1).unsqueeze(2)  # Add extra dimensions for broadcasting
    return mask.float()


def generate_look_ahead_mask(inp_len):
    """
    Generate a look-ahead mask for the input sequence length.

    Parameters:
        inp_len (int): Length of the input sequence.

    Returns:
        torch.Tensor: Look-ahead mask tensor.
    """
    mask = torch.triu(torch.ones(inp_len, inp_len), diagonal=1)
    return mask.bool()  # Convert to boolean mask


def generate_mask(inp, targ):
    """
    Generate masks for encoder and decoder inputs.

    Parameters:
        inp (torch.Tensor): Input sequence tensor.
        targ (torch.Tensor): Target sequence tensor.

    Returns:
        Tuple[torch.Tensor]: Tuple containing encoder padding mask, decoder look-ahead mask, and decoder padding mask.
    """
    # Ensure the device is the same as the input tensors
    device = inp.device

    # Encoder Padding Mask
    encoder_padding_mask = generate_padding_mask(inp)

    # Decoder Padding Mask: Use for global multi head attention for masking encoder output
    decoder_padding_mask = generate_padding_mask(inp)

    # Look Ahead Padding Mask
    decoder_look_ahead_mask = generate_look_ahead_mask(targ.size(1))

    # Decoder Padding Mask
    decoder_inp_padding_mask = generate_padding_mask(targ)
    # Combine look-ahead mask and input padding mask for decoder
    decoder_look_ahead_mask = torch.max(decoder_look_ahead_mask, decoder_inp_padding_mask)

    # Move masks to the same device as input tensors
    encoder_padding_mask = encoder_padding_mask.to(device)
    decoder_look_ahead_mask = decoder_look_ahead_mask.to(device)
    decoder_padding_mask = decoder_padding_mask.to(device)

    return encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask
