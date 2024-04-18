import torch
import torch.nn.functional as F
from transfomers.layers.testTransfomer import Transformer

# Define batch size, sequence size, and embedding size
batch_size = 2
sequence_size = 2
embedding_size = 512

# Create a random batch tensor with the specified shape
encoder_input_tensor = torch.randint(0, 1000, (5, 29)) # 1 Batch -> 5 sentence -> 29 words
decoder_input_tensor = torch.randint(0, 800, (5, 30)) # 1 Barch -> 5 sentence -> 29 words

model = Transformer(head_size=8, N=8, inp_vocab_size=1000, trg_vocab_size=800, hidden_size=512, d_ff=2048, dropout_rate=0.1, eps=1e-6)
output = model(encoder_input_tensor, decoder_input_tensor)
output = F.softmax(output, dim=-1)

# Get the index of the maximum value along the last dimension
print(output.size())
max_values, max_indices = torch.max(output, dim=-1)

print("Max props of values", max_values.size())
print("Indices of maximum values:", max_indices.size())

# -> 1 Batch -> 5 sentence -> 30 indices
