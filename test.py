from data import NMTDataset
import torch
import os
from model import Transformer
from trainer import Trainer
from optimizer import CustomLearningRate

home_dir = os.getcwd()
texts = ["Tom had a hard time raising enough money build the orphanage he'd promised to build It's so unfair. Everybody sat down If I knew it, I would tell you. Tom didn't notice the look of disdain on Mary's facTom hasn't gone anywhere. He's still in his room."]
input_lang = "en"
target_lang = "vi"
vocab_folder = '{}/saved_vocab/transformer/'.format(home_dir)

# Load dataset and tokenizer
nmt_dataset = NMTDataset(input_lang, target_lang, vocab_folder)
tokenizer = nmt_dataset.inp_tokenizer
tokenized_sequences = [tokenizer.encode(text).ids for text in texts]
tokenized_tensors = [torch.tensor(ids, dtype=torch.long) for ids in tokenized_sequences]
# Encode the input text
padded_sequences = torch.nn.utils.rnn.pad_sequence(tokenized_tensors, batch_first=True)
encoder_input = torch.tensor(padded_sequences, dtype=torch.long)
print(encoder_input)

start, end = tokenizer.encode('<start>').ids[0], tokenizer.encode('<end>').ids[0]
# Initialize the Transformer model
hyperparameters = {
    "N": 8,
    "inp_vocab_size": 25000,
    "trg_vocab_size": 25000,
    "head_size": 8,
    "hidden_size": 512,
    "d_ff": 2048,
    "dropout_rate": 0.5,
    "eps": 10
}

# Convert start and end tokens to indices
start_token = '<BOS>'
end_token = '<EOS>'
model = Transformer(**hyperparameters)
model_folder = '{}/model/transformer.h5'.format(home_dir)  # Change the file extension to .pth for PyTorch
model.load_state_dict(torch.load(model_folder))

# Initialize optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CustomLearningRate(optimizer, d_model=512)

# Initialize Trainer
checkpoint_folder = '{}/checkpoints/'.format(home_dir)
trainer = Trainer(model, optimizer, scheduler, 10, checkpoint_folder)

# Define a function to map tokens to indices

start_index = tokenizer.encode(start_token).ids
end_index = tokenizer.encode(end_token).ids
# Encode the start token
decoder_input = torch.tensor([start_index], dtype=torch.int64)

_ = model(encoder_input, decoder_input, None, None, None)

with torch.no_grad():
    result = trainer.predict(encoder_input, decoder_input, 50, end_token)
    print(result)
    decode_tokenizer = nmt_dataset.trg_tokenizer
    result_list = result[0].tolist()
    final = decode_tokenizer.decode(result_list)
    print(result_list)
    print(final)
