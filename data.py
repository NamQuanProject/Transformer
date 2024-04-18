import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from constance import *
import re
import pickle
import os
import io
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors, decoders, Regex

class NMTDataset(Dataset):
    def __init__(self, inp_lang, trg_lang, vocab_folder):
        self.inp_lang = inp_lang
        self.trg_lang = trg_lang
        self.vocab_folder = vocab_folder
        self.inp_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.inp_lang)
        self.trg_tokenizer_path = '{}{}_tokenizer.pickle'.format(self.vocab_folder, self.trg_lang)

        self.inp_tokenizer = None
        self.trg_tokenizer = None

        if os.path.isfile(self.inp_tokenizer_path):
            # Loading tokenizer
            with open(self.inp_tokenizer_path, 'rb') as handle:
                self.inp_tokenizer = pickle.load(handle)

        if os.path.isfile(self.trg_tokenizer_path):
            # Loading tokenizer
            with open(self.trg_tokenizer_path, 'rb') as handle:
                self.trg_tokenizer = pickle.load(handle)

        self.dataset = []  # Initialize your dataset attribute

    def get_training_corpus(self, lang_dataset):
        for i in range(0, len(lang_dataset), 1000):
            yield lang_dataset[i: i + 1000]

    def preprocess_sentence(self, w, max_length):
        w = w.lower().strip()
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = w.strip()

        # Truncate Length up to max_length
        w = " ".join(w.split()[:max_length + 1])
        # Add start and end token
        w = '{} {} {}'.format(BOS, w, EOS)
        return w

    def build_tokenizer(self, lang_dataset):
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        tokenizer.normalizer = normalizers.Sequence([ normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),
                normalizers.Replace(Regex(r"[\s]"), " "),
                normalizers.Lowercase(),
                normalizers.NFD(), normalizers.StripAccents()
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
        tokenizer.train_from_iterator(self.get_training_corpus(lang_dataset), trainer=trainer)
        tokenizer.decoder = decoders.WordPiece(prefix="##")
        return tokenizer

    def tokenize(self, lang_tokenizer, lang, max_length):
        # Tokenize the language data
        tensor = [lang_tokenizer.encode(text).ids[:max_length] for text in lang]
        # Padding
        padded_tensor = self.pad_sequences(tensor, max_length)
        return padded_tensor

    def pad_sequences(self, sequences, max_length):
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) < max_length:
                padded_sequence = sequence + [0] * (max_length - len(sequence))
            else:
                padded_sequence = sequence[:max_length]
            padded_sequences.append(padded_sequence)
        return padded_sequences

    def load_dataset(self, inp_path, targ_path, max_length, num_examples):
        # Load and preprocess the data
        inp_lines = io.open(inp_path, encoding=UTF_8).read().strip().split('\n')[:num_examples]
        targ_lines = io.open(targ_path, encoding=UTF_8).read().strip().split('\n')[:num_examples]

        inp_lines = [self.preprocess_sentence(inp, max_length) for inp in inp_lines]
        targ_lines = [self.preprocess_sentence(targ, max_length) for targ in targ_lines]

        return inp_lines, targ_lines

    def build_dataset(self, inp_path, targ_path, batch_size, max_length, num_examples):
        # Load and preprocess the data
        inp_lines, targ_lines = self.load_dataset(inp_path, targ_path, max_length, num_examples)

        # Build tokenizers
        self.inp_tokenizer = self.build_tokenizer(inp_lines)
        self.trg_tokenizer = self.build_tokenizer(targ_lines)

        # Tokenize and pad the data
        inp_tensor = self.tokenize(self.inp_tokenizer, inp_lines, max_length)
        targ_tensor = self.tokenize(self.trg_tokenizer, targ_lines, max_length)

        # Save tokenizers
        if not os.path.exists(self.vocab_folder):
            os.makedirs(self.vocab_folder)
        with open(self.inp_tokenizer_path, 'wb') as handle:
            pickle.dump(self.inp_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.trg_tokenizer_path, 'wb') as handle:
            pickle.dump(self.trg_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Split data into training and validation sets
        inp_tensor_train, inp_tensor_val, targ_tensor_train, targ_tensor_val = train_test_split(
            inp_tensor, targ_tensor, test_size=0.2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp_tensor_train = torch.tensor(inp_tensor_train, device=device)
        targ_tensor_train = torch.tensor(targ_tensor_train, device=device)
        inp_tensor_val = torch.tensor(inp_tensor_val, device=device)
        targ_tensor_val = torch.tensor(targ_tensor_val, device=device)
        # Create PyTorch data loaders
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(inp_tensor_train), torch.tensor(targ_tensor_train))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(inp_tensor_val), torch.tensor(targ_tensor_val))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
