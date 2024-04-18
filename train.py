import torch.nn as nn
import torch
import torch.nn.functional
from data import NMTDataset
from optimizer import CustomLearningRate
from model import Transformer
from argparse import ArgumentParser
import os
from trainer import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs")
    home_dir = os.getcwd()
    # parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--input-lang", default='en', type=str, required=True)
    parser.add_argument("--target-lang", default='vi', type=str, required=True)
    parser.add_argument("--input-path", default='{}/data/train/train.en'.format(home_dir), type=str)
    parser.add_argument("--target-path", default='{}/data/train/train.vi'.format(home_dir), type=str)
    parser.add_argument("--vocab-folder", default='{}/saved_vocab/transformer/'.format(home_dir), type=str)
    parser.add_argument("--checkpoint-folder", default='{}/checkpoints/'.format(home_dir), type=str)
    parser.add_argument("--buffer-size", default=64, type=str)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--max-length", default=40, type=int)
    parser.add_argument("--num-examples", default=1000000, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--n", default=8, type=int)
    parser.add_argument("--h", default=8, type=int)
    parser.add_argument("--d-ff", default=2048, type=int)
    parser.add_argument("--activation", default='relu', type=str)
    parser.add_argument("--dropout-rate", default=0.1, type=float)
    parser.add_argument("--eps", default=0.1, type=float)
    parser.add_argument("--model_folder", default='{}/model/transformer.h5'.format(home_dir), type=str)
    args = parser.parse_args()

    nmt_dataset = NMTDataset(args.input_lang, args.target_lang, args.vocab_folder)
    train_dataset, val_dataset = nmt_dataset.build_dataset(args.input_path, args.target_path, args.batch_size, args.max_length, args.num_examples)
    inp_tokenizer, targ_tokenizer = nmt_dataset.inp_tokenizer, nmt_dataset.trg_tokenizer

    inp_vocab_size = 25000
    targ_vocab_size = 25000

    hyperparameters = {
        "N": args.n,
        "inp_vocab_size": inp_vocab_size,
        "trg_vocab_size": targ_vocab_size,
        "head_size": args.h,
        "hidden_size": args.hidden_size,
        "d_ff": args.d_ff,
        "dropout_rate": args.dropout_rate,
        "eps": args.eps
    }
    model = Transformer(**hyperparameters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CustomLearningRate(optimizer, d_model=512)
    checkpoint_folder = args.checkpoint_folder
    epochs = args.epochs
    trainer = Trainer(model, optimizer, scheduler, epochs, checkpoint_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model.to(device)
    trainer.train(train_dataset)

    torch.save(model.state_dict(), args.model_folder)
    print("__SAVE SUCCESSFULLY__")



