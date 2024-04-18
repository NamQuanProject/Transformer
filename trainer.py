import torch
from transfomers.layers.generateMask import generate_mask

class Trainer:
    def __init__(self, model, optimizer, scheduler, epochs, checkpoint_folder):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.checkpoint_folder = checkpoint_folder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def cal_acc(self, real, pred):
        accuracies = torch.eq(real, torch.argmax(pred, dim=2))

        mask = torch.logical_not(torch.eq(real, 0))
        accuracies = torch.logical_and(mask, accuracies)

        accuracies = accuracies.float()
        mask = mask.float()
        return torch.sum(accuracies) / torch.sum(mask)

    def loss_function(self, real, pred):
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(pred.permute(0, 2, 1), real)

        return loss

    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        inp, tar = inp.to(self.device), tar.to(self.device)


        encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(inp, tar_inp)

        self.optimizer.zero_grad()

        preds = self.model(inp, tar_inp, encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask)
        loss = self.loss_function(tar_real, preds)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), self.cal_acc(tar_real, preds).item()

    def train(self, data_loader):
        print('=============Training Progress================')
        print('----------------Begin--------------------')

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_acc = 0.0

            for batch, (inp, tar) in enumerate(data_loader):
                loss, acc = self.train_step(inp, tar)
                total_loss += loss
                total_acc += acc

                if batch % 50 == 0:
                    print(f'Epoch {epoch + 1} Batch {batch} Loss {total_loss / (batch + 1):.3f} Accuracy {total_acc / (batch + 1):.3f}')

            print(f'Epoch {epoch + 1} Loss: {total_loss / len(data_loader):.3f} Accuracy: {total_acc / len(data_loader):.3f}')

        print('----------------Done--------------------')

    def predict(self, encoder_input, decoder_input, max_length, end_token):
        print('=============Inference Progress================')
        print('----------------Begin--------------------')

        for i in range(max_length):
            encoder_padding_mask, decoder_look_ahead_mask, decoder_padding_mask = generate_mask(encoder_input,
                                                                                                    decoder_input)

            preds = self.model(encoder_input, decoder_input, encoder_padding_mask, decoder_look_ahead_mask,
                               decoder_padding_mask)

            preds = preds[:, -1:, :]  # Get the last prediction

            predicted_id = torch.argmax(preds, dim=-1).item()

            decoder_input = torch.cat([decoder_input, torch.tensor([[predicted_id]])], dim=-1)

            # Return if the predicted_id matches the end_token
            if predicted_id == end_token:
                break

        return decoder_input
