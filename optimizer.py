import torch
import math
from torch.optim.lr_scheduler import OneCycleLR, MultiplicativeLR
class CustomLearningRate(torch.optim.lr_scheduler.MultiplicativeLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(CustomLearningRate, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        lrate = self.d_model ** (-0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lrate for _ in self.base_lrs]


