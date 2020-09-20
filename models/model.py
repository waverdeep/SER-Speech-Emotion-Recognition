import torch
import torch.nn as nn
import torch.nn.functional as F


class Vanilla_CNN(nn.Module):
    def __init__(self):
        super(Vanilla_CNN, self).__init__()
        conv1 = nn.Conv1d(1, 3)

