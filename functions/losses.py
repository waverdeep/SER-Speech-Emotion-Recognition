import torch.nn as nn


def choose_loss(name):
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()