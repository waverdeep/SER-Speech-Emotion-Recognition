import torch.optim as optim


def choose_optimizer(name, network,
                     lr=None,
                     momentum=None,
                     weight_decay=None):
    if name == 'SGD':
        return optim.SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

