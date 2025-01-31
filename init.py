import torch.nn as nn
import torch.nn.init as init

def xavier_uniform_relu(modules):
    xavier_uniform_initialisation('relu')

def xavier_uniform_sigmoid(modules):
    xavier_uniform_initialisation('sigmoid')

def xavier_uniform_initialisation(gain):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight.data, gain=init.calculate_gain(gain))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()