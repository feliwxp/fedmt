#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import numpy as np
import torch

# Libs
import torch.nn as nn
import torch.nn.functional as F

# Custom


##################
# Configurations #
##################


#############
# Functions #
#############
def client_selection(pk, k):
    """Select clients for training round t

    Args:
        pk (list[float]): list of probability pk for each client
        k (int): k selected clients

    Returns:
        clients (list[int]): list of indexes of clients selected
    """
    devices = list(range(len(pk)))
    weighted_pk = [x / sum(pk) for x in pk]
    clients = np.random.choice(devices, size=k, replace=False, p=weighted_pk)
    print("Clients selected: {}".format(clients))
    return clients


############################################
# Orchestration Class - <Model> #
############################################

'''
class ConvNet(nn.Module):
    """Small ConvNet for MNIST."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
'''


class Model(nn.Module):

    """Model that gets model params as inputs

    Attributes:
        cnn_layers (nn.Sequential): cnn layers of model
        linear_layers (nn.Sequential): linear layers of model
    """

    def __init__(self, model_params):
        super(Model, self).__init__()
        self.cnn_layers = model_params[0]
        self.linear_layers = model_params[1]

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


###########
# Scripts #
###########
