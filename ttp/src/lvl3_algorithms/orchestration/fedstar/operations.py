#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
import math


# Libs
import crypten
from filelock import Filelock
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchvision import datasets, transforms

# Custom


##################
# Configurations #
##################


#############
# Functions #
#############


@ray.remote
def create_noise_params(size):
    from src.worker.encryption import NoiseParams

    return NoiseParams(size)


############################################
# Orchestration Class - ParameterServer #
############################################

# Use a remote class when you 'need' distributed computing
# i.e you want to create massive amounts of these remote actors
# and call particular methods.
# I would like to set n = m to test out the niche case
# Should I set SecurityParams in the ParameterServer class?


class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


############################################
# Orchestration Class - SecurityParams #
############################################

# Replicate for each class defined


class SecurityParams:
    def __init__(self, num_of_participants, t, l, n, degree):
        self.num_of_participants = num_of_participants
        self.t = t
        self.l = l
        self.n = n
        self.degree = degree

    # def calc_ci(n, m, t):
    #     # need to fix
    #     if n % m == 0:
    #         s_i = np.floor(n/m) + 1
    #     else:
    #         s_i = np.floor(n/m)

    #     return s_i * t

    def calculate_j(n, num_of_participants, t):
        return t * math.ceil(n / m)


###########
# Scripts #
###########
