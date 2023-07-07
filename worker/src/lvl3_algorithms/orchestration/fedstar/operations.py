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


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # This is only set to finish evaluation faster.
            if batch_idx * len(data) > 1024:
                break
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def PolyGen(degree, beta, input):
    """
    Function to generate coefficients for the polynomial
    The highest degree's coefficient is first
    Last element of array should be the input of the participant
    """
    coefficient_array = np.random.uniform(0, beta, degree - 1)
    return np.append(coefficient_array, input)


def PolyEval(polynomial, coefficient_array):
    # Need to add mod beta, haven't checked
    return np.polyval(polynomial, coefficient_array)


############################################
# Orchestration Class - NoiseParams        #
############################################

# Replicate for each class defined
class NoiseParams:
    def __init__(self, size):
        self.noise = None
        self.blind = None
        self.size = size

    def NoiseGen(norm_loc=0, norm_scale=0.0, lap_loc=0, lap_scale=1.0):
        """
        Parameters:
        j (int): Length of array
        norm_loc: Center of the gaussian distribution
        norm_scale: std dev of the gaussian distribution
        lap_loc: Center of laplace distribution
        lap_scale: Decay of laplace distribution

        Returns:
        arr1: ndarray of summed noise elements
        """
        gaussian_noise_array = np.random.normal(norm_loc, norm_scale, self.size)
        laplace_noise_array = np.random.laplace(lap_loc, norm_scale, self.size)
        self.noise = gaussian_noise_array + laplace_noise_array

    def SampleBlind(low, high):
        """
        Parameters:
        low: Lowest possible value of the distribution
        high: highest possible value of the distribution
        size: size of array

        Returns:
        Array of randomly sampled blind terms
        """

        return np.random.uniform(low, high, self.size)

    def total_encrypted_noise(binseq):
        encrypted_noise = self.noise * binseq
        total_noise = encrypted_noise + self.blind
        return total_noise


############################################
# Orchestration Class - <INSERT NAME HERE> #
############################################

# Replicate for each class defined


###########
# Scripts #
###########
