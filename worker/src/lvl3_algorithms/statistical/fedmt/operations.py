#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import numpy as np
import os
import ray
import torch

# Libs
from filelock import FileLock
import torch.nn.functional as F
from torchvision import datasets, transforms

# Custom


##################
# Configurations #
##################


#############
# Functions #
#############


def initialise_workers(addresses, Client):

    clients = []
    for i in range(len(addresses)):
        wrk_client = ray.init(address=addresses[i], allow_multiple=True)
        clients.append(Client(wrk_client))

    return clients


def initialise_model(Client, model):
    Client.set_model(model)


def get_data_loader():
    """Safely downloads MNIST data.

    Returns:
        train_loader (torch.tensor): to be loaded during training
        test_loader (torch.tensor): to be loaded during testing
    """

    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=128,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("~/data", train=False, transform=mnist_transforms),
            batch_size=128,
            shuffle=True,
        )
    return train_loader, test_loader


# custom loss function
def custom_loss(output, target):
    """Custom loss function

    Args:
        output (np.array): Predicted prob(y) variable
        target (np.array): Target y variable

    Returns:
        loss (np.array)
    """
    loss = torch.mean((output - target) ** 2)
    return loss


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset.

    Args:
        model: model to be evaluated
        testloader (torch.tensor): used for validation at worker

    Returns:
        accuracy (float)
    """
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


def train_single(model, ps):
    """Train a single model

    Args:
        model (object): model to be trained
        ps (object): parameter server

    """

    # hyperparameters
    INTERATIONS = 80
    NUM_WORKERS = 2

    # initialize ray with workers
    ray.init(ignore_reinit_error=True)
    workers = [DataWorker.remote(model) for i in range(NUM_WORKERS)]

    # load dataloader
    test_loader = get_data_loader()[1]

    # get weights of parameter server
    current_weights = ps.get_weights.remote()

    gradients = {}

    # assign gradients of each worker
    for worker in workers:
        gradients[worker.compute_gradients.remote(current_weights)] = worker

    # train each worker
    for i in range(INTERATIONS * NUM_WORKERS):
        ready_gradient_list, _ = ray.wait(list(gradients))
        ready_gradient_id = ready_gradient_list[0]
        worker = gradients.pop(ready_gradient_id)

        # Compute and apply gradients.
        current_weights = ps.apply_gradients.remote(*[ready_gradient_id])
        gradients[worker.compute_gradients.remote(current_weights)] = worker

        if i % 10 == 0:
            # Evaluate the current model after every 10 updates.
            model.set_weights(ray.get(current_weights))
            accuracy = evaluate(model, test_loader)
            print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

    print("Final accuracy is {:.1f}.".format(accuracy))


def _get_weights_change(global_weights, local_weights):
    """helper function to calculate weights change

    Args:
        global_weights (dict): original global weights
        local_weights (dict): global weights after local training

    Returns:
        dict: changes in global weights
    """
    weights_change = dict(local_weights)

    for layer in list(global_weights.keys()):
        weights_change[layer] = local_weights[layer] - global_weights[layer]

    return weights_change


def train(delta, lr, model, model_v):
    """Train both global and local models

    Args:
        delta (float): hyperparameter in L2 regularizer
        lr (float): learning rate of optimizer
        model (object): global model to be trained
        model_v (object): local model to be trained

    Returns:
        weights_change (dict): changes in global weights
    """

    # train global model
    global_weights = model.get_weights()
    ps = ParameterServer.remote(lr, model)
    train_single(model, ps)

    # train local model with weights of updated global model
    ps_v = ParameterServer_v.remote(delta, lr, model, model_v)
    train_single(model_v, ps_v)

    # calculate change in global weights
    weights_change = _get_weights_change(global_weights, model.get_weights())

    return weights_change


############################################
# Orchestration Class - <ParameterServer> #
############################################


@ray.remote
class ParameterServer(object):
    """This parameter server holds a copy of the global model.
    During training, it will:
    - Receive gradients and apply them to its model.
    - Send the updated model back to the workers.

    Attributes:
        model (object): model to train
        optimizer (torch.optim): optimizer to train model
    """

    def __init__(self, lr, model):
        # General attributes
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def apply_gradients(self, *gradients):
        """Apply gradients to model and optimise model

        Returns:
            dict: updated weights of model
        """
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        """Get weights of model

        Returns:
            dict: weights of model
        """
        return self.model.get_weights()


############################################
# Orchestration Class - <ParameterServer_v> #
############################################


@ray.remote
class ParameterServer_v(object):
    """This parameter server holds a copy of the local model_v.
    During training, it will:
    - Receive gradients.
    - Calculate gradients based on solver for multi-task learning objective.
    - Apply gradients to its model.
    - Send the updated model back to the workers.

    Attributes:
        model (object): model to train
        optimizer (torch.optim): optimizer to train model
        delta (float): hyperparameter in L2 regularizer
        lr (float): learning rate of optimizer
    """

    def __init__(self, delta, lr, model, model_v):
        # General attributes
        self.model_v = model_v
        self.optimizer = torch.optim.SGD(self.model_v.parameters(), lr=lr)
        self.delta = delta
        self.model_weights = model.get_weights()

    def apply_gradients(self, *gradients):
        """Apply gradients to model and optimise model

        Returns:
            dict: updated weights of model
        """
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
        ]
        # add weights difference to local grad loss
        model_weights = self.model_v.get_weights()

        layers = list(model_weights.keys())

        for i in range(len(layers)):
            summed_gradients[i] += summed_gradients[i] + self.delta * (
                model_weights[layers[i]].numpy() - self.model_weights[layers[i]].numpy()
            )  # to change this part (zip)

        self.optimizer.zero_grad()
        self.model_v.set_gradients(summed_gradients)

        self.optimizer.step()

        return self.model_v.get_weights()

    def get_weights(self):
        """Get weights of model

        Returns:
            dict: weights of model
        """
        return self.model_v.get_weights()


############################################
# Orchestration Class - <DataWorker> #
############################################


@ray.remote
class DataWorker(object):
    """The worker will also hold a copy of the model. During training.
    It will continuously evaluate data and send gradients to the parameter server.
    The worker will synchronize its model with the Parameter Server model weights.

    Attributes:
        model (object): model to train
        data_iterator (iter): interator of train_loader of dataloader
    """

    def __init__(self, model):
        # General attributes
        self.model = model
        self.data_iterator = iter(get_data_loader()[0])

    def compute_gradients(self, weights):
        """Compute gradients of model

        Args:
            weights (dict): model weights

        Returns:
            dict: gradients
        """
        self.model.set_weights(weights)
        try:
            data, target = next(self.data_iterator)
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.data_iterator = iter(get_data_loader()[0])
            data, target = next(self.data_iterator)
        self.model.zero_grad()
        output = self.model(data)
        loss = loss(output, target)
        loss.backward()
        return self.model.get_gradients()


############################################
# Orchestration Class - <Client> #
############################################


class Client(object):
    """Represents a client as a cluster in ray

    Attributes:
        wrk_client (object): ClientContext in ray
        round (int): the current training round of the local model_v
        model_v (object): local model_v of client
    """

    def __init__(self, wrk_client):
        # General attributes
        self.wrk_client = wrk_client

        # Data attributes
        self.round = 1

    ###########
    # Setters #
    ###########
    def set_model(self, model_v):
        """Set model

        Args:
            model_v (object): local model_v of client
        """
        self.model_v = model_v

    ###########
    # Core Functions #
    ###########
    def increment_round(self):
        """Increase training round number by 1"""
        self.round += 1


###########
# Scripts #
###########
