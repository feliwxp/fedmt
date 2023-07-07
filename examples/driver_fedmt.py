#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import ray
import torch

# Libs
from http import client
import torch.nn as nn

# Custom
from ..ttp.src.lvl3_algorithms.statistical.fedmt.operations import (
    Model,
    client_selection,
)

from ..worker.src.lvl3_algorithms.statistical.fedmt.operations import initialise_workers

##################
# Configurations #
##################


#############
# Functions #
#############


@ray.remote
def start_training_job(model, model_v, delta, lr):
    from ..worker.src.lvl3_algorithms.statistical.fedmt.operations import train

    return train(model, model_v, delta, lr)


@ray.remote
def start_model_initialisation(Client, model):
    from ..worker.src.lvl3_algorithms.statistical.fedmt.operations import (
        initialise_model,
    )

    return initialise_model(Client, model)


###########
# Scripts #
###########


def main():
    """Function for multi-task federated learning"""

    # initialize ttp
    ray.init("ray://172.20.0.2:10001")

    # connect to clients and initialise local model
    addresses = ["ray://172.18.0.2:10001", "ray://172.19.0.3:10001"]
    clients = initialise_workers(addresses)

    # global parameter loading
    DELTA = 1
    LR = 1e-2
    T = 2
    K = 2

    # Receive model parameters to initialize model from orchestrator (human)
    model_params = [
        nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 3, kernel_size=3)
        ),
        nn.Sequential(nn.Linear(192, 10)),
    ]  # to do up model params

    # initialize global model
    global_model = Model(model_params)

    # get global weights to send to workers
    global_weights = global_model.get_weights()

    layers = global_weights.keys()

    # initialise local model for all clients
    for client in clients:
        with client.wrk_client:
            start_model_initialisation(client, Model(model_params))

    # start of T training rounds
    for t in T:
        # select k clients
        selected_clients = client_selection([0.4, 0.6], K)
        selected_workers = [clients[i] for i in selected_clients]

        for worker in selected_workers:
            all_weights_changes = []

            # ray client connection to worker
            with worker.wrk_client:

                # send model weights to worker first
                worker.model_v.load_state_dict(global_weights)  # actor remote

                # start training and get new weights
                weights_change = start_training_job.remote(
                    DELTA,
                    LR,
                    global_model,
                    worker.model_v,
                )

                all_weights_changes.append(weights_change.values())

                # save local model_v
                torch.save(worker.model_v, f"worker_{worker.round}_round.pt")

                # update training round number
                worker.model_v.increment_round()

        # aggregate global model weights through averaging
        for layer in layers:
            for weights_change in all_weights_changes:
                global_weights[layer] += weights_change[layer] / K

        # save global model
        torch.save(global_model, f"model_{t}_round.pt")


# need to save global model every round (model co-versioning) (worker saves vk)
# have state to track exported models with index (can use regex)

if __name__ == "__main__":
    main()
