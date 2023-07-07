# Federated Prototypes

Main repo hosting federated prototypes for Synergos

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Installation

```
# Download source repository
git clone https://gitlab.aisingapore.net/aims/federatedlearning/100e-bricks/federated-prototypes
```

## Usage
```
# Start 2 worker Clusters, each comprising of 1 head Ray node and 1 worker Ray node 
docker compose \
    -f docker-compose_workerCluster_1.yml \
    -f docker-compose_workerCluster_2.yml \
    up [--build]

# Terminate worker clusters
docker-compose \
    -f docker-compose_workerCluster_1.yml \
    -f docker-compose_workerCluster_2.yml \
    down
```

## Roadmap

1. Cryptographic Masking

2. Ambiguity Injection

3. Algorithmic Optimizations
    - Orchestration
        - [ ] FedCS
        - [ ] FedDropout
        - [ ] FedQuant
        - [ ] FedStar

    - Statistical
        - [ ] FedD
        - [ ] FedGNN
        - [ ] FedMT
        - [ ] FedProg
        - [ ] FedSem
        - [ ] FedSplit

4. Adversarial Mitigations

5. Contribution Calculation

6. Reward Calculation

7. Federated Explanability




## Contributing


## Authors and acknowledgment
