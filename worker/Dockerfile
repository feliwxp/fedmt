##############
# Base Image #
##############

FROM rayproject/ray:1.13.0-py39-gpu as base

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ADD . /worker
WORKDIR /worker

# FROM python:3.9.12-slim-buster as base

# RUN apt-get update
# RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     build-essential \
#     git\
#     pciutils

# EXPOSE 6379
# EXPOSE 8265

# ENTRYPOINT ["ray", "start", "--address", "host.docker.internal"]
# ENTRYPOINT ["python", "-m", "src.main"]
# ENTRYPOINT [                                \
#     "ray", "start",                         \
#         "--head",                           \
#         "--port=6379",                      \
#         "--redis-shard-ports=6380,6381",    \
#         "--object-manager-port=22345",      \
#         "--node-manager-port=22346",        \
#         "--dashboard-host=0.0.0.0",         \
#         "--block"                           \
# ]
# CMD ["--help"]

#  docker build --tag ray_worker:test .
# docker run --shm-size=10.3 -t -i -p 8265:8265 -p 6379:6379  rayproject/ray:1.13.0-py39-gpu