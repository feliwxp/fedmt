##############
# Base Image #
##############

FROM rayproject/ray:1.13.0-py39-gpu as base

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

ADD . /orchestrator
WORKDIR /orchestrator