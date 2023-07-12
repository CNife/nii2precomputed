FROM ubuntu:23.10

ARG minicondaUrl=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
ARG condaHome=/root/conda
ARG envName=nii2precomputed

WORKDIR /code

RUN apt update && \
    apt install -y wget &&  \
    wget -O miniconda.sh $minicondaUrl && \
    bash miniconda.sh -b -p $condaHome && \
    rm miniconda.sh && \
    $condaHome/bin/conda init bash

COPY main.py nii_2_precomputed.py util.py setup-conda-env.sh entrypoint.sh /code/
RUN bash /code/setup-conda-env.sh $envName

ENTRYPOINT /code/entrypoint.sh $envName
