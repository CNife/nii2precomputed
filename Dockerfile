FROM ubuntu:23.10 AS setup-conda

ARG condaVersion=py310_23.3.1-0
ARG sha256Sum=aef279d6baea7f67940f16aad17ebe5f6aac97487c7c03466ff01f4819e5a651
ARG condaHome=/root/conda

ENV PATH="${condaHome}/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN set -x && \
    echo "$sha256Sum miniconda.sh" > checksum && \
    wget -O miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-${condaVersion}-Linux-x86_64.sh && \
    sha256sum --check --status checksum && \
    bash miniconda.sh -b -p $condaHome && \
    rm miniconda.sh checksum && \
    ln -s $condaHome/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo "source $condaHome/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find $condaHome -follow -type f -name '*.a' -delete && \
    find $condaHome -follow -type f -name '*.js.map' -delete && \
    $condaHome/bin/conda clean -afy

CMD bash

FROM setup-conda AS setup-env

WORKDIR /code
ENV PYTHONPATH="/code"

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/usr/bin/env", "bash", "-c"]
RUN source $condaHome/etc/profile.d/conda.sh && \
    conda activate base && \
    conda install -y --update-all python=3.10 && \
    conda install -y -c fenglab zimg && \
    conda install -y numpy loguru typer && \
    pip install tensorstore

COPY convert_to_precomputed /code/convert_to_precomputed/
COPY vendor /code/vendor
CMD ["python", "-m", "convert_to_precomputed", "--help"]
