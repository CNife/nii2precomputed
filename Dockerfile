FROM continuumio/miniconda3:23.5.2-0 AS build

RUN conda install conda-pack

COPY conda-env.yaml .
RUN conda env create -f conda-env.yaml -n tmp_env

RUN conda-pack -n tmp_env -o /tmp/tmp_env.tar.gz -j -1 && \
    mkdir /env && \
    tar -xf /tmp/tmp_env.tar.gz -C /env && \
    rm /tmp/tmp_env.tar.gz
RUN /env/bin/conda-unpack

FROM debian:bookworm-slim AS runtime

COPY --from=build /env /env

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="${PYTHONPATH:-}:/code"
SHELL ["/usr/bin/env", "bash", "-c"]
RUN echo 'source /env/bin/activate' >> ~/.bash_profile

COPY --chmod=744 entrypoint.sh /
COPY convert_to_precomputed /code/convert_to_precomputed
COPY vendor /code/vendor

ENTRYPOINT ["/entrypoint.sh"]
