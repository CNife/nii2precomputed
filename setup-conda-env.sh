#! /usr/bin/env bash
set -euo pipefail

envName="$1"

apt update
apt upgrade -y
apt install -y libgl1

conda config --set always_yes true
conda create -n "$envName" python=3.10
conda activate "$envName"
conda install -c fenglab zimg
conda install numpy rich typer
pip install tensorstore
conda clean -a
