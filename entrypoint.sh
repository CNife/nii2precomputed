#! /usr/bin/env bash
source $HOME/.bashrc
set -euo pipefail

conda activate "$1"
python /code/main.py "${@:2}"
