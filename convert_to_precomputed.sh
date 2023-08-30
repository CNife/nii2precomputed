#! /usr/bin/env bash
set -euo pipefail

scriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${scriptDir}:${PYTHONPATH}"
python -m convert_to_precomputed "$@"
