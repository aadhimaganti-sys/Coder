#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python train.py --config config.yaml "$@"
