#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python finetune.py "$@"
