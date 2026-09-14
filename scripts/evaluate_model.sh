#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python evaluate.py "$@"
