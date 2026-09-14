# Raspberry Pi AI

This repository is a complete, CPU-first PyTorch image-classification starter
for Raspberry Pi 5. The default `TinyPiCNN` uses depthwise-separable
convolutions, small feature maps, and global average pooling. It avoids a
large flattened layer and defaults to two CPU threads, zero data-loader
workers, and float32 execution.

## Install

Use a PyTorch wheel compatible with the Pi OS and Python version first. Then:

```bash
python -m pip install -r requirements-dev.txt
export PYTHONPATH=src
```

## Run

The starter command runs one real training path with generated tensors, so it
is useful as an installation smoke test before adding data:

```bash
PYTHONPATH=src python train.py --config config.yaml --epochs 1
PYTHONPATH=src python benchmark.py --repetitions 20
```

For a real dataset, put images in `data/raw/class_name/image.jpg`, then adapt
the loader in `train.py` to use `FolderImageDataset`. The public package APIs
are in `src/pi_ai`: `build_model`, `Trainer`, `Predictor`, checkpoint helpers,
and TorchScript export.

Use `--threads 1` or a smaller batch if the Pi becomes hot. Measure latency on
the target board; desktop CPU timings are not representative.

The original single-file example remains available as `pi_model.py`.