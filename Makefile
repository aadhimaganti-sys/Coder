test:
	pytest

train:
	python train.py --config config.yaml

benchmark:
	python benchmark.py
