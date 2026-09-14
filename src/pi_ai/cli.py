import argparse


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--threads", type=int, default=None)
    return parser
