from pi_ai.utils.seed import seed_everything


def configure_reproducibility(seed: int) -> None:
    seed_everything(seed)
