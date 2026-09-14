from pi_ai.training.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path, model):
    path = tmp_path / "model.pt"
    save_checkpoint(path, model)
    payload = load_checkpoint(path, model)
    assert payload["epoch"] == 0
