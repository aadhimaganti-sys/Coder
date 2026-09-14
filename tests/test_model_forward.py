def test_forward_is_finite(model, batch):
    assert model(batch).isfinite().all()
