def test_model_shape(model, batch):
    assert model(batch).shape == (4, 3)
