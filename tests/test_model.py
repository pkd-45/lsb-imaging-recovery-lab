import torch

from lsbrecovery.model import TinyUNet


def test_model_shapes_for_one_and_three_channels():
    for c in (1, 3):
        model = TinyUNet(in_channels=c)
        y = model(torch.zeros(2, c, 64, 64))
        assert y.shape == (2, 2, 64, 64)
