import torch
import src.loss


def test_jaccard_loss():

    x = torch.randn((1, 10, 100, 100, 10)) > 0.5
    y = torch.randn((1, 10, 100, 100, 10)) > 0.1

    assert x.max() == 1
    assert y.max() == 1

    out = src.loss.jaccard_loss(x, y)

    assert out > 0
    assert out < 1
