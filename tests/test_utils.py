import pytest
import torch.nn as nn

from src.utils import get_loss_function

@pytest.mark.parametrize("loss_name, expected_class", [
    ("MSELoss", nn.MSELoss),
    ("BCEWithLogitsLoss", nn.BCEWithLogitsLoss),
    ("L1Loss", nn.L1Loss),
    ("CrossEntropyLoss", nn.CrossEntropyLoss),
])
def test_get_loss_function_valid(loss_name, expected_class):
    loss_fn = get_loss_function(loss_name)
    assert isinstance(loss_fn, expected_class)


def test_get_loss_function_invalid():
    with pytest.raises(ValueError):
        get_loss_function("InvalidLoss")
