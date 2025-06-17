# src/utils.py
import torch.nn as nn

def get_loss_function(loss_name: str, reduction: str = 'mean', **kwargs):
    """
    Returns a PyTorch loss function instance based on its name.
    Args:
        loss_name (str): Name of the loss function (e.g., "MSELoss", "BCEWithLogitsLoss", "L1Loss", "CrossEntropyLoss").
        reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        **kwargs: Additional arguments for the loss function constructor.
    Returns:
        torch.nn.modules.loss._Loss: An instance of the loss function.
    """
    loss_name_lower = loss_name.lower()
    if loss_name_lower == "mseloss":
        return nn.MSELoss(reduction=reduction, **kwargs)
    elif loss_name_lower == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss(reduction=reduction, **kwargs)
    elif loss_name_lower == "l1loss":
        return nn.L1Loss(reduction=reduction, **kwargs)
    elif loss_name_lower == "crossentropyloss":
        return nn.CrossEntropyLoss(reduction=reduction, **kwargs)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}. "
                         f"Supported: MSELoss, BCEWithLogitsLoss, L1Loss, CrossEntropyLoss.")