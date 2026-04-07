import torch
import torch.nn.functional as F


def compute_class_weights(labels, num_classes, mask=None, eps=1e-8):
    """
    Compute inverse-frequency class weights from labels.

    Args:
        labels (Tensor): Long tensor of labels (can include -1 for unknown).
        num_classes (int): Number of valid classes.
        mask (Tensor | None): Optional boolean mask to select labels (e.g., train mask).
        eps (float): Small value to avoid division by zero.
    """
    if mask is not None:
        labels = labels[mask]

    valid = labels >= 0
    labels = labels[valid]

    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum().clamp_min(eps)
    weights = total / (num_classes * counts.clamp_min(eps))
    return weights


def nll_loss_with_optional_weights(log_probs, targets, class_weights=None):
    """
    NLL loss with optional class weights for imbalance handling.
    """
    if class_weights is None:
        return F.nll_loss(log_probs, targets)
    return F.nll_loss(log_probs, targets, weight=class_weights)
