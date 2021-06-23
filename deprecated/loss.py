import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# THIS FILE CONTAINS THE DEPRECATED LOSSES FOR TRAINING THE MODELS
warnings.warn("This module is deprecated in favour of machine_learning", DeprecationWarning, stacklevel=2)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.reduction = reduction

    def forward(self, input_tensor, target):
        ce_loss = F.cross_entropy(input_tensor, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# Helper functions from fastai
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon, self.reduction = epsilon, reduction

    def forward(self, output, target):
        # number of classes
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        # (1-epsilon)* H(q,p) + epsilon*H(u,p)
        return (1 - self.epsilon) * nll + self.epsilon * (loss / c)
