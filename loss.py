import torch
import torch.nn as nn


class loss_combine:
    def __init__(self, smooth=1e-6, alpha=0.5):
        self.bce_loss = nn.BCELoss()
        self.smooth = smooth
        self.alpha = alpha

    def __call__(self, outputs, targets):
        return self.alpha * self.bce_loss(outputs, targets) + (1 - self.alpha) * self.DiceLoss(outputs, targets)

    def DiceLoss(self, pred, true):
        # pred: (batch_size, 1, height, width)
        # true: (batch_size, 1, height, width)
        pred = torch.sigmoid(pred)
        pred = pred.view(pred.size(0), -1)
        true = true.view(true.size(0), -1)
        intersection = (pred * true).sum(1)
        dice = (1- ((2. * intersection + self.smooth) / (pred.sum(1) + true.sum(1) + self.smooth)))
        return dice.mean()
