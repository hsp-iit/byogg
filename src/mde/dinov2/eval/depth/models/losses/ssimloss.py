import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

from ...models.builder import LOSSES

@LOSSES.register_module()
class SSIMLoss(nn.Module):

    def __init__(self, win_size, valid_mask=True, loss_weight=1.0, min_depth=None, max_depth=None, loss_nome="loss_structural"):
        super(SSIMLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001
    
    def ssimloss(self, pred, target):
        pred = torch.clamp(pred, min_depth, max_depth)
        target = torch.clamp(target, min_depth, max_depth)

        pred = (pred - pred.min()) / (pred.max() - pred.min())
        target = (target - target.min()) / (target.max() - target.min())

        score = ssim(target, pred, win_size=3, data_range=1.0)

        # from range -[-1, 1] to [0, 1] and then inverted to compute a loss metric (we want to minimize the residual)
        ssim_loss = 1 - (score + 1) / 2

        return ssim_loss

    def forward(self, pred, target):

        ssim_loss = self.loss_weight * self.ssimloss(pred, target)
        return ssim_loss
