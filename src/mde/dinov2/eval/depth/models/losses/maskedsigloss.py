import torch
import torch.nn as nn

from ...models.builder import LOSSES

@LOSSES.register_module()
class MaskedSigLoss(nn.Module):
    """MaskedSigLoss.
    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100, loss_name="masked_sigloss"
    ):
        super(MaskedSigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def masked_sigloss(self, input, target, obj_mask):
        '''
            obj_mask is an OpenCV binary mask
        '''
        # Apply mask to both input and target depths
        input[obj_mask == 0] = 0
        target[obj_mask == 0] = 0

        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, img_meta):
        """Forward function.
        Expects load_obj_masks set to True when building batches with the BatchCollector."
        """
        loss_masked_depth = self.loss_weight * self.masked_sigloss(depth_pred, depth_gt, img_meta["obj_mask"])
        return loss_masked_depth
