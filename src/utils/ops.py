import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

def scale_to_range(x, old_min, old_max, new_min, new_max):
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)

def compute_scale_factor(x, y, dense_depth, inv_sparse_depths, h=480, w=640, scaling_method='pooling'):
    if scaling_method == 'pooling':
        depth = F.avg_pool2d(torch.tensor(dense_depth).reshape(1, 1, h, w), kernel_size=4, stride=4).squeeze()
        valid_mask = np.bitwise_and(depth[y, x] >= 0.2, depth[y, x] <= 1.2)
        scale_factor = (depth[y, x][valid_mask] * torch.from_numpy(inv_sparse_depths)[valid_mask]).median().item()
    elif scaling_method == 'rescaling':
        rescale = transforms.Compose([transforms.Resize((int(h/4), int(w/4)))])
        depth = rescale(torch.tensor(dense_depth).reshape(1, 1, h, w)).squeeze()
        # valid_mask = np.bitwise_and(depth[y, x] >= 0.2, depth[y, x] <= 1.2)
        # scale_factor = (depth[y, x][valid_mask] * torch.from_numpy(inv_sparse_depths)[valid_mask]).median().item()
        print('before')
        scale_factor = (depth[y, x] * inv_sparse_depths).median().item()
        print('after')
    return scale_factor