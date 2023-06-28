import torch
import numpy as np

def init_params(normalized_expert, k, dev = "cpu"):
    num_init = 1

    means = torch.empty((k, normalized_expert.shape[1], normalized_expert.shape[2]), device = dev)

    num_expert = normalized_expert.shape[0]
    means[0] = normalized_expert[np.random.choice(num_expert), :, :]

    while num_init< k:
        diff = normalized_expert[None, :, :, :] - means[0:num_init, None, :, :]
        dist = torch.sum(diff * diff, dim=(2, 3))

        #dist is clusters x points
        closest_c_dist, min_indices = torch.min(dist, dim = 0)
        total_dist = torch.sum(closest_c_dist)
        means[num_init] = normalized_expert[np.random.choice(num_expert, p = closest_c_dist.cpu().numpy()/total_dist.item())]

        num_init+=1
        del diff
        del dist
        del closest_c_dist
        del min_indices
        del total_dist


    return means

def compute_sigma(normalized_expert, means):
