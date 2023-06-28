import numpy as np
import torch

def init_means(normalized_expert, k, dev = "cpu"):
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

def get_flattened_normalized_covariance(normalized_expert, means, dev = "cpu"):

    flattened_expert = torch.flatten(normalized_expert, start_dim = 1)
    flattened_means = torch.flatten(normalized_expert, start_dim = 1)

    matrix_size = (normalized_expert.shape[1] * normalized_expert.shape[2])

    flattened_covariance_matrices = torch.tensor(means.shape[0], matrix_size, matrix_size)

    k =means.shape[0]
    for i in range(k):
        diff = flattened_expert[:, :] - flattened_means[i, :]

        covariance_matrices[i] = torch.matmul(torch.t(diff), diff)/normalized_expert.shape[0]

    return flattened_covariance_matrices
        



def k_means_step(normalized_expert, k, means, dev = "cpu"):
    #data is points x horizon x ctrls
    #means is clusters x horizon x ctrls
    diff = normalized_expert[None, :, :, :] - means[:, None, :, :]

    #difference is now clusters x points x horizon x ctrls
    # dist is now clusters x points

    dist = torch.sum(diff * diff, dim=(2, 3))
    reclusters = torch.argmin(dist, dim = 0)

    new_means = torch.empty_like(means, device =dev)
    loss = 0
    for i in range(k):
        new_means[i] = torch.mean(normalized_expert[reclusters==i], dim = 0)
    
        loss += torch.sum(torch.where(reclusters==i,dist[i], torch.zeros_like(dist[i], device = torch.device('cuda:0'))))

    del diff
    del dist
    del reclusters

    return loss, new_means

def k_means_segment(expert, k=3, iterations = 200):
    #kplusplus init
    normalized_expert = torch.empty_like(expert, device = torch.device('cuda:0'))
    

    normalized_expert[:, :, 0] = (expert[:, :, 0] - 5)/5
    normalized_expert[:, :, 1] = (expert[:, :, 1] - 0)/3

    means = init_means(normalized_expert, k, dev = torch.device('cuda:0'))
    centers = torch.empty_like(means, device = torch.device('cuda:0'))
    
    for iter in range(iterations):
        loss, means = k_means_step(normalized_expert, k, means, dev = torch.device('cuda:0'))


    centers[:, :, 0] = (means[:, :, 0]* 5)+5
    centers[:, :, 1] = (means[:, :, 1]* 3)
    return centers, loss.item()
