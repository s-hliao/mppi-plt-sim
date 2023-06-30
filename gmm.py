import torch
import numpy as np

def init_params(normalized_expert, k, dev = "cpu"):
    num_init = 1

    means = torch.empty((k, normalized_expert.shape[1], normalized_expert.shape[2]), device = dev)

    num_expert = normalized_expert.shape[0]
    means[0] = normalized_expert[np.random.choice(num_expert), :, :]

    while num_init < k:
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

def flatten_params(normalized_expert, means):
    flat_expert = torch.flatten(torch.swapaxes(normalized_expert, 1, 2), start_dim = 1)
    flat_means = torch.flatten(torch.swapaxes(means, 1, 2), start_dim = 1)

    return flat_expert, flat_means

def compute_sigma(normalized_expert, k, means, flatten = False, dev = "cpu"):

    if flatten_params:
        flat_expert, flat_means = flatten_params(normalized_expert, means)
    else:
        flat_expert = normalized_expert
        flat_means = means

    matrix_size = (normalized_expert.shape[1] * normalized_expert.shape[2])

    flat_covariance_matrices = torch.empty((means.shape[0], matrix_size, matrix_size), 
        dtype = normalized_expert.dtype, device = dev)

    for i in range(k):
        diff = flat_expert[:, :] - flat_means[i, :]
        flat_covariance_matrices[i] = torch.matmul(torch.t(diff), diff)/normalized_expert.shape[0]

        del diff
    del flat_expert
    del flat_means

    return flat_covariance_matrices

def compute_prob(normalized_expert, means, sigma, flatten= False):
    if flatten:
        flat_expert, flat_means = flatten_params(normalized_expert, means)
    else:
        flat_expert = normalized_expert
        flat_means = means

    n = flat_means.shape[0]
    d = torch.linalg.det(sigma).item()
    denom = np.power(2*np.pi, n/2)*np.power(d, 0.5)

    inv = torch.linalg.pinv(sigma)

    exponent_calc = torch.empty(flat_expert.shape[0])

    diff = flat_expert-flat_means
    diffS = torch.matmul(diff, inv)
    epower = -.5 * torch.einsum("ij, ji->i",diffS, torch.t(diff))

    ans = (1/denom) * torch.exp(exponent_calc)

    return ans


def E_step(normalized_expert, means, sigma, pi, k, dev = "cpu"):

    n = means.shape[1]
    m = normalized_expert.shape[0]
    likelihood = torch.empty([k, m], device = dev)
    
    for cluster in range(k):
        likelihood[cluster,:] = torch.log(pi[cluster])+ torch.log(prob(X, means[cluster], sigma[cluster]))

    likelihood = likelihood-torch.max(likelihood, dim = 0)[None,:]
    recover = torch.exp(likelihood)
    total = torch.sum(recover, dim= 0)
    resp = recover/total[None,:]

    return resp

def M_step(normalized_expert, resp, k, dev = "cpu"):

    m = normalized_expert.shape[0]
    n = normalized_expert.shape[1]

    new_mu = torch.empty([k, n], device = dev)
    new_sigma = torch.empty([k, n, n], device = dev)
    new_pi = torch.empty(k, device = dev)

    for cluster in range(k):
        cluster_resp= torch.sum(resp[cluster])

        new_pi[cluster] = cluster_resp/m
        new_mu[cluster] = torch.sum(resp[cluster,:][:, None]*X, dim = 0) * 1/cluster_resp
        diff = normalized_expert-new_mu[cluster]
        new_sigma[cluster] = np.matmul(resp[cluster,:].T * diff.T, diff)/cluster_resp

    return new_mu, new_sigma, new_pi


def loglikelihood(normalized_expert, means, sigma, pi, k, dev = "cpu"):

    m = normalized_expert.shape[0]
    total = torch.zeros([m], device = dev)
    for cluster in range(k):
        total += pi[cluster] * prob(normalized_expert, mu[cluster], sigma[cluster])

    ll = np.log(total)
    ans = np.sum(ll)

    return ans

def run_model(expert_rollouts, k=3, iterations = 200):
    normalized_expert = torch.empty_like(expert, device = torch.device('cuda:0'))
    

    normalized_expert[:, :, 0] = (expert[:, :, 0] - 5)/5
    normalized_expert[:, :, 1] = (expert[:, :, 1] - 0)/4

    means, sigma, pi = init_params(normalized_expert, k, dev = torch.device('cuda:0'))
    centers = torch.empty_like(means, device = torch.device('cuda:0'))

    resp = None

    for iter in range interations():
        resp = E_step(normalized_expert, means, sigma, pi, k, torch.device('cuda:0'))
        means, sigma, pi = M_step(normalized_expert, resp, k)

    ll = loglikelihood(normalized_expert, means, sigma, pi, k, torch.device('cuda:0'))


