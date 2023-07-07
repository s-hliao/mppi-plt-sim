import torch
import numpy as np
import kmeans

def init_params(normalized_expert, k, dev = "cpu"):
    # num_init = 1

    # means = torch.empty((k, normalized_expert.shape[1], normalized_expert.shape[2]), device = dev)

    # num_expert = normalized_expert.shape[0]
    # means[0] = normalized_expert[np.random.choice(num_expert), :, :]

    # means, assigments, loss = kmeans.k_means_segment(normalized_expert, k, 100)

    # print(means)

    # resp = torch.empty((k, normalized_expert.shape[0]), dtype = torch.float64, device = dev)
    # for i in range(k):
    #     resp[i, :] = torch.where(assigments==i, 1., 0.)

    # while num_init < k:
    #     diff = normalized_expert[None, :, :, :] - means[0:num_init, None, :, :]
    #     dist = torch.sum(diff * diff, dim=(2, 3))

    #     #dist is clusters x points
    #     closest_c_dist, min_indices = torch.min(dist, dim = 0)
    #     total_dist = torch.sum(closest_c_dist)
    #     means[num_init] = normalized_expert[np.random.choice(num_expert, p = closest_c_dist.cpu().numpy()/total_dist.item())]

    #     num_init+=1
    #     del diff
    #     del dist
    #     del closest_c_dist
    #     del min_indices
    #     del total_dist

    #     num_init+=1
    #     del diff
    #     del dist
    #     del closest_c_dist
    #     del min_indices
    #     del total_dist
    
    # sigma = compute_sigma(normalized_expert,k, means, True)
    # pi = torch.ones(k, device = dev)/k



    # return means, sigma, pi

    return resp

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
        #print(flat_means[i, :])
        diff = torch.empty_like(flat_expert[:, :], dtype =torch.float64, device = dev)
        diff = flat_expert[:, :] - flat_means[i, :]

    
        flat_covariance_matrices[i] = torch.matmul(torch.t(diff), diff)/ normalized_expert.shape[0]
        torch.set_printoptions(threshold=10_000)
        #print(flat_covariance_matrices[i])
        #print(torch.linalg.det(flat_covariance_matrices[i]*10))

        
        del diff
    del flat_expert
    del flat_means

    return flat_covariance_matrices

def compute_prob(normalized_expert, means, sigma, flatten= False, dev = "cpu"):
    if flatten:
        flat_expert, flat_means = flatten_params(normalized_expert, means)
    else:
        flat_expert = normalized_expert
        flat_means = means

    #print(normalized_expert.shape)
    #print(means.shape)
    #print(sigma.shape)

    n = flat_means.shape[0]

    d = torch.linalg.det(sigma).item()
    
    #print(sigma)
    #print(means)
    #print(np.linalg.det(sigma.cpu() *3))

    denom = np.power(2*np.pi, n/2)*np.power(d, 0.5)

    #print(denom)

    inv = torch.linalg.pinv(sigma)

    exponent_calc = torch.empty(flat_expert.shape[0], device = dev)

    diff = flat_expert-flat_means

    diffS = torch.matmul(diff, inv)
    exponent_calc = -.5 * torch.einsum("ij, ji->i",diffS, diff.t())
    
    
    ans = (1/denom) * torch.exp(exponent_calc)
    # print(denom)
    # print(diff)

    # print(exponent_calc)
    #print(ans)

    return ans


def E_step(normalized_expert, means, sigma, pi, k, dev = "cpu"):

    n = means.shape[1]
    m = normalized_expert.shape[0]
    # likelihood = torch.empty([k, m], device = dev)
    cluster_prob = torch.empty([k, m], device = dev)

    # print(normalized_expert.shape)
    # print(means.shape)

    # for cluster in range(k):
    #     likelihood[cluster,:] = torch.log(pi[cluster])+torch.log(
    #         compute_prob(normalized_expert, means[cluster], sigma[cluster], dev= dev))  
    # print(likelihood)
    # likelihood = likelihood-torch.max(likelihood, dim = 0).values[None,:]
    print("e step params")
    print(means)
    print(sigma)
    print(pi.sum())

    # recover = torch.exp(likelihood)
    for cluster in range(k):
        cluster_prob[cluster, :] = pi[cluster] * compute_prob(normalized_expert, means[cluster], sigma[cluster], dev= dev)
        print("cluster prob")
        print(compute_prob(normalized_expert, means[cluster], sigma[cluster], dev= dev))
    total = torch.sum(cluster_prob, dim= 0)
    resp = cluster_prob/total[None,:]

    return resp

def M_step(normalized_expert, resp, k, dev = "cpu"):

    m = normalized_expert.shape[0]
    n = normalized_expert.shape[1]

    new_mu = torch.empty([k, n], device = dev)
    new_sigma = torch.empty([k, n, n], device = dev)
    new_pi = torch.empty(k, device = dev)

    print("responsibility")
    
    print(resp)
    
    for cluster in range(k):
        cluster_resp= torch.sum(resp[cluster])
        print("cluster_resp")
        print(cluster_resp)

        new_pi[cluster] = cluster_resp/m
        new_mu[cluster] = torch.sum(resp[cluster,:][:, None]*normalized_expert, dim = 0) * 1/cluster_resp
        diff = normalized_expert-new_mu[cluster]
        #torch.set_printoptions(threshold=1000)

        new_sigma[cluster] = torch.matmul(resp[cluster,:].t() * diff.t(), diff)/cluster_resp
    return new_mu, new_sigma, new_pi


def loglikelihood(normalized_expert, means, sigma, pi, k, dev = "cpu"):

    m = normalized_expert.shape[0]
    total = torch.zeros([m], dtype = torch.float64, device = dev)
    
    for cluster in range(k):
        total += pi[cluster] * compute_prob(normalized_expert, means[cluster], sigma[cluster], dev = dev)
    print("likelihood")
    print(total)
    ll = torch.log(total)
    ans = torch.sum(ll)

    return ans

def run_model(expert_rollouts, k=3, iterations = 200):
    means, assigments, loss = kmeans.k_means_segment(expert_rollouts, k, 100)

    resp = torch.zeros((k, expert_rollouts.shape[0]), dtype = torch.float32, device = torch.device('cuda:0'))
    for i in range(k):
        resp[i, :] = torch.where(assigments==i, 1., 0.)


    normalized_expert = expert_rollouts
    normalized_expert = torch.empty_like(expert_rollouts, device = torch.device('cuda:0'))
    

    normalized_expert[:, :, 0] = (expert_rollouts[:, :, 0] - 10)
    normalized_expert[:, :, 1] = (expert_rollouts[:, :, 1] - 0) 

    flat_expert, flat_means = flatten_params(normalized_expert, means)

    #print(means.shape)

    for iter in range(iterations):
        flat_means, sigma, pi = M_step(flat_expert, resp, k, torch.device('cuda:0'))
        resp = E_step(flat_expert, flat_means, sigma, pi, k, torch.device('cuda:0'))

        ll = loglikelihood(flat_expert, flat_means, sigma, pi, k, torch.device('cuda:0'))

        
        print (iter, ll)
    

    horizon_length = expert_rollouts.shape[1]

    means[:, :, 0] = (flat_means[:, :horizon_length])*5+5
    means[:, :, 1] = (flat_means[:, horizon_length:])*4+0

    sigma[:, :horizon_length, :] *= 5
    sigma[:, horizon_length:, :] *= 4

    sigma[:, :, :horizon_length] *= 5
    sigma[:, :, horizon_length:] *= 4

    #print(means.shape)

    return means, sigma, pi, ll


