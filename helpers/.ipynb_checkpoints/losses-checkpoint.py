import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


def my_modified_scarf_loss1(z1, z2, temperature=1, debug=False):
    ## scarf loss that has no positive sample in the denominator
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    
    z = torch.exp(sim/temperature)
    debug and print(z.shape)
        
        
    num_mask = torch.eye(z.shape[0]) # diagonal mask
    den_mask = torch.ones_like(z)
    if device>=0:
        num_mask = num_mask.to(device)
        den_mask = den_mask.to(device)
    den_mask = den_mask - num_mask 

    
    numerator = z.mul(num_mask).sum(dim=1)
    denominator = z.mul(den_mask).sum(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss

def get_lowest_mask(sim, n_low = 16, debug=False):
    # for scarf similarity matrix only
    device = sim.device
    
    tmp = torch.clone(sim)
    debug and print(tmp.shape)
    # # make diagonals inf to remove positive pairs from sort
    # j = torch.arange(tmp.shape[0])
    # tmp[j,j] = torch.inf
    # debug and print(tmp)
    
    # # no point in sorting if there are less elements
    # # this happens at the last mini-batch
    if n_low > sim.shape[0]:
        debug and print(f'low! {n_low} vs {sim.shape[0]}; returning all except diagonal')
        return torch.ones_like(sim).to(device) - torch.eye(sim.shape[0]).to(device)
    
    low_cols = (tmp.argsort()[:, :n_low]).reshape(-1, 1)
    mask2 = torch.zeros_like(tmp)
    low_rows = torch.arange(tmp.shape[0]).reshape(-1, 1).repeat(1, n_low)
    
    debug and print(low_rows.view(-1).shape, low_cols.view(-1).shape)
    
    mask2[low_rows.view(-1), low_cols.view(-1)] = 1

    return mask2

def my_modified_scarf_loss2(z1, z2, temperature=1, n_low=16, debug=False):
    ## scarf loss that has no positive sample in the denominator
    ## and limits the number of negative pairs in the denominator
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    sim_scaled = sim/temperature
    
    z = torch.exp(sim_scaled)
    debug and print(z.shape)
    
    num_mask = torch.eye(z.shape[0]) # diagonal mask
    den_mask = torch.ones_like(z)
    if device>=0:
        num_mask = num_mask.to(device)
        den_mask = den_mask.to(device)
    den_mask = den_mask - num_mask 
    
    den_mask = get_lowest_mask(sim_scaled, n_low=n_low).to(device)

    
    numerator = z.mul(num_mask).sum(dim=1)
    denominator = z.mul(den_mask).sum(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss

def my_modified_scarf_loss2a(z1, z2, temperature=1, debug=False):
    ## scarf loss that has no positive sample in the denominator
    ## and limits to 1 random negative pairs in the denominator
    device = z1.get_device()
    n_samples = z1.shape[0]
    b_idx = torch.concat([
        torch.arange(n_samples).reshape(-1, 1), 
        torch.randint(n_samples, (n_samples,1))
    ], dim=1).reshape(-1)   
    
    a = z1.repeat(1,2).view(-1, z1.shape[1]) 
    b = z2[b_idx]
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    debug and print(a.shape, b.shape)
    sim = F.cosine_similarity(a, b).view(-1, 2)
    debug and print(sim)
    
    
    z = torch.exp(sim/temperature)
    debug and print(z)
    
    # 1st column div by total row
    numerator = z[:, 0]
    denominator = z.sum(dim=1)
    # denominator = z[:, 1]
    
    debug and print('num', numerator)
    debug and print('den', denominator)
    
    losses = numerator / denominator
    debug and print('loss', losses)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses)
    
    loss = log_losses.mean()
    
    return loss


def my_modified_scarf_loss3(z1, z2, temperature=1, debug=False):
    ## scarf loss that has no positive sample in the denominator
    ## and limits the number of negative pairs in the denominator
    ## and increases positive pairs in numerator
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    sim_scaled = sim/temperature
    
    z = torch.exp(sim_scaled)
    debug and print(z.shape)
    
    num_mask = torch.eye(z.shape[0]) # diagonal mask
    den_mask = torch.ones_like(z)
    if device>=0:
        num_mask = num_mask.to(device)
        den_mask = den_mask.to(device)
    den_mask = den_mask - num_mask 
    
    num_mask1 = get_lowest_mask(-sim_scaled, 7).to(device)
    num_mask = num_mask + num_mask1
    
    debug and print(num_mask.sum(dim=1))

    
    numerator = z.mul(num_mask).sum(dim=1)
    denominator = z.mul(den_mask).sum(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss


def my_modified_scarf_loss4(z1, z2, temperature=1, debug=False):
    ## scarf loss that has no positive sample in the denominator
    ## and limits the number of negative pairs in the denominator
    ## and increases positive pairs in numerator
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    sim_scaled = sim/temperature
    
    z = torch.exp(sim_scaled)
    debug and print(z.shape)
    
    num_mask = torch.eye(z.shape[0]) # diagonal mask
    den_mask = torch.ones_like(z)
    if device>=0:
        num_mask = num_mask.to(device)
        den_mask = den_mask.to(device)
    den_mask = den_mask - num_mask 
    
    num_mask1 = get_lowest_mask(-sim_scaled, 7).to(device)
    num_mask = num_mask + num_mask1
    
    den_mask = get_lowest_mask(sim_scaled).to(device)
    
    debug and print(num_mask.sum(dim=1))

    
    numerator = z.mul(num_mask).sum(dim=1)
    denominator = z.mul(den_mask).sum(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss


def my_bad_scarf_loss(z1, z2, temperature=1, debug=False):
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    
    z = torch.exp(sim/temperature)
    debug and print(z.shape)
    
    # get only diagonals
    i = torch.arange(z.shape[0])
    numerator = z[i,i]
    denominator = z.mean(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss



def my_scarf_loss(z1, z2, temperature=1, debug=False):
    # implementation of line 8 of algorithm 1 in Bahri et al.
    device = z1.get_device()
    
    a = z1.repeat(1,z1.shape[0]).view(-1, z1.shape[1]) 
    b = z2.repeat(z2.shape[0],1)
    
    # move to gpu if z1 is in gpu
    if device>=0:
        a = a.to(device)
        b = b.to(device)
    
    sim = F.cosine_similarity(a, b).view(-1, z1.shape[0])
    
    z = torch.exp(sim/temperature)
    debug and print(z.shape)
    
    # get only diagonals
    i = torch.arange(z.shape[0])
    numerator = z[i,i]
    denominator = z.sum(dim=1)
    
    debug and print('num', numerator.shape)
    debug and print('den', denominator.shape)
    
    losses = numerator / denominator
    debug and print('loss', losses.shape)
    
    log_losses = -torch.log(losses)
    debug and print('log_loss', log_losses.shape)
    
    loss = log_losses.mean()
    
    return loss


def _loss_term(exp_sim, i, j, temperature=1):
    device = exp_sim.get_device()
    
    n_mask = torch.zeros(exp_sim.shape)
    d_mask = torch.ones(exp_sim.shape)
    
    if device>=0:
        n_mask = n_mask.to(device)
        d_mask = d_mask.to(device)
    
    # code above looks common n can be moved up the call stack

    n_mask[i,j] = 1
    d_mask[i,i] = 0

    n_sim = torch.mul(exp_sim, n_mask) / temperature
    d_sim = torch.mul(exp_sim, d_mask) / temperature
    
    n_sim = n_sim[i,:]
    d_sim = d_sim[i,:]
    
    # expanding to trace anomalies
    numerator = n_sim.sum(1)
    denominator = d_sim.sum(1)
    divided = numerator.div_(denominator)
    log_divided = torch.log(divided)
    sum_log_divided = log_divided.sum()
    
    loss_term = -sum_log_divided

    return loss_term


def _get_loss(sim, i,j, temperature=1):
    exp_sim = torch.exp(sim)
    
    return _loss_term(exp_sim, i, j, temperature) + _loss_term(exp_sim, j, i, temperature)

def _sim_2(z):
    z = F.normalize(z)
    # a1, b1 = torch.split(z, n)
    sim_fast = torch.mm(z,z.T)
    sim_fast = sim_fast.cpu()
    return sim_fast

def _sim_1(z):
    # manually make similarity matrix
    two_n, n_embedding = z.shape
    sim2 = torch.zeros(two_n, two_n)
    sim2_debug = []
    for i in range(two_n):
        row = []
        for j in range(two_n):
            u, v = z[i], z[j]
            u_norm = torch.linalg.norm(u, ord=2)
            v_norm = torch.linalg.norm(v, ord=2)
            
            # print(u,v, u_norm, v_norm, torch.dot(u,v))
            
            sim2[i,j] = torch.dot(u,v) / (u_norm * v_norm)
            # row.append(f'{i},{j}')
            
        # sim2_debug.append(row)
            
    # print(sim)
    # print(sim2_debug)
    return sim2
    
def _sim_0(z):
    two_n, n_embedding = z.shape
    # repeat on both axis to prepare for cos-sim computation
    a1=z.repeat(1, two_n).view(-1, n_embedding).contiguous()
    b1=z.repeat(two_n, 1).contiguous()

    sim = F.cosine_similarity(a1, b1).view(two_n, two_n).contiguous()
    return sim

# ref: simclr loss
def my_info_nce(z1,z2, temperature=1, method=0):
    ## z1 and z2 are embeddings of two batches X1 and X2
    ## where X1 and X2 are created by two transformation T1, T2 
    # before they inputted to the shared encoder
    a, b = z1, z2
   
    n, n_embedding = a.shape
    
    # concat two embedding such that 
    # index i of A is index 2*i-1 after concat
    # index i of B is index 2*i after concat
    z = torch.cat((a,b), dim=1).view(-1, n_embedding).contiguous()
    two_n, _ = z.shape
    
    sim = _sim_0(z) if method==0 else _sim_1(z) if method==1 else _sim_2(z)
    
    loss = 0
    
    i = torch.arange(0,n)
    
    loss2 = _get_loss(sim, 2*i, 2*i+1, temperature)
    
    # print(loss/two_n, loss2/two_n)
    loss = loss2
    
    return loss / (2*n)


def one_sided_infonce(z1, z2, temperature=1, method=0):

    a, b = z1, z2
    n, n_embedding = a.shape
    z = torch.cat((a,b), dim=1).view(-1, n_embedding).contiguous()
    two_n, _ = z.shape
    
    sim = _sim_0(z) if method==0 else _sim_1(z) if method==1 else _sim_2(z)
    
    loss = 0
    
    i = torch.arange(0,n)
    
    exp_sim = torch.exp(sim)
    # get loss(x, x-hat) only
    # ie. dont do loss(x-hat, x)
    loss = _loss_term(exp_sim, 2*i, 2*i+1, temperature) 
    
    return loss / (n)

def _get_cluster_loss(sim, i, j, y, temperature=1):
    exp_sim = torch.exp(sim)

    # i_prime contains sample index draw at random that belongs to the same cluster as i
    i_prime = torch.zeros_like(i)
    
    for c in torch.unique(y):
        # get sample index of same cluster
        idx = (y == c).nonzero(as_tuple=True)[0]
        # get a random sample index from same cluster
        r = torch.randint(idx.shape[0],(idx.shape[0],))
        r_idx = idx[r]
        # we map N sample to 2*N sample index to match sim matrix size
        i_prime[idx] = r_idx*2

    # j_prime is the corrupted version of i_prime
    # since all corrupted sample are the next index (odd), we add 1
    j_prime = i_prime+1

    # only one pair is used because sim(i,j) == sim(j,i)
    # i and i-hat + i and cluster-member + i and cluster-member-hat + i-hat and cluster-member-hat
    loss = _loss_term(exp_sim, i, j, temperature) + _loss_term(exp_sim, i, i_prime, temperature) \
        + _loss_term(exp_sim, i, j_prime, temperature) + _loss_term(exp_sim, i_prime, j_prime, temperature)
    return loss / 4

def my_cluster_info_nce(z1,z2, y, temperature=1, method=0):
    ## z1 and z2 are embeddings of two batches X1 and X2
    ## where X1 and X2 are created by two transformation T1, T2 
    # before they inputted to the shared encoder
    a, b = z1, z2
    
    # 2i+1 is the corrupted version of 2i
    # so they are also same cluster.
    # reshape y to same size as sim matrix length
    # y_prime = y.reshape([-1, 1]).repeat((1,2)).reshape(-1)
   
    n, n_embedding = a.shape
    
    # concat two embedding such that 
    # index i of A is index 2*i-1 after concat
    # index i of B is index 2*i after concat
    z = torch.cat((a,b), dim=1).view(-1, n_embedding).contiguous()
    two_n, _ = z.shape
    
    sim = _sim_0(z) if method==0 else _sim_1(z) if method==1 else _sim_2(z)
    
    loss = 0
    
    i = torch.arange(0,n)
    
    loss2 = _get_cluster_loss(sim, 2*i, 2*i+1, y, temperature)
    
    # print(loss/two_n, loss2/two_n)
    loss = loss2
    
    return loss / (n) # because we have get_cluster_loss already divides by 3


def _case2_loss_term(exp_sim, i, j, debug=False):
    n_mask = torch.zeros_like(exp_sim)
    d_mask = torch.ones_like(exp_sim)

    n_mask[i,j] = 1 # set positive pairs
    d_mask[i,i] = 0 # remove diagonals
    d_mask[i,j] = 0 # remove positive pairs
    
    debug and print(n_mask)
    debug and print(d_mask)

    n_sim = torch.mul(exp_sim, n_mask) 
    d_sim = torch.mul(exp_sim, d_mask)
    

    
    n_sim = n_sim[i,:]
    d_sim = d_sim[i,:]
    
    # expanding to trace anomalies
    numerator = n_sim.sum(1)
    denominator = d_sim.sum(1)
    divided = numerator.div_(denominator)
    log_divided = torch.log(divided)
    sum_log_divided = log_divided.sum()
    
    loss_term = -sum_log_divided
    
    return loss_term
    
def case2_cluster_info_nce(z1, z2, y, temperature=1, method=0):
    # since we are not producing two version
    # we do not need both z1 and z2
    # however, we take z2 as parameter to make it work with existing trainer_cluster_v3 code
    z = z1    
    n, n_embedding = z.shape
    
    sim = _sim_0(z) if method==0 else _sim_1(z) if method==1 else _sim_2(z)
    
    exp_sim = torch.exp(sim)/temperature
    
    # i = torch.arange(0,n)
    # j = torch.zeros_like(i)
    
    debug = torch.unique(y).shape[0]==1
    
    n_mask = torch.zeros_like(exp_sim)
    d_mask = torch.zeros_like(exp_sim)
    
    # print('z shape', exp_sim.shape)
    # print('n_mask shape', n_mask.shape)

    for c in torch.unique(y):
        # get sample indices that are in same cluster as c
        idx = (y == c).nonzero(as_tuple=True)[0]
        # select a random value from idx for each index
        r = torch.randint(idx.shape[0],(idx.shape[0],))
        r_idx = idx[r]

        # create positive pair mask
        n_mask[idx, r_idx] = 1

        # sample idx from other clusters
        reverse_idx = (y != c).nonzero(as_tuple=True)[0]
        # repeat and reshape to create indices for mask
        i = idx.reshape(-1, 1).repeat(1, reverse_idx.shape[0]).reshape(-1)
        j = reverse_idx.reshape(1, -1).repeat(1, idx.shape[0]).reshape(-1)
        # set 1 for pairs from diff clusters
        
        # print('min & max i', i.min(), i.max())
        # print('min & max j', j.min(), j.max())
        
        d_mask[i, j] = 1
        
    # create numerator and denominator matrices
    n_sim = torch.mul(exp_sim, n_mask) 
    d_sim = torch.mul(exp_sim, d_mask)
    
    # debug and print(n_mask.sum(1), d_mask.sum(1))
    
    # expanding to trace anomalies
    numerator = n_sim.sum(1)
    denominator = d_sim.sum(1)
    divided = numerator.div_(denominator)
    log_divided = torch.log(divided)
    sum_log_divided = log_divided.sum()
    
    loss_term = -sum_log_divided
    
    loss = loss_term # repeat above and add terms for case 3
    
    debug and print(loss)
    
    return loss / n


## copied from pytorch-scarf github for sanity checks
class NTXent(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        
        # print(similarity)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(similarity.get_device())
        # print(mask)
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
    
    
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def barlow_twins(z1, z2, lambd = 0.0051):
    # print(z1, z2)
    sizes = z1.shape
    assert z1.shape == z2.shape, 'embeddings not equal size'
    
    bn = torch.nn.BatchNorm1d(sizes[-1], affine=False, device=z1.device)
    # empirical cross-correlation matrix
    c = bn(z1).T @ bn(z2)
    # print(c.shape)

    batch_size = sizes[0]
    # sum the cross-correlation matrix between all gpus
    c.div_(batch_size)
    # torch.distributed.all_reduce(c)
    # turned off because there is single matrix in single gpu env
    

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss


def barlow_modified(z1, z2, lambd = 0.0051):
    # print(z1, z2)
    sizes = z1.shape
    assert z1.shape == z2.shape, 'embeddings not equal size'
    
    bn = torch.nn.BatchNorm1d(sizes[-1], affine=False, device=z1.device)
    # empirical cross-correlation matrix
    c = bn(z1).T @ bn(z2)
    # print(c.shape)

    batch_size = sizes[0]
    # sum the cross-correlation matrix between all gpus
    c.div_(batch_size)
    # torch.distributed.all_reduce(c)
    # turned off because there is single matrix in single gpu env
    

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).add_(-0.5).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss

pretrain_losses = {
    'simclr': my_info_nce,
    'cluster-simclr': my_cluster_info_nce,
    'cluster-simclr-case2': case2_cluster_info_nce,
    'barlow': barlow_twins,
    'barlow_modified': barlow_modified,
    'scarf': my_scarf_loss,
    'bad-scarf': my_bad_scarf_loss,
    'modified-scarf-1': my_modified_scarf_loss1,
    'modified-scarf-2': my_modified_scarf_loss2,
    'modified-scarf-3': my_modified_scarf_loss3,
    'modified-scarf-4': my_modified_scarf_loss4,
    'modified-scarf-2a': my_modified_scarf_loss2a,
    'one-sided-simclr': one_sided_infonce,
}