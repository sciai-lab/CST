import numpy as np
from sklearn.metrics import pairwise_distances
from numba import njit
import scipy.sparse  as sp
from .utils_terminal_side_edge import check_validity_edge_restriction
def choose_merging_neighBP(T_dict, coords, BP,edge_flows, alpha=1, merging_criterium='closest'):
    '''

    :param T_nx:
    :param coords:
    :param BP:
    :param alpha:
    :param merging_criterium:
        - closest (default) -> merges to closest neighbor
        - minflowcost -> merges to the neighbor connected by the edge with lowest cost*flow**alpha
        - tryall -> tries all possibilities and chooses the one with lowest cost given the current BP positions
    :return:
    '''
    neighbors_BP = T_dict[BP]


    if merging_criterium.lower()== 'closest':
        return choice_merge_closest(coords,neighbors_BP,BP)
    elif merging_criterium.lower()== 'minflowcost':
        return choice_merge_minflowcost(coords,neighbors_BP,BP,edge_flows,)
    elif merging_criterium== 'tryall':
        return choice_merge_tryall(coords, neighbors_BP, BP,edge_flows, alpha)
    else:
        raise NotImplementedError('The merging policy of the BP "%s" in the removal process is not implemented' % merging_criterium)

def choice_merge_closest(coords,neighbors_BP,BP):
    # compute distances between neighbors
    D2BP = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1)
    # compute idx closest neighbor
    merging_neighBP = neighbors_BP[np.argmin(D2BP)]
    return merging_neighBP

def choice_merge_minflowcost(coords,neighbors_BP,BP,edge_flows,alpha=1):
    # compute distances between neighbors and multiply with the flow**alpha
    D2BPflow = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1) * np.array(
        [edge_flows[(BP,neigh)] for neigh in neighbors_BP]) ** alpha
    # compute idx neighbor
    merging_neighBP = neighbors_BP[np.argmin(D2BPflow)]
    return merging_neighBP



def choice_merge_tryall(coords,neighbors_BP,BP,edge_flows,alpha):
    '''try all possible collapse possibilities and take the one that minimizes the cost'''
    if len(neighbors_BP)>50:
        D=pairwise_distances(coords[neighbors_BP])
        flows=np.array([edge_flows[(neighbor,BP)] for neighbor in neighbors_BP])**alpha
        flows.repeat(D.shape[0]).reshape((-1, D.shape[0])).T
        merging_neighBP=neighbors_BP[np.argmin((D*flows).sum(1))]
        return merging_neighBP
    else:
        flows = np.array([edge_flows[(neighbor, BP)] for neighbor in neighbors_BP])
        return core_merge_tryall_njit_with_dist(coords, neighbors_BP, flows, alpha=alpha)
@njit
def compute_pairwise_distances_njit(coords, neighbors_BP):
    n = len(neighbors_BP)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = np.sqrt(np.sum((coords[neighbors_BP[i]] - coords[neighbors_BP[j]]) ** 2))
            D[j, i] = D[i, j]
    return D
@njit
def core_merge_tryall_njit_with_dist(coords, neighbors_BP, flows, alpha):
    D = compute_pairwise_distances_njit(coords, neighbors_BP)
    if alpha != 1:
        flows = flows ** alpha
    flows = flows.repeat(D.shape[0]).reshape((-1, D.shape[0])).T
    merging_neighBP = neighbors_BP[np.argmin((D * flows).sum(1))]
    return merging_neighBP




