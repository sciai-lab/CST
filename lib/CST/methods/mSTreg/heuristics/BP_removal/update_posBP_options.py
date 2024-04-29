import numpy as np
from numba import njit

def BP_position_update_for_removal(coords, T_dict, neigh_BP_merged, num_terminals,
                                   edge_flows,alpha=1, criterium_BP_position_update='median'):
    '''

    :param coords: coords nodes
    :param T_nx: networkx graph
    :param BP: BP that was merged
    :param neigh_BP_merged: node to which BP was merged
    :param num_terminals: number terminals
    :param alpha: alpha value cost default =1
    :param criterium_BP_position_update:
        - median:
        - no_update
    :return:
    '''
    if criterium_BP_position_update.lower() =='no_update':
        return coords
    elif criterium_BP_position_update.lower()== 'median':
        coords=update_BPpos_median(coords,T_dict,neigh_BP_merged,num_terminals,edge_flows,alpha)
    else:
        raise NotImplementedError("position update %s in the BP removal process not implemented" % (criterium_BP_position_update))

    return coords
def update_BPpos_median(coords,T_dict,neigh_BP_merged,num_terminals,edge_flows,alpha):
    '''
    Updates position of the BP that was formed after the collapse (here referred as neigh_BP_merged), keeping its neighbors fixed.
    The new position is the weighted median of the neighbors (weights are given by the flows).
    :param coords:
    :param T_nx:
    :param neigh_BP_merged:
    :param num_terminals:
    :param alpha:
    :return:
    '''


    if neigh_BP_merged >= num_terminals:
        neigh_and_flows=np.array([[v,edge_flows[(neigh_BP_merged,v)] ** alpha] for v in T_dict[neigh_BP_merged]])
        return median_update(coords,neigh_BP_merged,neigh_and_flows[:,0].astype(int),neigh_and_flows[:,1])

    return coords

@njit
def median_update(coords: np.ndarray,neigh_BP_merged: int,neighs:np.ndarray,flows:np.ndarray):
    diff = 1
    while_it = 0
    eps=1e-15
    while diff > 1e-5 and while_it < 500:
        old = coords[neigh_BP_merged]
        # compute new coordinate merged point
        sumand_denominator = flows / (np.sqrt(np.sum((coords[neighs] - coords[neigh_BP_merged])**2, axis=1)) + eps)
        numerator = np.sum(coords[neighs] * sumand_denominator[:, np.newaxis], axis=0)
        denominator = np.sum(sumand_denominator)

        coords[neigh_BP_merged] = numerator / denominator

        diff = np.linalg.norm(old - coords[neigh_BP_merged])
        while_it += 1
    return coords

