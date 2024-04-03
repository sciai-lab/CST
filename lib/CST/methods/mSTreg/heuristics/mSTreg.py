import numpy as np
import warnings

from ..topology import topology
from ..geometry_optimization import fast_optimize
from scipy import sparse as sp
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist
from ..topology.topology import adj_to_adj_sparse
from ..topology.prior_topology import get_shortest_path_tree

from ....T_datacls.utilities.graphtools import ensure_connected_knn_graph, Wiener_index
from ..heuristics.BP_removal.BP_remove import removeBP
from ..heuristics.BP_removal.utils_terminal_side_edge import constrain2edge_index

import ctypes
from numpy import ctypeslib as npct
import os


file_path = os.path.abspath(__file__)
# Extract the directory path
directory_file = os.path.dirname(file_path)


EWBCST_lib = npct.load_library("fast_optimizer", "%s/../geometry_optimization/lib"%directory_file)

# define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

EWBCST_lib.calculate_EW_BCST.argtypes = [ctypes.c_int, array_2d_double, array_2d_int, array_1d_double, ctypes.c_double]



# opt_lib.calculate_EW_CBOT.restype = ctypes.c_double


def compute_BCST(topo, alpha, coords_terminals, maxiter_mSTreg=10,
                 maxfreq_mSTreg=10, return_CST=False,
                 order_criterium='closestterminals', merging_criterium='tryall',
                 criterium_BP_position_update='median',
                 Compute_CST_each_iter=True, demands=None, init_BP_coords=None, verbose=False,
                 mST_fromknn=True, ori_input_graph=None, policy_constraint='shortest_path',beta=1,
                 factor_terminal=1):
    """
       Compute the BCST  given a topology and terminal coordinates.

       Parameters:
           param: topo (topology class or np.ndarray): The topology object or adjacency matrix.
           If it is a np.ndarray, it is assumed to have size n-2 x 3 where n is the number of terminals. Each row indicates
           the three neighbors of a branching point.
           param: alpha (float): BCST parameter.
           param: coords_terminals (np.ndarray): The coordinates of the terminal nodes.
           param: maxiter_mSTreg (int): The maximum number of iterations for mST regularization. Default is 10.
           param: maxfreq_mSTreg (int): The maximum number of nodes added per edge for the mST regularization. Default is 10.
           param: return_CST (bool): Whether to return the CST. Default is False.
           param: order_criterium (str): The criterion for ordering branch points in the mapping from a full tree topology to a CST topology.
            Default is 'closestterminals'.
           param: merging_criterium (str): The criterion for merging branch points in the mapping from a full tree topology to
            a CST topology.. Default is 'tryall'.
           param: criterium_BP_position_update (str): The criterion for updating branch point positions in the mapping from a
            full tree topology to a CST topology. Default is 'median'.
           param: Compute_CST_each_iter (bool): Whether to compute the CST tree at each iteration of the mSTreg.
            If False,only computed at the end. Default is True.
           param: demands (None or np.ndarray): The demand matrix for CST optimization. Default is None.
           param: init_BP_coords (None or np.ndarray): The initial branch point coordinates for CST optimization. Default is None.
           param: mST_fromknn (bool): Whether to approximate the mST from a kNN graph. Default is True.
           param: ori_input_graph: Adjacency sparse matrix over terminals with the entries representing the euclidean distance
            between terminal nodes. If given it constraints the CST solutions to be spanning trees of this graph. In the
             BCST case, it constrains the full tree topologies to be derived from spanning trees of the original graph.
           param: policy_constraint (str): The mode of the constraint if ori_input_graph is give. Default is 'shortest_path'.

       Returns:
           If return_CST is False:
               topology: The BCST tree.
               float: The cost of the BCST tree.
               np.ndarray: The coordinates of the branch points in the BCST tree.
               np.ndarray: The edge weights of the BCST tree.
           If return_CST is True
               topology: The BCST tree.
               topology: The CST tree.
               float: The cost of the BCST tree.
               np.ndarray: The coordinates of the branch points in the BCST tree.
               np.ndarray: The edge flows (centralities) of the BCST tree.
       """

    if isinstance(topo, topology):
        adj = topo.adj
    elif isinstance(topo, np.ndarray):
        adj = topo

    num_terminals = len(coords_terminals)


    # Solve geometric optimization problem of the BPs
    best_topo_BCST, best_cost_BCST, best_coords, best_EW, _ = fast_optimize(adj, coords_terminals=coords_terminals, al=alpha,
                                                                  demands=demands, init_BP_coords=init_BP_coords,beta=beta,
                                                                            factor_terminal=factor_terminal)

    if return_CST and Compute_CST_each_iter:
        # Map the BCST to a CST
        if num_terminals>=100000:
            T_dict, filtered_coords, flows_dict = remove_collapsedBP_from_solution(adj=best_topo_BCST, flows=best_EW,
                                                                                   coords=best_coords,return_Tdict=True)
            best_topo_CST = removeBP(T=T_dict,edge_flows=flows_dict,num_terminals=num_terminals, coords=filtered_coords,
                                     order_criterium=order_criterium, merging_criterium=merging_criterium,
                                     criterium_BP_position_update=criterium_BP_position_update, alpha=alpha,
                                     ori_input_graph=ori_input_graph)  # removes BP from solution-> tree on original coordinates
        else:
            best_topo_CST = removeBP(adj_to_adj_sparse(best_topo_BCST, flows=best_EW), coords=best_coords,
                                       order_criterium=order_criterium, merging_criterium=merging_criterium,
                                       criterium_BP_position_update=criterium_BP_position_update, alpha=alpha,
                                     ori_input_graph=ori_input_graph)  # removes BP from solution-> tree on original coordinates

        #compute cost of CST
        best_cost_CST = Wiener_index(best_topo_CST, alpha=alpha) / (num_terminals ** (2 * alpha))

    improv = True
    niter = 0
    coords = best_coords
    topo_BCST = adj
    bestiter_mstreg_BCST = 0
    bestiter_mstreg_CST = 0
    #Iterations mST regularization
    while improv or niter < maxiter_mSTreg:
        if niter > maxiter_mSTreg:
            break
        improv = False
        #apply mST regularization
        newadj = topo_mST_reguralization(coords, adj=topo_BCST, max_freq=maxfreq_mSTreg, mST_fromknn=mST_fromknn,
                                         ori_input_graph=ori_input_graph, policy_constraint=policy_constraint)
        topo_BCST, cost_BCST, coords, EW, _ = fast_optimize(newadj, coords_terminals=coords_terminals, al=alpha,
                                                                  demands=demands, init_BP_coords=init_BP_coords, EW=None,
                                                            beta=beta,factor_terminal=factor_terminal)
        if cost_BCST < best_cost_BCST:
            #store best BCST solution
            best_topo_BCST, best_cost_BCST, best_coords, best_EW = topo_BCST, cost_BCST, coords, EW
            improv = True
            bestiter_mstreg_BCST = niter + 1

        if return_CST and Compute_CST_each_iter:
            # Map the BCST to a CST
            if num_terminals >= 100000:
                T_dict, filtered_coords, flows_dict = remove_collapsedBP_from_solution(adj=topo_BCST, flows=EW,
                                                                                       coords=coords,return_Tdict=True)
                topo_CST = removeBP(T=T_dict, edge_flows=flows_dict, num_terminals=num_terminals,
                                         coords=filtered_coords,
                                      order_criterium=order_criterium, merging_criterium=merging_criterium,
                                      criterium_BP_position_update=criterium_BP_position_update,
                                    alpha=alpha,ori_input_graph=ori_input_graph,policy_constraint=policy_constraint)
            else:
                topo_CST = removeBP(adj_to_adj_sparse(topo_BCST, flows=EW), coords=coords,
                                      order_criterium=order_criterium, merging_criterium=merging_criterium,
                                      criterium_BP_position_update=criterium_BP_position_update,
                                    alpha=alpha,ori_input_graph=ori_input_graph,policy_constraint=policy_constraint)  # removes BP from solution-> tree on original coordinates
            #compute cost of CST
            cost_CST = Wiener_index(topo_CST, alpha=alpha) / (num_terminals ** (2 * alpha))
            if best_cost_CST > cost_CST:
                #Store best CST
                best_cost_CST = cost_CST
                best_topo_CST = topo_CST
                bestiter_mstreg_CST = niter + 1
        if niter > maxiter_mSTreg:
            break

        niter += 1
    if verbose:
        print('best mSTreg obtained in iteration %i' % bestiter_mstreg_BCST)
        if return_CST and Compute_CST_each_iter:
            print('best mSTreg CST obtained in iteration %i' % bestiter_mstreg_CST)


    if return_CST:
        if not Compute_CST_each_iter:
            if num_terminals >= 100000:
        
                T_dict,filtered_coords,flows_dict=remove_collapsedBP_from_solution(adj=best_topo_BCST,flows=best_EW,
                                                                                   coords=best_coords,return_Tdict=True)
                best_topo_CST = removeBP(T=T_dict,edge_flows=flows_dict,num_terminals=num_terminals, coords=filtered_coords,
                                         order_criterium=order_criterium, merging_criterium=merging_criterium,
                                         criterium_BP_position_update=criterium_BP_position_update,
                                         alpha=alpha, ori_input_graph=ori_input_graph, policy_constraint=policy_constraint)
            else:
                best_topo_CST = removeBP(adj_to_adj_sparse(best_topo_BCST, flows=best_EW), coords=best_coords,
                                       order_criterium=order_criterium, merging_criterium=merging_criterium,
                                       criterium_BP_position_update=criterium_BP_position_update,
                                     alpha=alpha,ori_input_graph=ori_input_graph,policy_constraint=policy_constraint)  # removes BP from solution-> tree on original coordinates
            # compute cost of CST
            best_cost_CST = Wiener_index(best_topo_CST, alpha=alpha) / (num_terminals ** (2 * alpha))

        return topology(adj=best_topo_BCST), best_topo_CST, best_cost_BCST,best_cost_CST, best_coords, best_EW,
    return topology(adj=best_topo_BCST), best_cost_BCST, best_coords, best_EW




#%%
from numba import njit, prange

# @njit()
# def process_adj(adj, P, threshold, max_freq, min_length):
#     """
#     Process the adjacency matrix and perform filtering and sampling of coordinates.
#
#     Args:
#         adj (numpy.ndarray or None): Adjacency matrix encoding neighboring information.
#         P (numpy.ndarray): Input coordinates.
#         threshold (float): Threshold for determining repeated coordinates.
#         max_freq (int): Maximum frequency for sampling coordinates.
#
#     Returns:
#         numpy.ndarray: Processed coordinates.
#         int: Index of the sampling coordinate.
#
#     """
#     n = len(P) // 2 + 1
#     upper_bound_num_samples = (max_freq - 2) * len(P)
#     sampling_coords_combined_size = int(0.5 * upper_bound_num_samples)
#     partial_sampling_coords = np.empty((sampling_coords_combined_size, P.shape[1]))
#     sampling_coords_ls=[]
#     partial_sampling_coords_index = 0
#     num_sampled_coords=0
#     sub_idx_sampled = -1
#     if adj is not None and max_freq > 2:
#         for i in prange(len(adj)):
#             bp = n + i
#             for j in adj[i]:
#                 if bp < j:
#                     continue
#                 dist = np.linalg.norm(P[bp] - P[j])
#                 freq = min(int(np.ceil(dist / min_length)) + 1, max_freq)
#                 if freq > 2:
#                     # Pre-allocate array for sampling coordinates
#                     for k in range(1, freq - 1):
#                         sub_idx_sampled+=1
#                         partial_sampling_coords[sub_idx_sampled] = P[bp] + (k / (freq - 1)) * (P[j] - P[bp])
#                         num_sampled_coords+=1
#
#                         # Check if sampling_coords_combined is full
#                         if sub_idx_sampled+1 >= sampling_coords_combined_size:
#                             # Append the current array to the list and create a new array
#                             print(partial_sampling_coords.shape)
#
#                             partial_sampling_coords_index += 1
#                             sampling_coords_ls.append(partial_sampling_coords)
#                             sampling_coords_combined_size = min(int(0.3 * upper_bound_num_samples),upper_bound_num_samples- num_sampled_coords)
#                             partial_sampling_coords = np.empty((sampling_coords_combined_size, P.shape[1]))
#                             sub_idx_sampled = -1
#                             print('eo')
#
#
#                     # Update sampling_coords_combined and index
#                     if sub_idx_sampled>=0:
#                         sampling_coords_ls.append(partial_sampling_coords[:sub_idx_sampled+1])
#
#         non_repeated = (compute_pairwise_distances(P[:n], P[n:]) < threshold).sum(0) == 0
#
#         # Combine the arrays into a single numpy array
#         sampling_coords_combined = np.empty((num_sampled_coords, P.shape[1]))
#         start_index = 0
#         for idx in range(len(sampling_coords_ls)):
#             sampling_coord = sampling_coords_ls[idx]
#             end_index = start_index + len(sampling_coord)
#             sampling_coords_combined[start_index:end_index] = sampling_coord
#             start_index = end_index
#
#     else:
#         non_repeated = (compute_pairwise_distances(P[:n], P[n:]) < threshold).sum(0) == 0
#         sampling_coords_combined = np.empty((0, P.shape[1]))  # Initialize as empty NumPy array
#
#     return non_repeated, sampling_coords_combined
#
@njit()
def process_adj(adj, coords, max_freq, min_length,repeated_idxs=None):
    """
    Process the adjacency matrix and perform filtering and sampling of coordinates.

    Args:
        adj (numpy.ndarray or None): Adjacency matrix encoding neighboring information.
        coords (numpy.ndarray): Input coordinates.
        threshold (float): Threshold for determining repeated coordinates.
        max_freq (int): Maximum frequency for sampling coordinates.

    Returns:
        numpy.ndarray: Processed coordinates.
        int: Index of the sampling coordinate.

    """
    n = len(coords) // 2 + 1
    sampling_coords = []
    if adj is not None and max_freq > 2:
        for i in prange(len(adj)):
            if repeated_idxs is not None and i in repeated_idxs:
                continue
            bp = n + i
            for j in adj[i]:
                if bp < j:
                    continue
                elif repeated_idxs is not None and j in repeated_idxs:
                    continue
                dist = np.linalg.norm(coords[bp] - coords[j])
                freq = min(int(np.ceil(dist / min_length)) + 1, max_freq)
                if freq > 2:
                    # Pre-allocate array for sampling coordinates
                    sampling_coord = np.empty((freq - 2, coords.shape[1]))
                    for k in range(1, freq - 1):
                        sampling_coord[k - 1] = coords[bp] + (k / (freq - 1)) * (coords[j] - coords[bp])
                    sampling_coords.append(sampling_coord)


        sampling_coord_length = 0
        for sampling_coord in sampling_coords:
            sampling_coord_length += len(sampling_coord)


        sampling_coords_combined = np.empty((sampling_coord_length, coords.shape[1]))
        start_index = 0
        for sampling_coord in sampling_coords:
            end_index = start_index + len(sampling_coord)
            sampling_coords_combined[start_index:end_index] = sampling_coord
            start_index = end_index

    else:
        sampling_coords_combined = np.empty((0, coords.shape[1]))  # Initialize as empty Num


    return sampling_coords_combined

def get_idxs_collapsed_BPs(adj, coords, threshold,return_representative=False,num_terminals=None,
                           return_norm_diff=False):
    """
        Compute the indices of collapsed branch points based on the given adjacency matrix and coordinates.

        Parameters:
            adj (np.ndarray): The adjacency matrix. num_terminals-2 x 3 matrix, with num_terminals=number of terminals.
            coords (np.ndarray): The coordinates of the nodes.
            threshold (float): The distance threshold for collapsing branch points.

        Returns:
            np.ndarray: The indices of non-collapsed branch points.
            np.ndarray: The indices of collapsed branch points.
    """
    if num_terminals is None:
        num_terminals=len(coords)//2+1
    # Indices of BPs
    BP_indices = np.arange(num_terminals, len(coords))
    # Compute coordinate difference between BPs and its neighbors
    coord_diff = coords[BP_indices, np.newaxis] - coords[adj]
    # Compute distance between BPs and its neighbors
    norm_diff = np.linalg.norm(coord_diff, axis=2)
    # non_repeated_idxs = np.where((norm_diff < threshold).sum(0) == 0)[0]+num_terminals
    if return_norm_diff:
        non_repeated_idxs, repeated_idxs=_get_idxs_collapsed_BPs(adj, norm_diff, threshold, return_representative=return_representative,
                                num_terminals=num_terminals)
        return non_repeated_idxs, repeated_idxs,norm_diff
    else:
        return _get_idxs_collapsed_BPs(adj, norm_diff, threshold, return_representative=return_representative,
                                       num_terminals=num_terminals)

@njit
def while_collapse_generator(init, collapsedto, n):
    '''
    Helper function to backtrack the points to which a branch point is collapsed.
    :param init:
    :param collapsedto:
    :param n:
    :return:
    '''
    continua=True
    while continua:
        if n - 1 <= init:
            break
        if init<0:
            yield init+n
            break
        pred = collapsedto[init]
        yield init+n
        pred -= n
        if pred == init:
            break
        init = pred

# @njit
def _get_idxs_collapsed_BPs(adj, norm_diff, threshold, return_representative=False, num_terminals=None):
    """
        Helper function to compute the indices of collapsed branch points.

        Parameters:
            adj (np.ndarray): The adjacency matrix. num_terminals-2 x 3 matrix, with num_terminals=number of terminals.
            norm_diff (np.ndarray): The norm of coordinate differences.
            threshold (float): The distance threshold to consider as collapsed points
            return_representative (bool): If True, return the representative of each collapsed node.


        Returns:
            np.ndarray: The indices of non-collapsed branch points.
            np.ndarray: The indices of collapsed branch points.
            np.ndarray: The indices of the representative of each collapsed node.
                        Only returned if return_representative is True.
    """
    if num_terminals is None:
        num_terminals = len(norm_diff) + 2# num terminals
    
    total_num_nodes=len(adj)+num_terminals
    
    repeated_idxs = set()
    # array which indicates to which node a BP has been collapsed. It is initialized with 2*num_terminals, which is larger than any
    # node index. We keep track of the collapsed node with the smallest index, because we want to keep one representative
    # for each collapsed node.
    collapsedto=np.full(len(norm_diff), 2 * num_terminals, dtype=np.int64)


    for i in range(len(norm_diff)):
        bp= i + num_terminals
        for j in range(3):
            neigh= adj[i,j]
            if neigh>bp:
                continue
            if norm_diff[i, j] < threshold:
                repeated_idxs.add(bp)
                if neigh>=num_terminals:
                    repeated_idxs.add(neigh)

                #find representant of collapsed node

                possible_repres=[idx for idx in while_collapse_generator(i, collapsedto, num_terminals)]
                possible_repres += [idx for idx in while_collapse_generator(neigh - num_terminals, collapsedto, num_terminals)]
                repre=min(possible_repres)

                idxs= np.array(list(filter(lambda x: x >= num_terminals, possible_repres))) - num_terminals
                #update collapsedto array with new representant
                collapsedto[idxs] = repre



    # find representants of collapsed nodes. this are the ones which have been collapsed to themselves
    visited=set()
    representants=set()

    for bp in repeated_idxs:
        visited.add(bp)
        pred=collapsedto[bp - num_terminals]
        if return_representative:
            path=[bp - num_terminals]
        while 2*num_terminals>pred>=num_terminals:
            if pred == collapsedto[pred - num_terminals]:
                #If pred has been collapsed to itself, then it is a representant
                representants.add(pred)
                visited.add(pred)
                if return_representative:
                    #update collapsedto array with new representant of the backtrack path
                    collapsedto[path] = pred
                break
            if pred in visited:
                if return_representative:
                    collapsedto[path] = collapsedto[pred - num_terminals]
                break
            visited.add(pred)
            if return_representative:
                path.append(pred - num_terminals)
            pred=collapsedto[pred - num_terminals]
            if return_representative and pred<num_terminals:
                collapsedto[path] = pred
                break

    repeated_idxs=repeated_idxs.difference(representants)
    non_repeated_idxs = np.array(list(set(range(num_terminals, total_num_nodes)) - repeated_idxs))
    repeated_idxs = np.array(list(repeated_idxs))
    if return_representative:
        return non_repeated_idxs, repeated_idxs, collapsedto
    return non_repeated_idxs, repeated_idxs

def filter_and_sample_coords(adj, coords, threshold, max_freq):
    """
    Filter and sample coordinates based on the adjacency matrix.

    Args:
        adj (numpy.ndarray or None): Adjacency matrix encoding neighboring information.
        coords (numpy.ndarray): Input coordinates.
        threshold (float): Threshold for determining repeated coordinates.
        max_freq (int): Maximum frequency for sampling coordinates.

    Returns:
        numpy.ndarray: Processed coordinates.
        int: Index of the sampling coordinate.

    """
    num_terminals = len(coords) // 2 + 1

    
    non_collapsed_idxs,collapsed_idxs,norm_diff = get_idxs_collapsed_BPs(adj, coords, threshold,return_norm_diff=True)
    if len(non_collapsed_idxs)>0:
        coords_=np.vstack([coords[:num_terminals], coords[non_collapsed_idxs]])
    else:
        coords_ = coords[:num_terminals]
    if max_freq>2:
        # k1nn = kneighbors_graph(coords_, n_neighbors=1, mode='distance').data
        # k1nn[k1nn < 1e-3] = np.inf
        # min_length = k1nn.min()
        
        min_length = np.min(norm_diff[norm_diff>threshold])
        
        sampled_coords = process_adj(adj, coords, max_freq, min_length,repeated_idxs=collapsed_idxs)

        coords_ = np.vstack([coords_, sampled_coords])
        idx_sampling_coord=len(coords_)-len(sampled_coords)
    else:
        idx_sampling_coord = len(coords_)
    return coords_, idx_sampling_coord

def filter_and_sample_coords_nojit(adj, coords, threshold, max_freq):
    k1nn = kneighbors_graph(coords, n_neighbors=1, mode='distance').data
    k1nn[k1nn < 1e-3] = np.inf
    min_length = k1nn.min()

    n = len(coords) // 2 + 1

    sampling_coords = []
    if adj is not None and max_freq > 2:
        for i in range(len(adj)):
            bp = n + i
            for j in adj[i]:
                if bp < j:
                    continue
                dist = np.linalg.norm(coords[bp] - coords[j])
                freq = min(np.ceil(dist / min_length).astype(int) + 1, max_freq)
                if freq > 2:
                    sampling_coords.append(coords[bp] + np.linspace(0, 1, freq)[1:-1, None] * (coords[j] - coords[bp]))

        non_repeated = (cdist(coords[:n], coords[n:]) < threshold).sum(0) == 0
        idx_sampling_coord = n + len(np.where(non_repeated)[0])
        coords_ = np.vstack([coords[:n], coords[np.where(non_repeated)[0] + n], np.concatenate(sampling_coords)])
    else:
        non_repeated = (cdist(coords[:n], coords[n:]) < threshold).sum(0) == 0
        coords_ = np.vstack([coords[:n], coords[np.where(non_repeated)[0] + n]])
        idx_sampling_coord = len(coords_)

    return coords_, idx_sampling_coord


def topo_mST_reguralization(coords, threshold=1e-5, adj=None, max_freq=10,
                            jit=True, mST_fromknn=True, ori_input_graph=None, policy_constraint=None):
    '''
    Given the (branching points) BP and the terminal coordinates it computes the mST over these points. From this tree
    it computes a new topology.

    Motivation: The BP offer flexibility and their position may indicate the desired paths of the BCST solution
     which may try to compensate the deficiencies of the current topology. Thus, redefine the topology based on
     the BPs may be beneficial.

    :param coords: np.array with P of terminals (indices <n) and BP (indices >n)
    :param return_mST: if True returns the mST with all coordinates (BP+terminals)
    :param threshold: threshold to filter BP which share the same location. If distance of two BP is lower than
    threshold one of these is removed
    :param adj: adjacency format providing neighbors of the BP. If given it is used to augment points in the edges
    to have more coordinates so that the mST follows more the current solution
    :param max_freq: maximum number of augemented points per edge
    :param jit: if True uses numba to speed up the computation
    :param mST_fromknn: if True computes the mST from the knn graph. This is faster but may lead to different results
    than computing the mST from the distance matrix
    :param ori_input_graph: Adjacency sparse matrix with the entries representing the euclidean distance. If given it
     constraints the full tree topologies to be derived from spanning trees of the original graph.
    :return:
    '''

    n = len(coords) // 2 + 1
    if jit:
        coords_,idx_sampling_coord=filter_and_sample_coords(adj, coords, threshold, max_freq)
    else:
        coords_, idx_sampling_coord = filter_and_sample_coords_nojit(adj, coords, threshold, max_freq)


    #ensure that distances between different nodes is not 0
    # coords_=coords_+np.random.normal(0,1e-14,size=coords_.shape)
    if mST_fromknn:
        A=ensure_connected_knn_graph(coords_)

        if 0 in A.data:
            A=recompute_distances(A,coords_)
        if 0 in A.data:
            raise ValueError(
                'There are 0 distances in the distance matrix. This may cause problems in the mST computation')
            warnings.warn(
                'There are 0 distances in the distance matrix. This may cause problems in the mST computation')
            A.data += 1e-14
    else:
        A=cdist(coords_,coords_)
        if 0 in A[np.triu_indices_from(A, k=1)]:
            raise ValueError(
                'There are 0 distances in the distance matrix. This may cause problems in the mST computation')
            warnings.warn(
                'There are 0 distances in the distance matrix. This may cause problems in the mST computation')
            A += 1e-14

    # compute mST
    mST = sp.csgraph.minimum_spanning_tree(A)
    mST += mST.T

    if ori_input_graph is not None and policy_constraint== 'recursive':
        mST=constrain2edge_index(mST,num_terminals=n,ori_input_graph=ori_input_graph)

    newadj = update_topoBP2binarytopoBP(mST, n)
    return newadj

def recompute_distances(knn_graph, coords):
    knn_graph = knn_graph.tocoo()
    # zero_indices = np.where(knn_graph.data == 0)[0]
    # row_indices = knn_graph.row[zero_indices]
    # col_indices = knn_graph.col[zero_indices]
    #
    # distances = np.linalg.norm(P[row_indices] - P[col_indices], axis=1)
    # knn_graph.data[zero_indices] = distances

    distances = np.linalg.norm(coords[knn_graph.row] - coords[knn_graph.col], axis=1)
    knn_graph.data = distances

    return knn_graph


def update_topoBP2binarytopoBP(T, n):
    '''
        Given an adjacency matrix of a tree,T, with at least 2*(n-1) nodes (terminals + BP). The algorithm start traversing
        the tree T from the leaves. Every time two terminals are  traversed in the same path these are interconnected by a new
        BP in the new topology. Every time a bifurcation is encountered in the tree, another leave is used for starting.
        Imagine the following tree with 4 terminals (1,2,3,4) and two BP (5,6).

            1
            |
        4-5-6-2-3

        The algorithm could start from 1 and would find next node 6. Since node 6 is not a terminal nothing happens, but from here
        we would have two choices to follow the path. Thus we remember that  node 6 is represented by terminal 1 and start from another
        leave, say 3. Then 3 would encounter 2 which is also a terminal. In the new topology, these are connected by a BP (say 5.2)
        and we would continue the path from 2 (which now represents node 5.2 in the new topology). The next node enocuntered would be
        6, which represents node 1, i.e. a terminal. Thus in the new topology node 5.2 and node 1 are linked by another BP
        , say 6.2. Now we continue from node 6, which would represent node 6.2. We would continue the path till finding a
        terminal (or a representative of a terminal or BP in the newtopology) till there are no more nodes. In this case we would merge
        6.2 with node 4 (we do not add another BP since 4 is the last node to add). The final topo would be

           1
           |
        4-6.2-5.2-3
               |
               2


        :param T: sparse matrix of the adjacency matrix of the current soultion including the branching points (BP)
        :param n: Number of terminals (i.e. without BP). It is used to identify the terminals, which are indexed by the
        indices with value lower than n
    '''
    T_dict={i:T.indices[T.indptr[i]:T.indptr[i+1]].tolist() for i in range(T.shape[0])}
    # initialize an empty adjacency matrix for F
    newadj = np.full((n - 2, 3), -1, dtype=np.intc)
    track_indices_BP = np.zeros(n - 2, dtype=int)

    # node_representative keeps track of the node representative for each node
    node_representative = -np.ones(T.shape[0], dtype=np.int32)  # -1 means no representative
    node_representative[:n] = np.arange(n)  # terminals represent themselves

    leaves = np.where(T.astype(bool).sum(1) == 1)[0]  # get leaves
    visited = np.zeros(T.shape[0], dtype=bool)
    BP_idx  = n
    break_all = False
    for leaf in leaves:
        if break_all:
            break
        if visited[leaf]:
            continue
        node_stack = [leaf]

        while node_stack:
            u = node_stack.pop()
            visited[u] = True

            # remove u from T and add its neighbor(s) to the stack
            v = T_dict[u][0]
            del T_dict[u]
            T_dict[v].remove(u)
            if len(T_dict[v]) == 1:  # add only if it's a leaf
                node_stack.append(v)

            repr_u = node_representative[u]
            repr_v = node_representative[v]

            if repr_u>=0:  # u represents itself or a BP
                if repr_v >=0:
                    # check if node is the last one to add so that no extra last BP is added
                    if BP_idx >= n * 2 - 2 :
                        if  repr_u>= n:
                            aux_idx = repr_u - n
                            newadj[aux_idx, track_indices_BP[aux_idx]] = repr_v
                            track_indices_BP[aux_idx] += 1
                        if repr_v >= n:
                            aux_idx = repr_v - n
                            newadj[aux_idx, track_indices_BP[aux_idx]] = repr_u
                            track_indices_BP[aux_idx] += 1
                        break_all = True
                        break
                    if BP_idx<2*n-2:
                        aux_BP_idx = BP_idx - n
                        newadj[aux_BP_idx, track_indices_BP[aux_BP_idx]] = repr_u
                        track_indices_BP[aux_BP_idx] += 1

                        newadj[aux_BP_idx, track_indices_BP[aux_BP_idx]] = repr_v
                        track_indices_BP[aux_BP_idx] += 1

                        if repr_u >= n:
                            aux_idx = repr_u - n
                            newadj[aux_idx, track_indices_BP[aux_idx]] = BP_idx
                            track_indices_BP[aux_idx] += 1
                        if repr_v >= n:
                            aux_idx = repr_v - n
                            newadj[aux_idx, track_indices_BP[aux_idx]] = BP_idx
                            track_indices_BP[aux_idx] += 1


                        node_representative[v] = BP_idx
                        BP_idx+= 1  # update BP_idx

                else:
                    node_representative[v] = repr_u


    return newadj



def remove_collapsedBP_from_solution(adj,flows,coords,threshold_collapse=1e-5,
                                     return_Tdict=False,num_terminals=None):

    
    if num_terminals is None:
        num_terminals=len(adj)+2
    
    total_num_nodes=len(coords)
    
    num_BPs=total_num_nodes-num_terminals
    
    non_collapsed_idxs, collapsed_idxs,representants=get_idxs_collapsed_BPs(adj, coords,
                                                                            threshold_collapse,
                                                                            return_representative=True,num_terminals=num_terminals)
    non_collapsed_idxs=np.sort(non_collapsed_idxs)
    old2new_bp_idxs={old_bp_idx:new_bp_idx+num_terminals for old_bp_idx,new_bp_idx in zip(non_collapsed_idxs,range(len(non_collapsed_idxs)))}


    if return_Tdict:
        T_dict={i:set() for i in range(len(non_collapsed_idxs)+num_terminals)}
    else:
        T_bp_filtered=sp.lil_matrix((len(non_collapsed_idxs)+num_terminals,len(non_collapsed_idxs)+num_terminals))
    flows_dict={}
    for i in range(num_BPs):
        bp=i+num_terminals

        if representants[i] < total_num_nodes:
            repre_bp = representants[i]
        else:
            repre_bp = bp
        if repre_bp < num_terminals:
            new_bp_idx = repre_bp
        else:
            new_bp_idx = old2new_bp_idxs[repre_bp]
        
   
        for j,neigh in enumerate(adj[i]):
            if bp<neigh:
                continue


            # since it is not collapsed, it is a representative of itself
            
            if neigh<num_terminals:
                new_neigh_idx=neigh
                repre_neigh = neigh
            else:
                if representants[neigh-num_terminals]<total_num_nodes:
                    repre_neigh = representants[neigh-num_terminals]
                else:
                    repre_neigh = neigh
                if repre_neigh<num_terminals:
                    new_neigh_idx=repre_neigh
                else:
                    new_neigh_idx=old2new_bp_idxs[repre_neigh]
            if new_bp_idx == new_neigh_idx:
                continue
            if return_Tdict:
                
                T_dict[new_neigh_idx].add(new_bp_idx)
                T_dict[new_bp_idx].add(new_neigh_idx)
                if (new_neigh_idx, new_bp_idx) in flows_dict:
                    flows_dict[(new_neigh_idx, new_bp_idx)] = max(flows[i, j],flows_dict[(new_neigh_idx, new_bp_idx)])
                    flows_dict[(new_bp_idx, new_neigh_idx)] = max(flows[i, j],flows_dict[(new_bp_idx, new_neigh_idx)])
                else:
                    flows_dict[(new_neigh_idx, new_bp_idx)] = flows_dict[(new_bp_idx, new_neigh_idx)] = flows[i, j]
            else:
                T_bp_filtered[new_neigh_idx,new_bp_idx]=T_bp_filtered[new_bp_idx,new_neigh_idx]=np.linalg.norm(coords[repre_bp]-coords[repre_neigh])
                if new_neigh_idx<new_bp_idx:
                    if (new_neigh_idx,new_bp_idx) in flows_dict:
                        flows_dict[(new_neigh_idx,new_bp_idx)]=max(flows[i, j],flows_dict[(new_neigh_idx,new_bp_idx)])
                    else:
                        flows_dict[(new_neigh_idx, new_bp_idx)] = flows[i, j]
                else:
                    if (new_bp_idx,new_neigh_idx) in flows_dict:
                        flows_dict[(new_bp_idx,new_neigh_idx)]=max(flows[i, j],flows_dict[(new_bp_idx,new_neigh_idx)])
                    else:
                        flows_dict[(new_bp_idx, new_neigh_idx)] = flows[i, j]


    kept_idxs = np.concatenate([np.arange(num_terminals), non_collapsed_idxs])
    filtered_coords = coords[kept_idxs]
    
    if return_Tdict:
        T_dict={i:list(T_dict[i]) for i in T_dict}
        return T_dict,filtered_coords,flows_dict

    T_bp_filtered=T_bp_filtered.tocsr()
    T_bp_filtered.sort_indices()
    widths = [flows_dict[(i, j)] for i, j in zip(*sp.triu(T_bp_filtered).nonzero())]
    
    return T_bp_filtered,filtered_coords,widths,kept_idxs







