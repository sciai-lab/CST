from typing import Dict, Tuple,List

import logging
import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx
from .update_posBP_options import BP_position_update_for_removal
from .BP_collapse_options import choose_merging_neighBP#,merge_closestSP
from .OrderBP import get_orderBPcls
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=FutureWarning)
import numba as nb
from pqdict import pqdict
try:
    import networkit as nk
    nk_available=True
except:
    nk_available=False


def removeBP(T, coords,order_criterium='closestterminals',merging_criterium='tryall',criterium_BP_position_update='median',
             alpha=1, edge_flows=None,num_terminals=None,collapseBPs_CST=False):
    '''
    Given an adjacency matrix of a tree,T, representing final topology of a BCST problem, the function removes the Steiner points
    in the tree, by merging them with the one of its neighbors. The function returns the new topology with the Steiner points removed.

    :param T: topology with BP (sparse matrix)
    :param coords: coordinates nodes
    :param order_criterium: how to order the BPs for merging
    :param merging_criterium: To which neighbor should the BP merged
    :param criterium_BP_position_update: How to update the BPs position
    :param alpha: Alpha value cost-> default 1
    :param return_nx: return networkx or sparse matrix. Default False-> returns sparse
    :return:
    '''
    
    
    if sp.issparse(T):
        num_terminals = len(coords) // 2 + 1
        T_dict = {i: T.indices[T.indptr[i]:T.indptr[i + 1]].tolist() for i in range(T.shape[0])}
        edge_flows = dict(T.todok())
    elif isinstance(T, dict):
        T_dict=deepcopy(T)
        assert num_terminals is not None
        assert edge_flows is not None
    edge_weights = {}
    
    
    if collapseBPs_CST:
        #define edge weights
        define_edge_weights_dict(T_dict,coords,edge_weights,'all')
        
        #merge BPs which are collapsed in the same position
        merge_collapsed_BPs(T_dict,coords,edge_flows,edge_weights,num_terminals)
    else:
        if sp.issparse(T):
            pass
        elif isinstance(T, dict):
            define_edge_weights_dict(T_dict, coords,edge_weights, mode='terminal2terminal',num_terminals=num_terminals)
    
    
    #initialize queue
    Q=get_orderBPcls(order_criterium=order_criterium,coords=coords,T_dict=T_dict,num_terminals=num_terminals)

    altered_coords=coords.copy()
    visited_BPs=set()
    while not Q.empty():
        BP=Q.pop()
        if BP in visited_BPs:
            continue
        visited_BPs.add(BP)
        
        # Choose the neighbor to merge
        merging_neighBP=choose_merging_neighBP(T_dict=T_dict,coords=altered_coords,BP=BP,
                                               edge_flows=edge_flows,
                                               alpha=alpha,merging_criterium=merging_criterium)



        # Realize the merging
        mergeBP(T_dict,coords,BP,merging_neighBP,edge_flows=edge_flows,edge_weights=edge_weights,)
        

        # remove BP
        if criterium_BP_position_update!='average':
            #average needs the BP to compute the new position still
            remove_node_from_Tdict(T_dict=T_dict,node=BP,edge_weights=edge_weights,edge_flows=edge_flows)
            
        
        #update position BP
        altered_coords=BP_position_update_for_removal(coords=altered_coords, T_dict=T_dict,
                                                      neigh_BP_merged=merging_neighBP, num_terminals=num_terminals,
                                                        edge_flows=edge_flows, alpha=alpha,
                                                      criterium_BP_position_update=criterium_BP_position_update)


        #add new / update elements queue
        Q.put(coords=altered_coords,T_dict=T_dict,merging_node=merging_neighBP,removed_node=BP)
        

    
    T_sparse = sp.dok_matrix((num_terminals, num_terminals))  # , dtype=T.dtype)
    try:
        # Attempt to update using _dict if available (valid for scipy 1.13.0 version)
        T_sparse._dict.update(edge_weights)
    except AttributeError:
        try:
            # If _dict is not available, attempt to use _update (valid for scipy 1.9.3 version)
            T_sparse._update(edge_weights)
        except AttributeError:
            # If neither _dict nor _update are available, update individual elements
            # This fallback method may be slower and less efficient
            for e in edge_weights:
                T_sparse[e[0], e[1]] = edge_weights[e]
    
    T_sparse = T_sparse.tocsr()
    return T_sparse




def mergeBP(T_dict: Dict[int, List[int]],
            coords: np.ndarray,
            BP: int,
            merging_neighBP: int,
            edge_flows: Dict[Tuple[int, int], float],
            edge_weights: Dict[Tuple[int, int], float]):
    for i, neighbor in enumerate(T_dict[BP]):
        if neighbor != merging_neighBP:
            d = np.linalg.norm(coords[merging_neighBP] - coords[neighbor])
            T_dict[neighbor].append(merging_neighBP)
            T_dict[merging_neighBP].append(neighbor)
            edge_flows[(neighbor, merging_neighBP)] = edge_flows[(merging_neighBP, neighbor)] = edge_flows[
                (neighbor, BP)]
            edge_weights[(neighbor, merging_neighBP)] = edge_weights[(merging_neighBP, neighbor)] = d


def merge_collapsed_BPs(T_dict: Dict[int, List[int]],
                        coords: np.ndarray,
                        edge_flows: Dict[Tuple[int, int], float],
                        edge_weights: Dict[Tuple[int, int], float],
                        num_terminals: int):
    '''
    Merges BPs which are collapsed in the same position
    '''
    BP_pqdict = pqdict({BP: len(T_dict[BP]) for BP in T_dict if BP >= num_terminals})
    while len(BP_pqdict) > 0:
        BP = BP_pqdict.pop()
        neighbors = T_dict.get(BP, None)
        if neighbors is None:
            continue
        for merging_node in neighbors:
            d = np.linalg.norm(coords[BP] - coords[merging_node])
            if d < 1e-5:
                # Realize the merging
                mergeBP(T_dict, coords, BP, merging_node, edge_flows=edge_flows, edge_weights=edge_weights)

                # remove BP
                remove_node_from_Tdict(T_dict=T_dict, node=BP, edge_weights=edge_weights, edge_flows=edge_flows)

                break


def remove_node_from_Tdict(T_dict: Dict[int, List[int]],
                            node: int,
                            edge_flows: Dict[Tuple[int, int], float],
                            edge_weights: Dict[Tuple[int, int], float]):
    for neigh in T_dict[node]:
        T_dict[neigh].remove(node)
        del edge_flows[(neigh, node)]
        del edge_flows[(node, neigh)]
        edge_weights.pop((node, neigh), None)
        edge_weights.pop((neigh, node), None)
    del T_dict[node]



def define_edge_weights_dict(T_dict: Dict[int, list],
                             coords: np.ndarray,
                             edge_weights: Dict[Tuple[int, int], float],
                             mode: str = 'all',
                             num_terminals: int = 0):
    if mode == 'terminal2terminal':
        assert (num_terminals > 0)
        for i in T_dict:
            for j in T_dict[i]:
                if i < j and j < num_terminals:
                    edge_weights[(j, i)] = edge_weights[(i, j)] = np.linalg.norm(coords[i] - coords[j])

    elif 'all':
        for i in T_dict:
            for j in T_dict[i]:
                if i < j:
                    edge_weights[(j, i)] = edge_weights[(i, j)] = np.linalg.norm(coords[i] - coords[j])

