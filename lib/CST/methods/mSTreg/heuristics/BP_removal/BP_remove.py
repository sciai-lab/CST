import logging
import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx
from .update_posBP_options import BP_position_update_for_removal
from .BP_collapse_options import choose_merging_neighBP#,merge_closestSP
from .OrderBP import get_orderBPcls
import warnings
from .utils_terminal_side_edge import get_terminal_edge_sides,check_validity_edge_restriction,constrain2edge_index
from copy import deepcopy
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import networkit as nk
    nk_available=True
except:
    nk_available=False

import time

import sys
def removeBP(T, coords,order_criterium='closestterminals',merging_criterium='tryall',criterium_BP_position_update='median',
             alpha=1, edge_flows=None,num_terminals=None):
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
        edge_weights = {}
    elif isinstance(T, dict):
        T_dict=deepcopy(T)
        assert num_terminals is not None
        assert edge_flows is not None
        edge_weights = {(i,j):np.linalg.norm(coords[i]-coords[j]) for i in T_dict for j in T_dict[i] if i<num_terminals and j<num_terminals}



    

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
        Q.put(coords=altered_coords,T_dict=T_dict,merging_node=merging_neighBP)
        

    
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




def mergeBP(T_dict,coords,BP,merging_neighBP,edge_flows,edge_weights,
            terminal_edge_sides=None):
   

    for i,neighbor in enumerate(T_dict[BP]):
        if neighbor!=merging_neighBP:
            d=np.linalg.norm(coords[merging_neighBP]-coords[neighbor])
            T_dict[neighbor].append(merging_neighBP)
            T_dict[merging_neighBP].append(neighbor)
            edge_flows[(neighbor,merging_neighBP)]=edge_flows[(merging_neighBP,neighbor)]=edge_flows[(neighbor,BP)]
            edge_weights[(neighbor,merging_neighBP)]=edge_weights[(merging_neighBP,neighbor)]=d
            if terminal_edge_sides is not None:
                new_e=min(neighbor,merging_neighBP),max(neighbor,merging_neighBP)
                old_e=min(neighbor,BP),max(neighbor,BP)
                terminal_edge_sides[new_e]={}

                terminal_edge_sides[new_e][neighbor]=terminal_edge_sides[old_e][neighbor]

                terminal_edge_sides[new_e][merging_neighBP] = terminal_edge_sides[old_e][BP]
                del terminal_edge_sides[old_e]


    

def remove_node_from_Tdict(T_dict, node, edge_weights, edge_flows):
    for neigh in T_dict[node]:
        T_dict[neigh].remove(node)
        del edge_flows[(neigh, node)]
        del edge_flows[(node, neigh)]
        edge_weights.pop((node, neigh), None)
        edge_weights.pop((neigh, node), None)
    del T_dict[node]










