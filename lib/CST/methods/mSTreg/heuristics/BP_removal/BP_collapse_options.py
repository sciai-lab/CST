import numpy as np
from sklearn.metrics import pairwise_distances
from numba import njit
import scipy.sparse  as sp
from .utils_terminal_side_edge import check_validity_edge_restriction
def choose_merging_neighBP(T_dict, coords, BP,edge_flows, alpha=1, merging_criterium='closest',ori_input_graph_dict=None,
                           terminal_edge_sides=None,SP_non_possible_merges=None):
    '''

    :param T_nx:
    :param coords:
    :param BP:
    :param alpha:
    :param merging_criterium:
        - closest (default) -> merges to closest neighbor
        - minflowcost -> merges to the neighbor connected by the edge with lowest cost*flow**alpha
        - tryall -> tries all possibilities and chooses the one with lowest cost given the current BP positions
    :param ori_input_graph_dict: dict adjaceny representation over terminals with the entries representing the euclidean distance
        between terminal nodes. If given it forces that the removal of BP generates a spanning trees of this graph.
    :return:
    '''
    neighbors_BP = T_dict[BP]


    if merging_criterium.lower()== 'closest':
        return choice_merge_closest(coords,neighbors_BP,BP,ori_input_graph_dict=ori_input_graph_dict,
                                    terminal_edge_sides=terminal_edge_sides,T_dict=T_dict,
                                    SP_non_possible_merges=SP_non_possible_merges)
    elif merging_criterium.lower()== 'minflowcost':
        return choice_merge_minflowcost(coords,neighbors_BP,BP,edge_flows,ori_input_graph_dict=ori_input_graph_dict,
                                    terminal_edge_sides=terminal_edge_sides,T_dict=T_dict,
                                    SP_non_possible_merges=SP_non_possible_merges)
    elif merging_criterium== 'tryall':
        return choice_merge_tryall(coords, neighbors_BP, BP,edge_flows, alpha,ori_input_graph_dict=ori_input_graph_dict,
                                    terminal_edge_sides=terminal_edge_sides,T_dict=T_dict,
                                    SP_non_possible_merges=SP_non_possible_merges)
    else:
        raise NotImplementedError('The merging policy of the BP "%s" in the removal process is not implemented' % merging_criterium)

def choice_merge_closest(coords,neighbors_BP,BP,ori_input_graph_dict=None,
                         terminal_edge_sides=None,T_dict=None,
                         SP_non_possible_merges=None):
    # compute distances between neighbors
    D2BP = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1)
    if ori_input_graph_dict is None:
        # compute idx closest neighbor
        merging_neighBP = neighbors_BP[np.argmin(D2BP)]
        return merging_neighBP

    neighbors_BP_sorted=np.array(neighbors_BP)[np.argsort(D2BP)]
    return neighbors_BP_sorted
    # return compute_merging_node_with_edge_constrains(neighbors_BP_sorted,BP,T_dict,ori_input_graph_dict,
    #                                                  terminal_edge_sides,SP_non_possible_merges)
def choice_merge_minflowcost(coords,neighbors_BP,BP,edge_flows,alpha=1,ori_input_graph_dict=None,
                         terminal_edge_sides=None,T_dict=None,
                         SP_non_possible_merges=None):
    # compute distances between neighbors and multiply with the flow**alpha
    D2BPflow = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1) * np.array(
        [edge_flows[(BP,neigh)] for neigh in neighbors_BP]) ** alpha
    if ori_input_graph_dict is None:
        # compute idx neighbor
        merging_neighBP = neighbors_BP[np.argmin(D2BPflow)]
        return merging_neighBP

    neighbors_BP_sorted = np.array(neighbors_BP)[np.argsort(D2BPflow)]
    return neighbors_BP_sorted
    # return compute_merging_node_with_edge_constrains(neighbors_BP_sorted,BP,T_dict,ori_input_graph_dict,
    #                                                  terminal_edge_sides,SP_non_possible_merges)


def choice_merge_tryall(coords,neighbors_BP,BP,edge_flows,alpha=1,ori_input_graph_dict=None,
                         terminal_edge_sides=None,T_dict=None,
                         SP_non_possible_merges=None):
    '''try all possible collapse possibilities and take the one that minimizes the cost'''

    D=pairwise_distances(coords[neighbors_BP])
    flows=np.array([edge_flows[(neighbor,BP)] for neighbor in neighbors_BP])**alpha
    flows=np.tile(flows,(D.shape[0],1))
    if ori_input_graph_dict is None:
        merging_neighBP=neighbors_BP[np.argmin((D*flows).sum(1))]
        return merging_neighBP

    neighbors_BP_sorted = np.array(neighbors_BP)[np.argsort((D*flows).sum(1))]
    return neighbors_BP_sorted
    # return compute_merging_node_with_edge_constrains(neighbors_BP_sorted,BP,T_dict,ori_input_graph_dict,
    #                                                  terminal_edge_sides,SP_non_possible_merges)



# def merge_closestSP(T_dict,P,weights='distance',alpha=1,T=None):
#     n=len(P)//2+1
#     T_=set_weights(T_dict,P,weights=weights,alpha=alpha,T=T)
#
#
#     D=get_Shortest_path_matrix(T_)
#     closest_terminals=np.argmin(D[n:,:n],axis=1)
#
#
#     T_collapsed = sp.lil_matrix((n, n))
#     added_edges = set()
#
#     for i, terminal_bp1 in enumerate(closest_terminals):
#         bp1 = i + n
#         for node in T_dict[bp1]:
#             if node > bp1:
#                 continue
#             if node < n:
#                 closest_terminal_node = node
#             else:
#                 closest_terminal_node = closest_terminals[node - n]
#             edge = (closest_terminal_node, terminal_bp1)
#             if edge not in added_edges:
#                 T_collapsed[edge[1],edge[0]] = T_collapsed[edge[0],edge[1]] = np.linalg.norm(P[closest_terminal_node] - P[terminal_bp1])
#                 added_edges.add(edge)
#                 added_edges.add((edge[1],edge[0]))
#     return T_collapsed
@njit
def set_weights(T_dict,coords,weights='distance',alpha=1,T=None):
    T_ = sp.lil_matrix((len(T_dict),len(T_dict)))
    if weights == 'distance':
        for i in T_dict.keys():
            T_[i, T_dict[i]] = np.linalg.norm(coords[i] - coords[T_dict[i]],axis=1)
    elif weights == 'distance_flow':
        for i in T_dict.keys():
            T_[i, T_dict[i]] = (T[i, T_dict[i]] ** alpha).multiply(np.linalg.norm(coords[i] - coords[T_dict[i]]),axis=1)
    else:
        raise NotImplementedError('The weights option %s is not implemented. Only "distance" and "distance_flow" are allowed' % weights)
    return T_


def compute_merging_node_with_edge_constrains(neighbors_BP_sorted,BP,T_dict,ori_input_graph_dict,terminal_edge_sides,
                                              SP_non_possible_merges):
    num_terminals = len(ori_input_graph_dict)
    for merging_neighBP in neighbors_BP_sorted:
        if merging_neighBP in SP_non_possible_merges[BP] or ( BP in SP_non_possible_merges[merging_neighBP]):
            continue
        if check_validity_edge_restriction((BP, merging_neighBP), T_dict,
                                           ori_input_graph_dict, terminal_edge_sides,
                                           SP_non_possible_merges=SP_non_possible_merges):
            return merging_neighBP
        SP_non_possible_merges[BP].add(merging_neighBP)
        SP_non_possible_merges[merging_neighBP].add(BP)


    assert False, "should never reach this point, because there should always be a valid merge"