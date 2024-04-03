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
             alpha=1,ori_input_graph=None,policy_constraint='shortest_path',
             edge_flows=None,num_terminals=None):
    '''
    Given an adjacency matrix of a tree,T, representing final topology of a BOT problem, the function removes BP

    :param T: topology with BP (sparse matrix)
    :param coords: coordinates nodes
    :param order_criterium: how to order the BPs for merging
    :param merging_criterium: To which neighbor should the BP merged
    :param criterium_BP_position_update: How to update the BPs position
    :param alpha: Alpha value cost-> default 1
    :param return_nx: return networkx or sparse matrix. Default False-> returns sparse
    :param edge_restriction: list of tuples representing the edges that are allowed to be in the final tree. If None,
    all edges are allowed
    :param ori_input_graph: Adjacency sparse matrix over terminals with the entries representing the euclidean distance
            between terminal nodes. If given it forces that the removal of BP generates a spanning trees of this graph.
    :return:
    '''

    if ori_input_graph is not None and policy_constraint=='recursive':
        # sys.setrecursionlimit(T.shape[0]**2)
        # print(sys.getrecursionlimit())
        return removeBP_constrained(T, coords, ori_input_graph,
                             order_criterium=order_criterium, merging_criterium=merging_criterium,
                             criterium_BP_position_update=criterium_BP_position_update,
                             alpha=1,edge_flows=edge_flows,num_terminals=num_terminals)

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

    # if ori_input_graph is not None:
    #     # compute the terminal nodes at each side of each edge (only the ones that are connected via a path not containing other terminals)
    #     terminal_edge_sides = get_terminal_edge_sides(T_dict, num_terminals)
    #     ori_input_graph_dict = {
    #         i: ori_input_graph.indices[ori_input_graph.indptr[i]:ori_input_graph.indptr[i + 1]].tolist() for i in
    #         range(ori_input_graph.shape[0])}
    #
    #     SP_non_possible_merges = {SP: set() for SP in range(2 * num_terminals - 2)}
    # else:
    #     terminal_edge_sides=None
    #     ori_input_graph_dict=None
    #     SP_non_possible_merges=None
    
    
    # if merging_criterium=='closest_SP':
    #     T_sparse=merge_closestSP(T_dict,P,weights='distance',alpha=alpha,T=T)
    #     return T_sparse.tocsr()

    

    Q=get_orderBPcls(order_criterium=order_criterium,coords=coords,T_dict=T_dict,num_terminals=num_terminals)

    altered_coords=coords.copy()
    visited_BPs=set()
    while not Q.empty():
        BP=Q.pop()
        if BP in visited_BPs:
            continue
        visited_BPs.add(BP)
        
        
        # num_BPneighs=len(T_dict[BP])
        # if len(Q) < 15000:
        #     sorted_neighs=sorted([(len(T_dict[BP]),BP) for BP in Q.Q],reverse=True)
        # s_time_choose=time.time()
        merging_neighBP=choose_merging_neighBP(T_dict=T_dict,coords=altered_coords,BP=BP,
                                               edge_flows=edge_flows,
                                               alpha=alpha,merging_criterium=merging_criterium)



        # s_time_merge=time.time()
        # Realize the merging
        mergeBP(T_dict,coords,BP,merging_neighBP,edge_flows=edge_flows,edge_weights=edge_weights,)
                # terminal_edge_sides=terminal_edge_sides,
                # SP_non_possible_merges=SP_non_possible_merges)

        # remove BP
        if criterium_BP_position_update!='average':
            #average needs the BP to compute the new position still
            remove_node_from_Tdict(T_dict=T_dict,node=BP,edge_weights=edge_weights,edge_flows=edge_flows)
            # if terminal_edge_sides is not None:
            #     del terminal_edge_sides[tuple(sorted((BP,merging_neighBP)))]

        # if terminal_edge_sides is not None:
        #     #update terminal edge sides by removing the terminals at the side of a non possible merge
        #     terminal_edge_sides = get_terminal_edge_sides(T_dict, num_terminals,non_possible_merges=SP_non_possible_merges)
        
        # s_time_update=time.time()
        #update position BP
        altered_coords=BP_position_update_for_removal(coords=altered_coords, T_dict=T_dict,
                                                      neigh_BP_merged=merging_neighBP, num_terminals=num_terminals,
                                                        edge_flows=edge_flows, alpha=alpha,
                                                      criterium_BP_position_update=criterium_BP_position_update)


        # s_time_put=time.time()
        #add new / update elements queue
        Q.put(coords=altered_coords,T_dict=T_dict,merging_node=merging_neighBP)
        # if len(Q)<15000:
        #     print(sorted_neighs[:10])
        # print('left iterations=%i, num_neighbors=%i,total_time=%f, choose=%f, merge=%f, update=%f, put=%f'%(len(Q),num_BPneighs,
        #                                                                                                              time.time()-s_time_choose,
        #                                                                                                              s_time_merge-s_time_choose,
        #                                                                                                              s_time_update-s_time_merge,
        #                                                                                                              s_time_put-s_time_update,
        #                                                                                                              time.time()-s_time_put))
        

    if ori_input_graph is not None and policy_constraint=='shortest_path':
        return reconnect_via_shortespath(edge_weights,ori_input_graph=ori_input_graph)

    T_sparse = sp.dok_matrix((num_terminals, num_terminals))#, dtype=T.dtype)
    T_sparse._update(edge_weights)
    T_sparse = T_sparse.tocsr()
    return T_sparse

# def reconnect_via_shortespath(T,ori_input_graph):
#     edges_T = [(u, v) for u, v in zip(*sp.triu(T).nonzero())]
#     T = T.tolil()
#     # dict with nodes as keys and adjacent node of edges removed as values
#     adjacent_edges_removed = {}
#
#     for e in edges_T:
#         u, v = e
#         if ori_input_graph[u, v] == 0:
#             #set to zero edges not in original graph
#             T[u, v] = T[v, u] = 0
#             if u in adjacent_edges_removed:
#                 adjacent_edges_removed[u].append(v)
#             else:
#                 adjacent_edges_removed[u] = [v]
#             if v in adjacent_edges_removed:
#                 adjacent_edges_removed[v].append(u)
#             else:
#                 adjacent_edges_removed[v] = [u]
#
#     # compute connected components T
#     _, cc_T = sp.csgraph.connected_components(T, directed=False, return_labels=True)
#
#     ################################
#     '''
# 	Connect via shortest path between nodes of removed edges
# 	'''
#     # sort edges by number of adjacent edges removed
#     nodes_with_adj_edges_removed = sorted(adjacent_edges_removed.keys(), key=lambda x: len(adjacent_edges_removed[x]),
#                                           reverse=True)
#
#
#     for u in nodes_with_adj_edges_removed:
#         if T.nnz == 2 * (T.shape[0] - 1):
#             break
#         dists,predecessors = sp.csgraph.shortest_path(ori_input_graph, indices=u,
#                                                 directed=False, unweighted=False, return_predecessors=True)
#         adjacent_edges_removed[u]=np.array(adjacent_edges_removed[u])[np.argsort(dists[adjacent_edges_removed[u]])]
#         for v in adjacent_edges_removed[u]:
#             if cc_T[u] == cc_T[v]:
#                 continue
#             # find shortest path between u and v
#             node=v
#             while predecessors[node]!=-9999:
#                 x, y = node, predecessors[node]
#                 if cc_T[x] != cc_T[y]:
#                     T[x, y] = T[y, x] = ori_input_graph[x, y]
#                     cc_T[np.where(np.logical_or(cc_T == cc_T[x], cc_T == cc_T[y]))] = min(cc_T[x], cc_T[y])
#                 node = y
#                 if cc_T[u] == cc_T[v]:
#                     break
#             if T.nnz == 2 * (T.shape[0] - 1):
#                 break
#
#
#     return T.tocsr()


def reconnect_via_shortespath(edges_weights,ori_input_graph):
    # dict with nodes as keys and adjacent node of edges removed as values
    adjacent_edges_removed = {}
    edges_k=list(edges_weights.keys())
    for e in edges_k:
        if e[0]>e[1]:
            continue
        if ori_input_graph[e[0],e[1]] == 0:
            #set to zero edges not in original graph
            del edges_weights[e]
            del edges_weights[e[1],e[0]]
            if e[0] in adjacent_edges_removed:
                adjacent_edges_removed[e[0]].append(e[1])
            else:
                adjacent_edges_removed[e[0]] = [e[1]]
            if e[1] in adjacent_edges_removed:
                adjacent_edges_removed[e[1]].append(e[0])
            else:
                adjacent_edges_removed[e[1]] = [e[0]]

    T=sp.dok_matrix((ori_input_graph.shape[0],ori_input_graph.shape[0]),dtype=ori_input_graph.dtype)
    T._update(edges_weights)
    # compute connected components T
    _, cc_T = sp.csgraph.connected_components(T, directed=False, return_labels=True)

    ################################
    '''
	Connect via shortest path between nodes of removed edges
	'''
    # sort edges by number of adjacent edges removed
    nodes_with_adj_edges_removed = sorted(adjacent_edges_removed.keys(), key=lambda x: len(adjacent_edges_removed[x]),
                                          reverse=True)
    
    if nk_available:
        # print('Using networkit')
        return reconnect_via_shortespath_nk(ori_input_graph,nodes_with_adj_edges_removed,adjacent_edges_removed,T,cc_T)
    else:
        # print('WARNING: Networkit not available, using scipy')
        return reconnect_via_shortespath_sp(ori_input_graph,nodes_with_adj_edges_removed,adjacent_edges_removed,T,cc_T)

def reconnect_via_shortespath_nk(ori_input_graph,nodes_with_adj_edges_removed,adjacent_edges_removed,T,cc_T):
    ori_input_graph_nk=nk.nxadapter.nx2nk(nx.from_scipy_sparse_array(ori_input_graph),weightAttr='weight')
    
    for u in nodes_with_adj_edges_removed:
        if T.nnz == 2 * (T.shape[0] - 1):
            break
        #compute shortest path tree
        dijks = nk.distance.Dijkstra(ori_input_graph_nk, u)
        dijks.run()
        # get distances
        dists=np.array(dijks.getDistances())
        
        # get predecessors for each node
        predecessors=dijks.getPredecessors
        
        adjacent_edges_removed[u]=np.array(adjacent_edges_removed[u])[np.argsort(-dists[adjacent_edges_removed[u]])]
        for v in adjacent_edges_removed[u]:
            if cc_T[u] == cc_T[v]:
                # suma+=1
                continue
            # find shortest path between u and v
            node=v
            while predecessors(node)!=[]:
                x, y = node, predecessors(node)[0]
                if cc_T[x] != cc_T[y]:
                    T[x, y] = T[y, x] = ori_input_graph[x, y]
                    if cc_T[x]<cc_T[y]:
                        cc_T[np.where(cc_T == cc_T[y])] = cc_T[x]
                    else:
                        cc_T[np.where(cc_T == cc_T[x])] = cc_T[y]
                node = y
                if cc_T[u] == cc_T[v]:
                    break
            if T.nnz == 2 * (T.shape[0] - 1):
                break

    return T.tocsr()
def reconnect_via_shortespath_sp(ori_input_graph,nodes_with_adj_edges_removed,adjacent_edges_removed,T,cc_T):
    for u in nodes_with_adj_edges_removed:
        if T.nnz == 2 * (T.shape[0] - 1):
            break
        dists,predecessors = sp.csgraph.shortest_path(ori_input_graph, indices=u,
                                                directed=False, unweighted=False, return_predecessors=True)
        adjacent_edges_removed[u]=np.array(adjacent_edges_removed[u])[np.argsort(-dists[adjacent_edges_removed[u]])]
        for v in adjacent_edges_removed[u]:
            if cc_T[u] == cc_T[v]:
                # suma+=1
                continue
            # find shortest path between u and v
            node=v
            while predecessors[node]!=-9999:
                x, y = node, predecessors[node]
                if cc_T[x] != cc_T[y]:
                    T[x, y] = T[y, x] = ori_input_graph[x, y]
                    if cc_T[x]<cc_T[y]:
                        cc_T[np.where(cc_T == cc_T[y])] = cc_T[x]
                    else:
                        cc_T[np.where(cc_T == cc_T[x])] = cc_T[y]
                node = y
                if cc_T[u] == cc_T[v]:
                    break
            if T.nnz == 2 * (T.shape[0] - 1):
                break


    return T.tocsr()


def mergeBP(T_dict,coords,BP,merging_neighBP,edge_flows,edge_weights,
            terminal_edge_sides=None,
            SP_non_possible_merges=None):
    # num_terminals=len(P)//2+1
    # if terminal_edge_sides is not None:
    #     differ = set()
    #     for node in SP_non_possible_merges[BP]:
    #         ee = min(BP, node), max(BP, node)
    #         if ee in terminal_edge_sides.keys():
    #             differ.update(terminal_edge_sides[ee][node])
    #             continue
    #         ee = min(merging_neighBP, node), max(merging_neighBP, node)
    #         if ee in terminal_edge_sides.keys():
    #             differ.update(terminal_edge_sides[ee][node])

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


    #         if terminal_edge_sides is not None:
    #             new_e=min(neighbor,merging_neighBP),max(neighbor,merging_neighBP)
    #             old_e=min(neighbor,BP),max(neighbor,BP)
    #             terminal_edge_sides[new_e]={}
    #
    #             terminal_edge_sides[new_e][neighbor]=terminal_edge_sides[old_e][neighbor].difference(differ)
    #
    #             if merging_neighBP >= num_terminals:
    #                 terminal_edge_sides[new_e][merging_neighBP] = terminal_edge_sides[old_e][BP].difference(differ)
    #             else:
    #                 terminal_edge_sides[new_e][merging_neighBP]=set([merging_neighBP]).difference(differ)
    #             del terminal_edge_sides[old_e]
    # if terminal_edge_sides is not None:
    #     e_merged=min(BP,merging_neighBP),max(BP,merging_neighBP)
    #     for i, neighbor in enumerate(T_dict[merging_neighBP]):
    #         if neighbor==BP:
    #             continue
    #         e = min(neighbor, merging_neighBP), max(neighbor, merging_neighBP)
    #
    #         terminals_reachable_from_e_merged=set(terminal_edge_sides[e_merged][merging_neighBP]).union(terminal_edge_sides[e_merged][BP])
    #         terminals_reachable_from_e_merged.difference_update(differ)
    #
    #         terminal_edge_sides[e][neighbor].intersection_update(terminals_reachable_from_e_merged)
    #
    #         terminal_edge_sides[e][merging_neighBP].intersection_update(terminals_reachable_from_e_merged)

def remove_node_from_Tdict(T_dict, node, edge_weights, edge_flows):
    for neigh in T_dict[node]:
        T_dict[neigh].remove(node)
        del edge_flows[(neigh, node)]
        del edge_flows[(node, neigh)]
        edge_weights.pop((node, neigh), None)
        edge_weights.pop((neigh, node), None)
    del T_dict[node]



#%%%%%%%%%%%%%%%%
# CONSTRAINED  BP REMOVAL
##################

def removeBP_constrained(T, coords,ori_input_graph,
                         order_criterium='closestterminals',merging_criterium='tryall',criterium_BP_position_update='median',
                        alpha=1,edge_flows=None,num_terminals=None):
    '''
    Given an adjacency matrix of a tree,T, representing final topology of a BOT problem, the function removes BP

    :param T: topology with BP (sparse matrix)
    :param coords: coordinates nodes
    :param order_criterium: how to order the BPs for merging
    :param merging_criterium: To which neighbor should the BP merged
    :param criterium_BP_position_update: How to update the BPs position
    :param alpha: Alpha value cost-> default 1
    :param return_nx: return networkx or sparse matrix. Default False-> returns sparse
    :param edge_restriction: list of tuples representing the edges that are allowed to be in the final tree. If None,
    all edges are allowed
    :param ori_input_graph: Adjacency sparse matrix over terminals with the entries representing the euclidean distance
            between terminal nodes. If given it forces that the removal of BP generates a spanning trees of this graph.
    :return:
    '''

    num_terminals = len(coords) // 2 + 1


    if sp.issparse(T):
        num_terminals = len(coords) // 2 + 1
        T_dict = {i: T.indices[T.indptr[i]:T.indptr[i + 1]].tolist() for i in range(T.shape[0])}
        edge_flows = dict(T.todok())
    elif isinstance(T, dict):
        T_dict=deepcopy(T)
        assert num_terminals is not None
        assert edge_flows is not None
    edge_weights = {}



    ori_input_graph_dict = {
        i: ori_input_graph.indices[ori_input_graph.indptr[i]:ori_input_graph.indptr[i + 1]].tolist() for i in
        range(ori_input_graph.shape[0])}


    Q=get_orderBPcls(order_criterium=order_criterium,coords=coords,T_dict=T_dict,num_terminals=num_terminals)

    altered_coords=coords.copy()
    # visited_BPs=set()
    merged_edges=[]
    succsesful_removal = remove_BP_constrained_backtracking(T_dict=T_dict, Q=Q, coords=coords,
                                                          altered_coords=altered_coords, merged_edges=merged_edges,
                                                          edge_weights=edge_weights, edge_flows=edge_flows,
                                                          num_terminals=num_terminals,
                                                          terminal_edge_sides=None,
                                                          ori_input_graph_dict=ori_input_graph_dict,
                                                          alpha=alpha, merging_criterium=merging_criterium,
                                                          criterium_BP_position_update=criterium_BP_position_update)

    assert succsesful_removal, 'No BP could be removed'
    T_sparse = sp.dok_matrix((num_terminals, num_terminals), dtype=T.dtype)
    T_sparse._update(edge_weights)
    return T_sparse.tocsr()
def remove_BP_constrained_backtracking(T_dict, Q, coords, altered_coords, merged_edges, edge_weights, edge_flows,
                                       num_terminals=None,
                                       terminal_edge_sides=None, ori_input_graph_dict=None, SP_non_possible_merges=None,
                                       alpha=1, merging_criterium='tryall', criterium_BP_position_update='median'):
    if SP_non_possible_merges is None:
        SP_non_possible_merges = {SP: set() for SP in range(2 * num_terminals - 2)}

    if terminal_edge_sides is None:
        # compute the terminal nodes at each side of each edge (only the ones that are connected via a path not containing other terminals)
        terminal_edge_sides = get_terminal_edge_sides(T_dict, num_terminals)
    while not Q.empty():
        BP,BP_value=Q.popitem()
        # if BP in visited_BPs:
        #     continue
        # visited_BPs.add(BP)

        neighbors_BP_sorted=choose_merging_neighBP(T_dict=T_dict,coords=altered_coords,BP=BP,
                                               edge_flows=edge_flows,
                                               alpha=alpha,merging_criterium=merging_criterium,
                                               terminal_edge_sides=terminal_edge_sides,
                                               ori_input_graph_dict=ori_input_graph_dict,
                                               SP_non_possible_merges=SP_non_possible_merges)
        valid_merge=False
        for merging_neighBP in neighbors_BP_sorted:
            if merging_neighBP in SP_non_possible_merges[BP] or (BP in SP_non_possible_merges[merging_neighBP]):
                continue
            if check_validity_edge_restriction((BP, merging_neighBP), T_dict,
                                               ori_input_graph_dict, terminal_edge_sides,
                                               SP_non_possible_merges=SP_non_possible_merges):

                neighs_BP= T_dict[BP]
                flow_BP_merging_neighBP=edge_flows[(BP,merging_neighBP)]
                side_BP_mergeneigBP=terminal_edge_sides[tuple(sorted((BP,merging_neighBP)))]



                # Realize the merging
                mergeBP(T_dict, coords, BP, merging_neighBP, edge_flows=edge_flows, edge_weights=edge_weights,
                        terminal_edge_sides=terminal_edge_sides,
                        SP_non_possible_merges=SP_non_possible_merges)
                merged_edges.append((BP, merging_neighBP))


                # remove BP
                if criterium_BP_position_update != 'average':
                    # average needs the BP to compute the new position still
                    remove_node_from_Tdict(T_dict=T_dict, node=BP, edge_weights=edge_weights, edge_flows=edge_flows)
                    if terminal_edge_sides is not None:
                        del terminal_edge_sides[tuple(sorted((BP, merging_neighBP)))]

                # if terminal_edge_sides is not None:
                #     #update terminal edge sides by removing the terminals at the side of a non possible merge
                #     terminal_edge_sides = get_terminal_edge_sides(T_dict, num_terminals,non_possible_merges=SP_non_possible_merges)

                # update position BP
                old_altered_coord_merged_BP= altered_coords[merging_neighBP]
                altered_coords = BP_position_update_for_removal(coords=altered_coords, T_dict=T_dict,
                                                                neigh_BP_merged=merging_neighBP,
                                                                num_terminals=num_terminals,
                                                                edge_flows=edge_flows, alpha=alpha,
                                                                criterium_BP_position_update=criterium_BP_position_update)

                # add new / update elements queue
                Q_values_before_update=Q.put(coords=altered_coords, T_dict=T_dict, merging_node=merging_neighBP,return_dict_changed=True)


                solved_succesfully= remove_BP_constrained_backtracking(T_dict=T_dict, Q=Q, coords=coords,
                                                          altered_coords=altered_coords, merged_edges=merged_edges,
                                                          edge_weights=edge_weights, edge_flows=edge_flows,
                                                          num_terminals=num_terminals,
                                                          terminal_edge_sides=terminal_edge_sides,
                                                          ori_input_graph_dict=ori_input_graph_dict,
                                                           SP_non_possible_merges=SP_non_possible_merges.copy(),
                                                          alpha=alpha, merging_criterium=merging_criterium,
                                                          criterium_BP_position_update=criterium_BP_position_update)
                if solved_succesfully:
                    return True
                else:
                    valid_merge=False
                    #undo the merge
                    merged_edges.pop()
                    altered_coords[merging_neighBP]=old_altered_coord_merged_BP
                    undomergeBP(T_dict, coords, BP, merging_neighBP, neighs_BP, flow_BP_merging_neighBP,
                                side_BP_mergeneigBP, edge_flows, edge_weights,
                                terminal_edge_sides=terminal_edge_sides,
                                )#recover T_dict, edge_flows, edge_weights, terminal_edge_sides
                    SP_non_possible_merges[BP].add(merging_neighBP)
                    SP_non_possible_merges[merging_neighBP].add(BP)
                    Q.restore_values(Q_values_before_update)
            else:
                SP_non_possible_merges[BP].add(merging_neighBP)
                SP_non_possible_merges[merging_neighBP].add(BP)
        if valid_merge==False:
            Q.restore_values({BP:BP_value})
            return False

    return True

def undomergeBP(T_dict,coords,BP,merging_neighBP,neighs_BP,flow_BP_merging_neighBP,
                side_BP_mergeneigBP,edge_flows,edge_weights,
            terminal_edge_sides=None,
            ):

    T_dict[BP]=[]
    for i,neighbor in enumerate(neighs_BP):
        d=np.linalg.norm(coords[BP]-coords[neighbor])
        T_dict[BP].append(neighbor)
        T_dict[neighbor].append(BP)
        if merging_neighBP!=neighbor:
            T_dict[neighbor].remove(merging_neighBP)
            T_dict[merging_neighBP].remove(neighbor)
            edge_flows[(neighbor,BP)]=edge_flows[(BP,neighbor)]=edge_flows[(merging_neighBP,neighbor)]
        else:
            edge_flows[(merging_neighBP,BP)]=edge_flows[(BP,merging_neighBP)]=flow_BP_merging_neighBP
        edge_weights[(neighbor,BP)]=edge_weights[(BP,neighbor)]=d
        if merging_neighBP!=neighbor:
            del edge_weights[(merging_neighBP,neighbor)]
            del edge_weights[(neighbor,merging_neighBP)]
            del edge_flows[(neighbor,merging_neighBP)]
            del edge_flows[(merging_neighBP,neighbor)]

        if terminal_edge_sides is not None:
            if merging_neighBP==neighbor:
                old_e = min(neighbor, BP), max(neighbor, BP)
                terminal_edge_sides[old_e]=side_BP_mergeneigBP
                continue
            new_e=min(neighbor,merging_neighBP),max(neighbor,merging_neighBP)
            old_e=min(neighbor,BP),max(neighbor,BP)
            terminal_edge_sides[old_e]={}

            terminal_edge_sides[old_e][neighbor]=terminal_edge_sides[new_e][neighbor]

            terminal_edge_sides[old_e][BP] = terminal_edge_sides[new_e][merging_neighBP]
            del terminal_edge_sides[new_e]
            # if neighbor in differ:








