from copy import deepcopy
import scipy.sparse as sp
import numpy as np
def get_terminal_edge_sides(T_dict_,num_terminals=None,non_possible_merges=None):
    '''
    Computes a dictionary where the keys are the edges of the tree and the values are dictionaries whose keys are the
    nodes of the edge. The values of the inner dictionary is a set containing the terminal nodes connected to the node,
    indicated by the key of the inner dictionary, via a path not containing other terminals.

    Remark: the edges are represented by tuples (i,j) with i<j.


    :param T_dict:
    :param num_terminals:
    :return: dict of dicts
    '''
    terminal_edge_sides={}


    # Remove non possible merges from T_dict by removing the edges between the nodes that cannot be merged
    if non_possible_merges is not None:
        # copy T_dict to avoid modifying the original one
        T_dict = deepcopy(T_dict_)
        for node in non_possible_merges.keys():
            if node not in T_dict.keys():
                continue
            for non_possible_merge_node in non_possible_merges[node]:
                if node>non_possible_merge_node or non_possible_merge_node not in T_dict.keys():
                    continue
                if non_possible_merge_node in T_dict[node]:
                    T_dict[node].remove(non_possible_merge_node)
                    T_dict[non_possible_merge_node].remove(node)
    else:
        #in this case T_dict is not modified
        T_dict=T_dict_

    if num_terminals is None:
        num_terminals=len(T_dict)//2-1

    visited = {node: False for node in T_dict.keys()}
    for i in T_dict.keys():
        for j in T_dict[i]:
            if i>j:
                continue
            terminal_edge_sides[(i,j)]= {}
            visited[i]=visited[j]=True
            terminal_edge_sides[(i,j)][i]=dfs_terminal_edge_side(i, T_dict, visited.copy(), num_terminals, terminal_edge_sides)
            terminal_edge_sides[(i,j)][j]=dfs_terminal_edge_side(j, T_dict, visited.copy(), num_terminals, terminal_edge_sides)
            visited[i] = visited[j] = False
    return terminal_edge_sides


def dfs_terminal_edge_side(node, graph, visited, num_terminals, terminal_edge_sides=None):
    """
    Depth-first search function that explores each branch till a terminal point is found. The output is a set of all
    the terminals that are connected to the input node, without crossing any other terminal.
    """
    visited[node] = True
    if node<num_terminals:
        return set([node])
    else:
        connected_components = set()

    for neighbor in graph[node]:
        if not visited[neighbor]:
            if terminal_edge_sides is not None:
                u,v=min(node,neighbor),max(node,neighbor)
                if (u,v) in terminal_edge_sides.keys() and neighbor in terminal_edge_sides[(u,v)].keys():
                    connected_components.update(terminal_edge_sides[(u,v)][neighbor])
                    visited[neighbor] = True
                    continue

            connected_components.update(dfs_terminal_edge_side(neighbor, graph, visited,num_terminals,terminal_edge_sides))

    return connected_components


def check_validity_edge_restriction(e, T_dict, ori_input_graph_dict,
                                    terminal_edge_sides, SP_non_possible_merges=None):
    '''
    Checks if the edges in edge_restriction are in the tree T_dict

    :param T_dict:
    :param ori_input_graph_dict:
    :param edge_restriction:
    :return:
    '''
    e=min(e),max(e)

    def check(x, y):
        '''
        Check if exists a node in the side of y that is connected to at least one node of each side
        of the neighbors of x
        (merge in direction x->y) This implies that the edges nodes neighboring x must not necessarily be merged
        with the resulting merged node (xy)

        original graph:
         a----c
         |   /|
         | /  |
         b    d


        T_dict:                                     valid merge

       a            c                              a      c
        \          /                                \    /
         \        /                                  \  /
          x----->y                       ->           xy
         /        \                                  /  \
        /          \                                /    \
       b            d                              b      d

        Valid merge because one node in the side of y (c) is connected in the original graph
        to one node of each side of the other edges neighboring x (a and b).
        If the merge is valid, it implies the edges a--xy and b--xy could survive the whole process.
        :param x:
        :param y:
        :return:
        '''
        side_y = terminal_edge_sides[e][y]
        #list of sets of nodes in the side of the neighbors of x
        sides_neigh_x= []
        if x < len(ori_input_graph_dict):
            intersection_neighborhoods_sidex = set(ori_input_graph_dict[x]).union([x])
        else:
            intersection_neighborhoods_sidex = set(range(len(ori_input_graph_dict)))
            for neigh_x in T_dict[x]:
                if neigh_x == y:
                    continue
                e_ = min(neigh_x, x), max(neigh_x, x)
                union_aux = set()
                # construct the union of the neighborhood of the side of the neigh_x in the edge e_
                sides_neigh_x.append(terminal_edge_sides[e_][neigh_x])
                if neigh_x<len(ori_input_graph_dict):
                    union_aux=set([neigh_x]).union(ori_input_graph_dict[neigh_x])
                else:
                    for node in terminal_edge_sides[e_][neigh_x]:
                        union_aux.update(set([node]).union(ori_input_graph_dict[node]))
                # intersect the union with the other neighborhoods of the sides of the other neighbors of x
                intersection_neighborhoods_sidex = intersection_neighborhoods_sidex.intersection(union_aux)

        # potential terminals to which the merged new node xy could be collapsed to
        potential_terminals= side_y.intersection(intersection_neighborhoods_sidex)
        if SP_non_possible_merges is None:
            return potential_terminals

        #remove potential terminals that imply a merge that is not possible. This is done by checking if there is a
        #path from the potential terminal to any of the nodes in each side of a neighbor  of x that crosses more than
        # two non possible merges
        for node in potential_terminals.copy():
            for side_neigh_x in sides_neigh_x:
                side_neigh_x=side_neigh_x.intersection(ori_input_graph_dict[node])
                #find path from node to all nodes on the side of x:
                paths=find_paths_to_sinks(T_dict,node, side_neigh_x)
                non_reachable_side=True
                for path in paths:
                    if count_number_non_possible_merges_in_path(path, SP_non_possible_merges)<=1:
                        non_reachable_side=False
                        break

                if non_reachable_side:
                    potential_terminals.remove(node)
                    break
        return potential_terminals



    num_terminals=len(ori_input_graph_dict)
    if len(terminal_edge_sides[e][e[0]]) <= len(terminal_edge_sides[e][e[1]]):
        u,v=e[0],e[1]
    else:
        v,u=e[0],e[1]


    if u<len(ori_input_graph_dict) and v<len(ori_input_graph_dict):
        # if both nodes are terminals, check if they are connected in the original graph
        if v in ori_input_graph_dict[u]:
            return True
        else:
            return False

    if SP_non_possible_merges is None:
        if u<num_terminals:
            return len(check(v,u))>0
        return len(check(u,v))>0 or len(check(v,u))>0

    if u >= len(ori_input_graph_dict):
        uv_may_merge2= check(u, v)
    else:
        uv_may_merge2 = set()

    if v >= len(ori_input_graph_dict):
        vu_may_merge2= check(v, u)
    else:
        vu_may_merge2 = set()
    if len(uv_may_merge2) == 0 and len(vu_may_merge2) == 0:
        return False
    elif u < len(ori_input_graph_dict) or v < len(ori_input_graph_dict):
        # if one of the nodes is a terminal then no need to update the non possible merges
        return True

    SP_non_possible_merges_cp=deepcopy(SP_non_possible_merges)
    exists_valid_path=update_non_possible_merges(u,v, uv_may_merge2.union(vu_may_merge2), T_dict, SP_non_possible_merges_cp)
    if not exists_valid_path:
        return False

    for k in SP_non_possible_merges_cp.keys():
        SP_non_possible_merges[k]=SP_non_possible_merges_cp[k]
    return True

    # if not any([SP_non_possible_merges_cp[k].issuperset(T_dict[k]) for k in T_dict.keys() if  k not in (u,v)]):
    #     union_neighs = set(T_dict[u]).union(T_dict[v]).difference(set([u, v]))
    #     if SP_non_possible_merges_cp[u].issuperset(union_neighs):
    #         return False
    #     if SP_non_possible_merges_cp[v].issuperset(union_neighs):
    #         return False
    #
    #
    #     for k in SP_non_possible_merges_cp.keys():
    #         SP_non_possible_merges[k]=SP_non_possible_merges_cp[k]
    #     # update_terminal_edge_sides_func(u,v, SP_non_possible_merges,terminal_edge_sides)
    #     return True
    # else:
    #     return False

def update_non_possible_merges(x,y, xyx_may_merge2, T_dict, SP_non_possible_merges):
    if xyx_may_merge2 is None:
        return
    paths=find_paths_to_sinks(T_dict,x,xyx_may_merge2)
    if paths is None:
        return
    exists_valid_path=False
    SP_possible_merges_in_path = set([x,y])

    for path in paths:
        if not check_path_is_valid(path, SP_non_possible_merges):
            continue
        exists_valid_path=True
        for i in range(len(path) - 1):
            node1,node2=path[i],path[i+1]
            if (node1,node2)==(x,y) or (node1,node2)==(y,x):
                continue
            SP_possible_merges_in_path.add(node2)
            break
    # add non possible merges neighboring x and y
    SP_non_possible_merges[y]=SP_non_possible_merges[x]=(set(T_dict[x]).union(T_dict[y])).difference(SP_possible_merges_in_path)
    # add opposite direction non possible merges
    for node in SP_non_possible_merges[y]:
        SP_non_possible_merges[node].update([x,y])
    return exists_valid_path
    # if xy_may_merge2 is None:
    #     return
    # paths=find_paths_to_sinks(T_dict,x,xy_may_merge2)
    # if paths is None:
    #     return

    # SP_possible_merges_in_path={}
    # SP_non_possible_merges_in_path = {}
    # for path in paths:
    #     is_valid=True
    #     #check all steps in path are valid merges
    #     for i in range(len(path)-1):
    #         node1,node2=path[i],path[i+1]
    #         if node1 in SP_non_possible_merges.keys() and node2 in SP_non_possible_merges[node1]:
    #             is_valid=False
    #             break
    #         if node2 in SP_non_possible_merges.keys() and node1 in SP_non_possible_merges[node2]:
    #             is_valid=False
    #             break
    #     if not is_valid:
    #         for i in range(len(path) - 1):
    #             node1, node2 = path[i], path[i + 1]
    #             if node1 not in SP_possible_merges_in_path.keys():
    #                 SP_non_possible_merges_in_path[node1] = {node2}
    #             else:
    #                 SP_non_possible_merges_in_path[node1].add(node2)
    #
    #             if node2 not in SP_possible_merges_in_path.keys():
    #                 SP_non_possible_merges_in_path[node2] = {node1}
    #             else:
    #                 SP_non_possible_merges_in_path[node2].add(node1)
    #
    #     else:
    #         for i in range(len(path)-1):
    #             node1,node2=path[i],path[i+1]
    #             if node1 not in SP_possible_merges_in_path.keys():
    #                 SP_possible_merges_in_path[node1]={node2}
    #             else:
    #                 SP_possible_merges_in_path[node1].add(node2)
    #
    #             if node2 not in SP_possible_merges_in_path.keys():
    #                 SP_possible_merges_in_path[node2]={node1}
    #             else:
    #                 SP_possible_merges_in_path[node2].add(node1)
    # for k in SP_non_possible_merges_in_path.keys():
    #     if k in SP_possible_merges_in_path:
    #         SP_non_possible_merges_in_path[k].difference(SP_possible_merges_in_path[k])
    #
    # for k in SP_possible_merges_in_path.keys():
    #     if k!=x and k in SP_non_possible_merges.keys():
    #         SP_non_possible_merges[k].update(set(T_dict[k]).difference(SP_possible_merges_in_path[k]))
    # if len(SP_possible_merges_in_path.keys())==0:
    #     k=paths[0][1]
    #     SP_non_possible_merges[k].update(set(T_dict[k]).difference([x]))
    #
    # for k in SP_non_possible_merges_in_path.keys():
    #     if k not in SP_non_possible_merges.keys() or k==x:
    #         continue
    #     SP_non_possible_merges[k].update(SP_non_possible_merges_in_path[k])
def find_paths_to_sinks(tree, source, sinks):
    if not isinstance(sinks, set):
        sinks = {sinks}  # Convert single node to a set containing that node

    def dfs(node, parent, current_path):
        current_path.append(node)

        if node in remaining_sinks:
            all_paths.append(list(current_path))
            remaining_sinks.remove(node)  # Mark this node as reached

        if not remaining_sinks:  # If all sinks have been reached, stop the search
            return

        for neighbor in tree[node]:
            if neighbor != parent:
                dfs(neighbor, node, current_path)

        current_path.pop()

    all_paths = []
    remaining_sinks = set(sinks)  # Make a copy of sinks to keep track of unreached sinks
    dfs(source, parent=None, current_path=[])

    return all_paths if all_paths else None


# def update_terminal_edge_sides_func(x, y, SP_non_possible_merges, terminal_edge_sides):
#
#     e=min(x, y),max(x, y)
#     if x in SP_non_possible_merges.keys():
#         for neigh_x in SP_non_possible_merges[x]:
#             if neigh_x==y:
#                 continue
#             e_=min(x, neigh_x),max(x, neigh_x)
#             if e_ in terminal_edge_sides:
#                 terminal_edge_sides[e][x].difference_update(terminal_edge_sides[e_][neigh_x])
#
#     if y in SP_non_possible_merges.keys():
#         for neigh_y in SP_non_possible_merges[y]:
#             if neigh_y==x:
#                 continue
#             e_=min(y, neigh_y),max(y, neigh_y)
#             if e_ in terminal_edge_sides:
#                 terminal_edge_sides[e][y].difference_update(terminal_edge_sides[e_][neigh_y])

def check_path_is_valid(path,non_possible_merges):
    '''checks if all edges in path are valid merges'''
    for i in range(len(path)-1):
        if path[i+1] in non_possible_merges[path[i]]:
            return False
    return True

def remove_non_possible_merges(T_dict_,non_possible_merges,return_copy=True):
    if return_copy:
        T_dict = deepcopy(T_dict_)
    else:
        T_dict=T_dict_
    for node in non_possible_merges.keys():
        if node not in T_dict.keys():
            continue
        for non_possible_merge_node in non_possible_merges[node]:
            if node > non_possible_merge_node or non_possible_merge_node not in T_dict.keys():
                continue
            if non_possible_merge_node in T_dict[node]:
                T_dict[node].remove(non_possible_merge_node)
                T_dict[non_possible_merge_node].remove(node)
    return T_dict


def count_number_non_possible_merges_in_path(path, SP_non_possible_merges):
    counter=0
    for i in range(len(path)-1):
        if path[i+1] in SP_non_possible_merges[path[i]]:
            counter+=1

    return counter



def constrain2edge_index(T,num_terminals,ori_input_graph):
    '''
    Checks if the edges are valid (in the sense they may originate a valid subtree of the original input graph).
    If not, the edge is removed from the tree, and the nodes of each side are connected via the edges of the mST
    :param T:
    :param num_terminals:
    :param ori_input_graph:
    :return:
    '''


    # Convert sparse matrix to adjacency list representation.
    T_dict={i:T.indices[T.indptr[i]:T.indptr[i+1]].tolist() for i in range(T.shape[0])}
    ori_input_graph_dict={i:ori_input_graph.indices[ori_input_graph.indptr[i]:ori_input_graph.indptr[i+1]].tolist() for i in range(ori_input_graph.shape[0])}

    # compute the terminal nodes at each side of each edge (only the ones that are connected via a path not containing other terminals)
    terminal_edge_sides=get_terminal_edge_sides(T_dict,num_terminals)

    edges_T=[(u,v) for u,v in zip(*sp.triu(T).nonzero())]
    T=T.tolil()
    #dict with nodes as keys and adjacent node of edges removed as values
    # adjacent_edges_removed={}
    for e in edges_T:
        u,v=e
        #check if each terminal in side1 is directly connected to at least one terminal in side2 by an edge
        valid_edge = check_validity_edge_restriction(e,T_dict,ori_input_graph_dict,terminal_edge_sides)
        if not valid_edge:
            T[u,v]=T[v,u]=0
            # if u in adjacent_edges_removed:
            #     adjacent_edges_removed[u].append(v)
            # else:
            #     adjacent_edges_removed[u]=[v]
            # if v in adjacent_edges_removed:
            #     adjacent_edges_removed[v].append(u)
            # else:
            #     adjacent_edges_removed[v]=[u]

    # compute connected components T
    _,cc_T = sp.csgraph.connected_components(T, directed=False, return_labels=True)

    ################################


    ################################
    '''
    Connect with edges of the mST of original graph
    '''
    #compute mst of original graph
    mST_ori_graph=sp.csgraph.minimum_spanning_tree(ori_input_graph)

    #use edges from mST_ori_graph to reconnect T and form a new tree
    edges_mST_ori_graph=[(u,v) for u,v in zip(*mST_ori_graph.nonzero())]
    for e in edges_mST_ori_graph:
        u,v=e
        if cc_T[u]!=cc_T[v]:
            T[u,v]=T[v,u]=ori_input_graph[u,v]
            cc_T[np.where(np.logical_or(cc_T==cc_T[u], cc_T==cc_T[v]))]=min(cc_T[u],cc_T[v])
        if T.nnz==2*(T.shape[0]-1):
            break
    ##########################################
    idxs_largest_cc=np.where(cc_T==cc_T[0])[0]
    T=T.tocsr()
    T=T[idxs_largest_cc,:][:,idxs_largest_cc]

    return T