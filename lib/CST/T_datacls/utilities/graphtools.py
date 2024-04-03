
import numpy as np
from scipy import sparse as sp
from scipy.sparse import diags
import networkx as nx
import scipy.sparse as sp

try:
    import networkit as nk
    nk_available=True
except:
    nk_available=False

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components

# from pynndescent import NNDescent
# import faiss

def adjacency2degree(A):
    """ Compute the degree matrix for a give adjacency matrix A"""
    return diags(np.asarray(A.sum(0)).reshape(-1), format="csc")


def adjacency2transition(A, D=None):
    """ Compute the transition matrix associated with the adjacency matrix A"""
    if D is None:
        D = adjacency2degree(A)
    return A *D.power(-1)


def get_Shortest_path_matrix(A,min_norm=1):
    if nk_available:
        if 'scipy.sparse' in str(type(A)):
            if min_norm!=1:
                A=A.power(min_norm)
            G = nk.nxadapter.nx2nk(nx.from_scipy_sparse_matrix(A, create_using=nx.Graph()), weightAttr='weight')
        elif 'networkx' in str(type(A)):
            if min_norm!=1:
                A=nx.adjacency_matrix(A)
                A=A.power(min_norm)
                A=nx.from_scipy_sparse_matrix(A, create_using=nx.Graph())
            G=nk.nxadapter.nx2nk(A, weightAttr='weight')
        else:
            if min_norm!=1:
                A=nk.algebraic.adjacencyMatrix(A, matrixType='sparse')
                A=A.power(min_norm)
                A=nk.nxadapter.nx2nk(nx.from_scipy_sparse_matrix(A, create_using=nx.Graph()), weightAttr='weight')
            G=A
        Distances_ = nk.distance.APSP(G)
        Distances_.run()
        D = np.array(Distances_.getDistances())
    else:
        if 'scipy.sparse' in str(type(A)):
            pass
        else:
            A=nx.adjacency_matrix(A.to_undirected())
        if min_norm!=1:
            A=A.power(min_norm)
        D = sp.csgraph.floyd_warshall(A, directed=False, unweighted=False)
    if min_norm!=1:
        D=D**(1/min_norm)
    return D


#%%


def compute_knn_graph(coords, num_neighs,knn_type='symmetric'):
    '''
    Computes the knn graph of the points in coords.
    :param coords: coordinates of the points
    :param num_neighs: number of neighbors to consider
    :param knn_type: type of knn graph, possible types are symmetric, mutual and directed. Default is symmetric.
        If directed, the graph is not symmetric.
        If symmetric, the graph is symmetric and an edge is present if one of the two points is a nearest neighbor.
        If mutual, an edge is present if and only if the two points are mutual nearest neighbors. The graph is symmetric.
    :return:
    '''
    knn_graph = kneighbors_graph(coords, n_neighbors=num_neighs, mode='distance')
    if knn_type=='symmetric':
        return knn_graph.maximum(knn_graph.T)
    elif knn_type=='mutual':
        knn_graph=knn_graph.minimum(knn_graph.T)
        knn_graph.eliminate_zeros()
        return knn_graph
    return knn_graph

    # Faiss implemetation
    # index = faiss.IndexFlatL2(coords.shape[1])
    # index.add(coords.astype(np.float32))
    # knn_dists, knn_indices = index.search(coords.astype(np.float32), num_neighs + 1)
    # num_data_points = coords.shape[0]
    # row_indices = np.repeat(np.arange(num_data_points), num_neighs + 1)
    # col_indices = knn_indices.flatten()
    # distances_data = knn_dists.flatten()
    # knn_graph= csr_matrix((distances_data, (row_indices, col_indices)), shape=(num_data_points, num_data_points))
    # if knn_type=='symmetric':
    #     return knn_graph.maximum(knn_graph.T)
    # elif knn_type=='mutual':
    #     knn_graph=knn_graph.minimum(knn_graph.T)
    #     knn_graph.eliminate_zeros()
    #     return knn_graph
    # return knn_graph

    # # # Create an instance of the NNDescent class
    # nnd = NNDescent(coords, n_neighbors=num_neighs)
    #
    # # Build the KNN graph
    # knn_indices, knn_dists = nnd.neighbor_graph
    #
    # # Create a CSR matrix from knn_indices and knn_dists
    # num_data_points = coords.shape[0]
    # row_indices = np.repeat(np.arange(num_data_points), num_neighs)
    # col_indices = knn_indices.flatten()
    # distances_data = knn_dists.flatten()
    #
    # knn_graph= sp.csr_matrix((distances_data, (row_indices, col_indices)), shape=(num_data_points, num_data_points))
    # if knn_type=='symmetric':
    #     return knn_graph.maximum(knn_graph.T)
    # elif knn_type=='mutual':
    #     knn_graph=knn_graph.minimum(knn_graph.T)
    #     knn_graph.eliminate_zeros()
    #     return knn_graph
    # return knn_graph

def is_graph_connected(graph):
    num_components, _ = connected_components(graph, directed=False)
    return num_components == 1

def ensure_connected_knn_graph(coords,num_neighs=None,connect_with_mST=False,
                               knn_type='symmetric'):
    '''
    Ensure that the knn graph is connected by increasing the number of neighbors till the graph is connected.
    If connect_with_mST is True, then the graph is connected by adding edges to the minimum spanning tree of the graph.
    :param coords:
    :param num_neighs:
    :param connect_with_mST:
    :param knn_type: type of knn graph, possible types are symmetric, mutual and directed. Default is symmetric.
        If directed, the graph is not symmetric.
        If symmetric, the graph is symmetric and an edge is present if one of the two points is a nearest neighbor.
        If mutual, an edge is present if and only if the two points are mutual nearest neighbors. The graph is symmetric.
    :return:
    '''
    if num_neighs is None:
        num_neighs = min(max(int(np.log(len(coords))), 3), len(coords)-1)
    connected = False
    iteration=0
    while not connected:
        knn_graph = compute_knn_graph(coords, num_neighs,knn_type=knn_type)
        if connect_with_mST and iteration==0:
            knn_graph_init= knn_graph.copy()

        if is_graph_connected(knn_graph):
            connected = True
        else:
            num_neighs *= 2
        iteration+=1
        if connect_with_mST:
            # to compute the mST we use the symmetric knn graph
            knn_type='symmetric'

    if connect_with_mST and iteration!=1:
        #add edges of the mST to the knn graph
        mST=sp.csgraph.minimum_spanning_tree(knn_graph)
        mST=mST+mST.T
        knn_graph=knn_graph_init.maximum(mST)
    
    if knn_type=='symmetric':
        return knn_graph.maximum(knn_graph.T)
    elif knn_type=='mutual':
        return knn_graph.minimum(knn_graph.T)
    elif knn_type=='directed':
        return knn_graph
    else:
        raise NotImplementedError('knn_type %s not implemented, possible types are symmetric, mutual and directed'%(knn_type))


#%%
'''
FUNCTIONS TO COMPUTE THE EDGES CENTRALITIES OF A TREE (WITHOUT STEINER POINTS)
'''

def count_nodes_sides(adj_list, cur, par,nodes_side_count=None):
    '''
    Helper recursive function to count the number of nodes at one side of each edge of the tree.

    Parameters
    ----------
    adj_list : TYPE
        DESCRIPTION.
    cur : int
        current node.
    par : int
        parent node.
    nodes_side_count : dict of tuples indicating the nodes at each side. Keys are the edges of the tree.
        DESCRIPTION. The default is None.

    Returns
    -------
    nodes_side_count : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    '''
    if nodes_side_count==None:
        nodes_side_count={}
        for v in adj_list.keys():
            for u in adj_list[v]:
                e=tuple(sorted((u,v)))
                nodes_side_count[e]=0

    e=tuple(sorted((par,cur)))
    #If current nodes is leaf node and is not the node provided by the calling function then return 1
    if len(adj_list[cur]) == 1 and par != 0:
        nodes_side_count[e] = 1
        return nodes_side_count,nodes_side_count[e]
    count = 1
    #count the number of nodes recursively for each neighbor of current node.
    for neighbor in adj_list[cur]:
        if neighbor != par:
            nodes_side_count,count_aux= count_nodes_sides(adj_list, neighbor, cur,nodes_side_count)
            count+=count_aux

    # while returning from recursion assign the result obtained in the edge[][] matrix.

    nodes_side_count[e] = count
    return nodes_side_count,nodes_side_count[e]


def compute_nodes_side_count(tree):
    '''
    Count nodes at one side of the edges of a tree. The side is the one containing a certain but arbitrary leaf node.

    Parameters
    ----------
    tree : TYPE
        DESCRIPTION.

    Returns
    -------
    nodes_side_count : dict of tuples, keys are edges and values are the number of nodes at  one side of the edge.
        DESCRIPTION.

    '''
    isnk=is_networkit_graph(tree)

    if isnk:
        adj_list={node: [neigh for neigh in tree.iterNeighbors(node)] for node in tree.iterNodes()}
        #initialize the nodes_side_count dictionary
        nodes_side_count={tuple(sorted(e)):0 for e in tree.iterEdges()}
        #find a leaf node
        for node in tree.iterNodes():
            if tree.degree(node)==1:
                par=node
                cur=next(tree.iterNeighbors(node))
                break
    else:
        adj_list=nx.to_dict_of_lists(tree)
        # initialize the nodes_side_count dictionary
        nodes_side_count={tuple(sorted(e)):0 for e in tree.edges()}
        #find a leaf node
        for node in tree:
            if tree.degree(node)==1:
                par=node
                cur=next(tree.neighbors(node))
                break
    nodes_side_count,_=count_nodes_sides(adj_list,cur,par,nodes_side_count)
    return nodes_side_count

#
# def get_edge_centrality_components_method(tree):
#     A=nx.linalg.graphmatrix.adjacency_matrix(tree,weight=None).tolil()
#     centrality=[1]*tree.number_of_edges()
#     n=tree.number_of_nodes()
#     for i,e in enumerate(tree.edges()):
#         A[e[0],e[1]]=0
#         cc,labels=sp.csgraph.connected_components(A)
#         x=np.count_nonzero(cc)
#         centrality[i]*=x*(n-x)
#         A[e[0],e[1]]=1
#
#     return centrality


def centrality_weights_tree(tree,norm=True,max_width=1):
    '''
     Computes the edge centrality of the edges in a tree. This is used for the
    widths when plotting. The edge centrality on a tree coincides with the number
    of nodes to the left of the edge times the number of nodes to the right of the
    edge of e: c(e)=l(e)*r(e)

    Parameters
    ----------
    tree : TYPE
        DESCRIPTION.
    norm : TYPE, optional
        DESCRIPTION. The default is True.
    max_width : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    nodes_side_count : TYPE
        DESCRIPTION.
    widths : TYPE
        DESCRIPTION.

    '''
    if 'scipy.sparse' in str(type(tree)):
        if not tree.has_sorted_indices:
            tree.sort_indices()
        try:
            tree=nx.from_scipy_sparse_matrix(tree,create_using=nx.Graph())
        except AttributeError:
            tree = nx.from_scipy_sparse_array(tree, create_using=nx.Graph())
    if nx.number_connected_components(tree)==1:
        if nk_available:
            widths=nk_edge_centrality(tree,None)
        else:
            nodes_side_count=compute_nodes_side_count(tree)
            widths=compute_centrality_edges(tree,nodes_side_count)

        if norm==True:
            widths=max_width*np.array(widths)/max(widths)
        return widths
    else:
        num2edge=dict(enumerate(tree.edges()))
        edge2num={num2edge[i]:i for i in num2edge.keys()}
        widths=[0]*len(edge2num)
        for nodes_subtree in nx.connected_components(tree):
            if len(nodes_subtree)==1:
                continue
            subtree=nx.subgraph(tree,nodes_subtree)
            nodes_side_count=compute_nodes_side_count(subtree)
            widths_subtree=compute_centrality_edges(subtree,nodes_side_count)
            if norm==True:
                widths_subtree=max_width*np.array(widths_subtree)/max(widths_subtree)
            for i, e in enumerate(subtree.edges):
                try:
                    widths[edge2num[e]]=widths_subtree[i]
                except:
                    widths[edge2num[(e[1],e[0])]]=widths_subtree[i]


    return  np.array(widths)


def compute_centrality_edges(tree,nodes_side_count,alpha=1,weight=False):
    '''
    Computes the edge centrality of the edges in a tree. Given the number of nodes in one side of the edge, the edge
    centrality on a tree coincides with the number of nodes to the left of the edge times the number of nodes to the
    right of the edge of e: c(e)=l(e)*r(e)

    :param tree:
    :param nodes_side_count:
    :param alpha:
    :param weight:
    :return:
    '''
    widths=[]
    if is_networkit_graph(tree):
        n=tree.numberOfNodes()

        for e in tree.iterEdges():
            e=tuple(sorted(e))
            n1=nodes_side_count[e]
            n2=n-n1
            if weight:
                w=tree.weight(e[0],e[1])
            else:
                w=1
            if alpha=='min':
                widths.append(w * min(n1, n2))
            else:
                widths.append(w*(n1*n2)**alpha)
    else:
        n=tree.number_of_nodes()

        for e in tree.edges():
            e=tuple(sorted(e))
            n1=nodes_side_count[e]
            n2=n-n1
            if weight:
                w=tree.get_edge_data(e[0],e[1])['weight']
            else:
                w=1
            if alpha == 'min':
                widths.append(w * min(n1, n2))
            else:
                widths.append(w * (n1 * n2) ** alpha)
    return widths


def nk_edge_centrality(T,weightattribute='weight'):
    if nk_available:
        if not is_networkit_graph(T):
            tree=nk.nxadapter.nx2nk(T,weightAttr=weightattribute)
        else:
            tree=T

        tree.indexEdges()
        EC=np.array(nk.centrality.Betweenness(tree,normalized=False,computeEdgeCentrality=True).run().edgeScores())
        # return EC
        EC_reordered=np.empty(EC.shape)
        contador=0
        for i,edges in enumerate(zip(T.edges(),tree.iterEdges())):
            e,e2=edges
            # assert(e2 in T.edges())
            # if e!=e2:
            #     contador+=1
            #     print(contador,e,e2)
            assert(e==e2)
            idx=tree.edgeId(*e)
            EC_reordered[i]=EC[idx]
        return EC_reordered


def tree_node_centrality(tree):
    if is_networkit_graph(tree):
        return tree_node_centrality_nk(tree)
    else:
        return tree_node_centrality_nx(tree)


def tree_node_centrality_nx(tree):
    centrality=nx.betweenness_centrality(tree,normalized=False)
    n=tree.number_of_nodes()
    nodes_centrality=np.zeros(n)
    list_nodes=list(tree.nodes())
    for node in tree:
        u= list_nodes.index(node)
        nodes_centrality[u]=n-1+centrality[u]
    return nodes_centrality


def tree_node_centrality_nk(tree):
    centrality=nk.centrality.Betweenness(tree,normalized=False).run().scores()
    n=tree.numberOfNodes()
    nodes_centrality=np.zeros(n)
    for u in tree.iterNodes():
        nodes_centrality[u]=n-1+centrality[u]
    return nodes_centrality


def Wiener_index(tree,alpha=1):
    '''
    It computes sum_e w(e)*(l(e)*r(e))^a
    where l(e) nodes to the left of edge e
    and r(e) nodes to the right of edge e

    Returns
    -------
    None.

    '''
    if 'scipy.sparse' in str(type(tree)):
        tree = nx.from_scipy_sparse_matrix(tree, create_using=nx.Graph())
    nodes_side_count=compute_nodes_side_count(tree)
    widths=compute_centrality_edges(tree,nodes_side_count,alpha,weight=True)
    return sum(widths)


def is_networkit_graph(G):
    try:
        G.numberOfEdges()
        return True
    except:
        return False