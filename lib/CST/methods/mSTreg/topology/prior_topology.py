import warnings
import numpy as np
import scipy.sparse as sp
import networkx as nx
from numba import njit
from ..topology import topology
from scipy.cluster.hierarchy import  linkage

from ..geometry_optimization.fast_optimizer_general import fast_optimize_general

try:
    import networkit as nk
    networkit_installed=True
except ImportError:
    warnings.warn('networkit not installed. Using networkx')
    networkit_installed=False

def sparsemat2fulltopoadj(T, coords=None, return_topo=True):
    '''
    decouples BP in topology with no BP

    :param T: sparse matrix-> adjacency matrix of current topology without BP
    :param coords: Coordinates terminals
    :param return_topo: if True returns as topology class, otherwise return adj
    :return:
    '''
    if coords is None:
        return sparsemat2fulltopoadj_default(T,return_topo)
    else:
        return sparsemat2fulltopoadj_linkage(T, coords, return_topo)
def sparsemat2fulltopoadj_default( T , return_topo=True):
    label_max = T.shape[0]-1
    degrees_MST = T.astype(bool).sum(1).A
    T_nx = nx.from_scipy_sparse_matrix(T)
    for node in np.where(degrees_MST != 1)[0]:
        label_max += 1
        neighbours = list(nx.neighbors(T_nx, node))

        neighbours.append(node)
        T_nx.remove_node(node)
        T_nx.add_edge(neighbours[0], label_max)
        T_nx.add_edge(neighbours[1], label_max)

        for nd in neighbours[2:-1]:
            T_nx.add_edge(label_max, label_max + 1)
            T_nx.add_edge(label_max + 1, nd)
            label_max += 1

        T_nx.add_edge(label_max, neighbours[-1])

    adj=nx_to_adj(T_nx)
    if return_topo:
        return topology(adj=adj)
    else:
        return adj


def sparsemat2fulltopoadj_linkage(T, coords, return_topo=True):
    '''

    :param T: sparse matrix-> adjacency matrix of current topology without BP
    :param coords: Coordinates terminals
    :param return_topo: if True returns as topology class, otherwise return adj
    :return:
    '''
    label_max = T.shape[0] - 1
    degrees_MST = T.astype(bool).sum(1).A
    T_nx = nx.from_scipy_sparse_matrix(T)
    coords_ = np.concatenate([coords, np.zeros(shape=(coords.shape[0] - 2, coords.shape[1]))])
    for node in np.where(degrees_MST != 1)[0]:
        label_max += 1
        neighbours = list(nx.neighbors(T_nx, node))
        num_neighbors = len(neighbours)
        if num_neighbors == 2:
            T_nx.remove_node(node)
            T_nx.add_edge(neighbours[0], label_max)
            T_nx.add_edge(neighbours[1], label_max)
            T_nx.add_edge(label_max, node)
        else:


            neighbours.extend([node])
            Z = linkage(coords_[neighbours], 'single', optimal_ordering=True)
            # dist_to_neighbors=np.linalg.norm(P[node,:]-P[neighbours],axis=1)

            neighbours.extend(label_max + np.arange(num_neighbors - 1))
            label_max += num_neighbors - 2

            T_nx.remove_node(node)
            bp_dendrogram_counter = num_neighbors
            repre_node = {i: i for i in neighbours}
            for z in Z[:-1]:
                bp_dendrogram_counter += 1
                idx_nd1 = neighbours[int(z[0])]
                idx_nd2 = neighbours[int(z[1])]
                idx_new_nd = neighbours[bp_dendrogram_counter]
                T_nx.add_edge(repre_node[idx_nd1], idx_new_nd)
                T_nx.add_edge(repre_node[idx_nd2], idx_new_nd)
                repre_node[idx_nd1] = repre_node[idx_nd2] = idx_new_nd
                coords_[idx_new_nd] = (coords_[idx_nd1, :] + coords_[idx_nd2, :] + coords_[node, :]) / 3
            T_nx.add_edge(idx_new_nd, repre_node[neighbours[int(min(Z[-1][:2]))]])

    adj = nx_to_adj(T_nx)
    if return_topo:
        return topology(adj=adj)
    else:
        return adj


def karger_init(A,temperature=1):
    # get edges and weigths
    edges_u,edges_v=sp.triu(A).nonzero()

    # get costs
    costs=sp.triu(A).data#np.exp(-temperature*A.data)

    # # np.random.exponential samples from the exponential distribution with scale 1.
    # # We want to sample from p(score > t) = exp(-tc). This means we have to
    # # _divide by_ c! See e.g. http://www.math.wm.edu/~leemis/chart/UDR/PDFs/ExponentialS.pdf
    # scores=np.random.exponential(scale=temperature,size=costs.shape)
    # scores = scores / costs
    # # sort scores in descending order
    # perm = np.argsort(scores)[::-1]

    probs=np.exp(-costs/temperature)

    #tree connected components. Initially each node is its own component
    connected_components={i:set([i]) for i in range(A.shape[0])}

    representant_class={i:i for i in range(A.shape[0])}

    idx_edge_perm=-1
    T=sp.lil_matrix(A.shape)

    edge_idx_sampled=[]
    while len(connected_components)>1:
        idx_edge_perm+=1

        # u,v=edges_u[perm[idx_edge_perm]],edges_v[perm[idx_edge_perm]]

        probs=probs/probs.sum()
        # Check for NaN values
        if np.isnan(probs).any():
            # Replace NaN values with 1e-10
            probs[np.isnan(probs)] = 1e-10
            probs[edge_idx_sampled]=0
            # Normalize again
            probs = probs / probs.sum()
        idx_edge=np.random.choice(range(len(probs)),p=probs)
        edge_idx_sampled.append(idx_edge)
        probs[idx_edge]=0
        u, v = edges_u[idx_edge], edges_v[idx_edge]

        rep_u=representant_class[u]
        rep_v=representant_class[v]

        if rep_v == rep_u:
            continue

        if rep_u>rep_v:
            u,v=v,u
            rep_u,rep_v=rep_v,rep_u


        #set edge value in tree
        T[u,v]=T[v,u]=A[u,v]

        #merge components
        connected_components[rep_u].update(connected_components[rep_v])


        for z in connected_components[rep_v]:
            representant_class[z]=rep_u
        del connected_components[rep_v]


    return T


def nx_to_adj(nx_graph):
    N=nx.number_of_nodes(nx_graph)
    n=N//2+1

    adj=[]

    for bp in range(n,N):
        adj.append(list(nx.neighbors(nx_graph, bp)))


    return np.array(adj,dtype=np.intc)


def sp_to_adj(sp_graph):
    N=sp_graph.shape[0]
    n=N//2+1

    adj=sp_graph.nonzero()[1][n:].reshape((n-2,3)).astype(np.intc)

    return adj




##%
#RANDOM INIT
def random_bin_tree(num_leaves):
    leaves=[0,1]
    T_nx=nx.Graph()
    T_nx.add_edge(0,1)
    max_node=1
    while len(leaves)!=num_leaves:
        idx=np.random.choice(range(len(leaves)))
        leave=leaves.pop(idx)

        max_node+=1
        T_nx.add_edge(leave, max_node)
        leaves.append(max_node)

        max_node += 1
        T_nx.add_edge(leave, max_node)
        leaves.append(max_node)

    terminal=0
    BP=num_leaves
    mapping={}
    for node in T_nx.nodes():
        if T_nx.degree(node)==1:
            mapping[node]=terminal
            terminal+=1
        elif T_nx.degree(node)==3:
            mapping[node] = BP
            BP += 1
        else:
            raise ValueError("node can not have degree%i"%T_nx.degree(node))

    T_nx=nx.relabel_nodes(T_nx,mapping)
    return nx_to_adj(T_nx)




def incremental_star(coords_terminals, alpha=1, demands=None, return_intermediate_trees=False):
    NUM_BPS = 1
    
    num_terminals = coords_terminals.shape[0]
    
    
    
    if demands is None:
        supply_array = np.array([1 / num_terminals])  # np.array([(nsites-1)/nsites])
        demand_array = -np.ones(num_terminals - 1) / num_terminals
        demands = np.append(supply_array, demand_array)

    
    #initialize sparse array star tree centered at center_coord
    T=sp.lil_matrix((num_terminals*2-2,num_terminals*2-2))
    
    #add edges from center to terminals (index center equal to num_terminals)
    T[:num_terminals,num_terminals]=1
    T[num_terminals,:num_terminals]=1
    T[num_terminals,num_terminals]=0
    
    neighbors_center=np.arange(num_terminals).tolist()
    flowsfromcenter=np.full(num_terminals,1/num_terminals).tolist()#assume source is centered at center_coord. Later is symmetrized
    
    
    num_neighbors_center= num_terminals
    
    #compute optimal position Steiner/Branching points
    T_, cost, SPcoords_arr, widths, _,_=fast_optimize_general(T[:num_terminals+NUM_BPS,:num_terminals+NUM_BPS].tocsr(), coords_terminals,
                                                              al=alpha, improv_threshold=1e-7,
                          demands=demands)
    
    if return_intermediate_trees:
        intermediate_trees=[{'T':T_,'cost':cost,'coords':SPcoords_arr.copy(),'widths':widths.copy()}]
    
    #set center coord
    center_coord=SPcoords_arr[num_terminals]
    
    # initialize optimal_angles
    optimal_angles = np.full((num_terminals,num_terminals), compute_optimal_angle(1/num_terminals, 1/num_terminals, alpha))
    optimal_angles = np.triu(optimal_angles, k=1)
    
    while NUM_BPS<num_terminals-2:
        #compute angles
        angles=compute_angles(SPcoords_arr[neighbors_center], center_coord)
        angles=np.triu(angles,k=1)
        
        
        
        u,v=np.unravel_index(np.argmax(optimal_angles-angles),angles.shape)
        
        
        #disconnect from center
        T[[neighbors_center[u],neighbors_center[v]],num_terminals]=0
        T[num_terminals,[neighbors_center[u], neighbors_center[v]]] = 0

        
        #connect to new BP
        T[[neighbors_center[u],neighbors_center[v]],NUM_BPS+num_terminals]=1
        T[NUM_BPS+num_terminals,[neighbors_center[u], neighbors_center[v]]] = 1
        
        #connect center to new BP
        T[num_terminals,NUM_BPS+num_terminals]=1
        T[NUM_BPS+num_terminals,num_terminals]=1
        
        
        
        if u>v:
            neighbors_center.pop(u)
            neighbors_center.pop(v)
            flow_u = flowsfromcenter.pop(u)
            flow_v = flowsfromcenter.pop(v)
        else:
            neighbors_center.pop(v)
            neighbors_center.pop(u)
            flow_u = flowsfromcenter.pop(v)
            flow_v = flowsfromcenter.pop(u)
        neighbors_center.append(NUM_BPS + num_terminals)
        flowsfromcenter.append(flow_u + flow_v)
        
        
        NUM_BPS += 1
        num_neighbors_center-=1

        #compute new optimal position of BP/SP
        T_, cost, SPcoords_arr, widths, _, _ = fast_optimize_general(T[:num_terminals+NUM_BPS,:num_terminals+NUM_BPS].tocsr(),
                                                                     coords_terminals, al=alpha,
                                                                     improv_threshold=1e-7,
                                                                     demands=demands)
        
        if return_intermediate_trees:
            intermediate_trees.append({'T': T_, 'cost': cost, 'coords': SPcoords_arr.copy(), 'widths': widths.copy()})
        # set center coord
        center_coord = SPcoords_arr[num_terminals]
        
        
        
        
        #update optimal angles
        if NUM_BPS!=num_terminals-2:
            # remove rows optimal angles
            optimal_angles = np.delete(optimal_angles, [u, v], 0)
            optimal_angles = np.delete(optimal_angles, [u, v], 1)
            
            # compute new optimal angles with respect to new BP
            optimal_angle_newBP=[compute_optimal_angle(flowsfromcenter[i],flowsfromcenter[-1],alpha) for i in range(num_neighbors_center-1)]
            
            optimal_angles=np.hstack((optimal_angles,np.array(optimal_angle_newBP)[:,None]))
            optimal_angles=np.vstack((optimal_angles,np.zeros((1,optimal_angles.shape[1]))))
            
    if return_intermediate_trees:
        return T,intermediate_trees
    return T

@njit
def compute_optimal_angle(flow_a,flow_b,alpha):
    symmetric_joint_flow=((flow_a+flow_b)*(1-(flow_a+flow_b)))
    symmetric_flow_a=(flow_a*(1-flow_a))
    symmetric_flow_b=(flow_b*(1-flow_b))
    numerator=symmetric_joint_flow**(2*alpha)-symmetric_flow_a**(2*alpha)-symmetric_flow_b**(2*alpha)
    denominator=2*(symmetric_flow_a*symmetric_flow_b)**(alpha)
    return np.arccos(numerator/denominator)
    
    

@njit
def compute_angles(coords,center):
    #center terminal_coords at center
    vectors=coords-center
    
    
    norm_vectors= (vectors**2).sum(1)**0.5
    # normalize vectors
    vectors/=norm_vectors[:,None]
    
    #compute angles
    angles=np.arccos(vectors.dot(vectors.T))
    return angles
    

def get_shortest_path_tree(adj_matrix, roots):
    num_nodes = adj_matrix.shape[0]
    tree_matrix = sp.lil_matrix((num_nodes, num_nodes), dtype=adj_matrix.dtype)
    if isinstance(roots,int):
        roots=[roots]
    if len(roots)>num_nodes:
        roots=set(range(num_nodes)).difference(roots)
    for root in roots:
        if networkit_installed:
            graph=nk.nxadapter.nx2nk(nx.from_scipy_sparse_array(adj_matrix, create_using=nx.Graph()), weightAttr='weight')
            dijkstra = nk.distance.Dijkstra(graph, root,storePaths=True,storeNodesSortedByDistance=True)
            dijkstra.run()
            
            # Get the predecessor array and build the adjacency matrix of the shortest path tree
            predecessors = [dijkstra.getPredecessors(i)[0] if i!=root else -9999 for i in range(num_nodes) ]
            order_nodes=dijkstra.getNodesSortedByDistance()[::-1]
        
        else:
            distances, predecessors = sp.csgraph.dijkstra(adj_matrix, indices=root, return_predecessors=True)
            order_nodes=np.argsort(distances)[::-1]
    
        # Construct the shortest path tree adjacency matrix
        visited_nodes = set([root])
        for node in order_nodes:
            if node in visited_nodes:
                continue
            current_node = node
            while current_node != root:
                visited_nodes.add(current_node)
                predecessor = predecessors[current_node]
                tree_matrix[predecessor, current_node]=tree_matrix[current_node,predecessor] = adj_matrix[predecessor, current_node]
                if predecessor in visited_nodes:
                    break
                current_node = predecessor
            if len(visited_nodes)==num_nodes:
                break

    return tree_matrix.tocsr()
    
    
    

