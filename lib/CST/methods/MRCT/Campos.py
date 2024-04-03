import numpy as np
import scipy.sparse as spp
import itertools
from pqdict import pqdict
#%%
class priority_weight:
    def __init__(self,wd,jsp):
        self.wd=wd
        self.jsp=jsp
    def __lt__(self, other):
        if self.wd<other.wd:
            return True
        else:
            return (self.jsp>other.jsp)
    def __le__(self, other):
        if self.wd <= other.wd:
            return True
        else:
            return (self.jsp >= other.jsp)
    def __gt__(self, other):
        return not self.__le__(self,other)
    def __ge__(self, other):
        return not self.__lt__(self,other)

    def __repr__(self):
        return "wd =%0.3f, jsp =%0.3f"%(self.wd,self.jsp)

def Campos_MRCT(W,C1=0.2,C2=0.6,C3=0.2,C4=None,C5=None):

    #number nodes
    n=W.shape[0]

    # Define tree
    T = spp.lil_matrix((n, n))

    #set neighbors
    if isinstance(W,np.ndarray):
        if np.count_nonzero(W)==n**2-n:
            neighbors={i:list(itertools.chain(range(i), range(i + 1, n))) for i in range(n)}
        else:
            neighbors={i:W[i].nonzero()[0].tolist() for i in range(n)}

        #mean weights and standard deviation weights
        mu,std=np.mean(W[W.nonzero()]),np.std(W[W.nonzero()])
    elif spp.isspmatrix(W):
        neighbors = {i: W[i].nonzero()[1].tolist() for i in range(n)}
        # mean weights and standard deviation weights
        data=W.tocsr().data
        mu, std = np.mean(data), np.std(data)
    # Set parameters C4 and C5
    if C4 is None or C5 is None:
        Thr=0.4+0.005*(n-10)
        if std/mu<Thr:
            if C4 is None:
                C4=1
            if C5 is None:
                C5=1
        else:
            if C4 is None:
                C4=0.9
            if C5 is None:
                C5=0.1

    ## Initialize parameters
    #degrees
    deg= np.zeros(n)
    #weighted degree
    w_deg = np.zeros(n)
    # max adjacent weight
    w_max = np.zeros(n)
    for node in range(n):
        for neighbor in neighbors[node]:
            if neighbor>node:
                deg[node]+=1
                w_deg[node]+=W[node,neighbor]
                w_max[node]=max(w_max[node],W[node,neighbor])

                deg[neighbor] += 1
                w_deg[neighbor] += W[neighbor, node]
                w_max[neighbor] = max(w_max[neighbor], W[neighbor,node])

    # 1st priority weight nodes
    wd = np.ones(n)*np.inf
    # 2nd priority weight nodes
    jsp=np.zeros(n)

    #node weights wrt tree
    w_nodes = np.ones(n)*np.inf
    #cost to root. root is the initial vertex
    c_root = np.ones(n)*np.inf
    # #spanning potential
    # sp = [0] * n
    # sp_max=0
    # for node in range(n):
    #     sp[node]=C1*deg[node]+C2*deg[node]/w_deg[node]+C3/w_max[node]
    #     if sp[node]>sp_max:
    #         sp_max=sp[node]
    #         root=node
    root=np.argmax(C1*deg+C2*deg/w_deg+C3/w_max)

    w_nodes[root]=0
    c_root[root]=0
    wd[root]=0
    jsp[root] = np.inf

    parents=[-1]*n

    #nodes present in tree
    visited_nodes=set([root])

    #priority list
    pq=pqdict({root:priority_weight(wd[root],jsp[root])})
    while len(pq)>0:
        u=pq.pop()
        for v in neighbors[u]:
            if v not in visited_nodes:
                wd_aux=C4*W[u,v]+C5*(c_root[u]+W[u,v])
                jsp_aux=(deg[u]+deg[v])+(deg[u]+deg[v])/(w_deg[u]+w_deg[v])
                if wd_aux<wd[v]:
                    wd[v]=wd_aux
                    jsp[v]=jsp_aux
                    c_root[v]=c_root[u] + W[u, v]
                    parents[v]=u
                elif (wd_aux==wd[v] and jsp_aux>=jsp[v]):
                    jsp[v]=jsp_aux
                    c_root[v] = c_root[u] + W[u, v]
                    parents[v] = u
                pq[v]=priority_weight(wd[v],jsp[v])
        visited_nodes.add(u)
        if parents[u]>=0:
            T[u, parents[u]] = T[parents[u], u] = W[u, parents[u]]
    # for u in range(n):
    #     if u==root:
    #         continue
    #     T[u,parents[u]]=T[parents[u],u]=W[u,parents[u]]
    return T

if __name__=='__main__':
    from sklearn.metrics import pairwise_distances
    from utils.utils import Wiener_index, plot_graph,centrality_weights_tree
    import matplotlib.pyplot as plt
    from graph_generation.graph_generation import create_points
    n=50
    alpha=1
    np.random.seed(15)
    # P=np.random.uniform(0,1,size=(n,2))
    distribution = 'estein%i.%i' % (n, 5)
    P, P_vis = create_points(n=n, distribution=distribution, noise_perc=0,
                             Random=False, return_dictionary=False)[:2]

    D=pairwise_distances(P)
    # mask = np.random.uniform(0, 1, D.shape) > 0.97
    # D = spp.lil_matrix(mask * D)
    # D = D.maximum(D.T)


    '''
    #EXAMPLE PAPER
    P=np.array([[0,1],[-0.5,.5],[0,0],[.5,.5],[0.5,-0.5],[-1,0],[-0.5,-0.5],[0,-1]])
    D=spp.lil_matrix((8,8))
    D[0,1]=D[0,3]=D[6,7]=1
    D[1,2]=D[2,4]=D[5,6]=3
    D[1,5]=D[2,3]=D[2,6]=D[3,4]=D[4,7]=2
    D = D.maximum(D.T)
    # '''
    k1=1
    k2=2

    plot_graph(spp.lil_matrix(D),P,title='G',figsize=(12,6),counter_plot=0,k1=k1,k2=k2)


    T=Campos_MRCT(D)
    widths=centrality_weights_tree(T,max_width=8)
    costT=Wiener_index(T,alpha=alpha)#/(len(P)**2)
    plot_graph(T,P,widths=widths,title='Cost=%0.5f'%costT,counter_plot=1,k1=k1,k2=k2)
    plt.tight_layout()
    plt.show()


