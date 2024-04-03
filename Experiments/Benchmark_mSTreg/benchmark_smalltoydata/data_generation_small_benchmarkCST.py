
import os
print('cwd',os.getcwd())

if '/Benchmark_mSTreg' in os.getcwd():

    os.chdir('../../../')
    print('cwd updated',os.getcwd())

    import sys
    sys.path.append(os.getcwd())
    print(sys.path)

import networkx as nx

from lib.CST.methods.mSTreg import generate_topologies
import numpy as np
from lib.CST.T_datacls import T_data,save_object
from lib.CST.T_datacls.utilities.graphtools import Wiener_index,  centrality_weights_tree
from sklearn.metrics import pairwise_distances
import multiprocessing as mp
from functools import partial






#%%

def brute_force(iteration=None,n=5,alpha_range=np.linspace(0,1,11),P=None,save=True,folder=''):
    if P is None:
        P = np.random.random((n, 2)) * 10

    n=len(P)
    num_topos = 1
    for i in range(2 * n):
        k = 2 * n - 5 - 2 * i
        if k > 1:
            num_topos *= k
        else:
            break

    tdata=T_data(P)
    Branch_cost={alpha:np.inf for alpha in alpha_range}
    best_adj={}
    BCST_costs_dict={alpha: np.empty(num_topos) for alpha in alpha_range}
    for i,Branch_topo in enumerate(generate_topologies(n=num_topos,nsites=n)):

        adj=Branch_topo.adj
        for alpha in alpha_range:
            tdata.compute_BCST(alpha=alpha, init_topo=adj,
                               return_topo_CST=False,maxiter_mSTreg=-1)
            BCST_costs_dict[alpha][i]=tdata.costs['BCST_%0.2f' % alpha]
            if Branch_cost[alpha] > tdata.costs['BCST_%0.2f' % alpha]:
                best_adj[alpha]=adj
                Branch_cost[alpha]=tdata.costs['BCST_%0.2f' % alpha]

    for alpha in alpha_range:
        tdata.compute_BCST(alpha=alpha, init_topo=best_adj[alpha],
                           return_topo_CST=False,maxiter_mSTreg=-1)
    tdata.BCST_costs_dict = {alpha:sorted(ls)[:1000] for alpha,ls in BCST_costs_dict.items()}

    CT_cost={alpha:np.inf for alpha in alpha_range}
    D=pairwise_distances(P)
    best_Tnx={}
    CST_costs_dict = {alpha: np.empty(n**(n-2)) for alpha in alpha_range}
    for i in range(n**(n-2)):
        prufer_code=np.unravel_index(i,[n]*(n-2))
        T_nx=nx.from_prufer_sequence(prufer_code)
        for e in T_nx.edges():
            u,v=e
            T_nx[u][v]['weight']=D[u,v]
        for alpha in alpha_range:
            cost=Wiener_index(T_nx,alpha) / (n ** (2 * alpha))
            CST_costs_dict[alpha][i]=cost
            if CT_cost[alpha]>cost:
                best_Tnx[alpha]=T_nx
                CT_cost[alpha]=cost
    tdata.CST_costs_dict={alpha:sorted(ls)[:1000] for alpha,ls in CST_costs_dict.items()}
    for alpha in alpha_range:
        tdata.trees['best_CST_%0.2f'%alpha]=nx.adjacency_matrix(best_Tnx[alpha],nodelist=list(range(n)))
        tdata.widths_trees['best_CST_%0.2f' % alpha] = centrality_weights_tree(best_Tnx[alpha],max_width=1)
    if save:
        os.makedirs(folder,True)
        save_object(tdata,folder+str(iteration).zfill(4))
    return iteration

n_range=[5]
for n in n_range:
    print(n)
    start_iterations=0
    end_iterations=10
    cpu_counter = min(mp.cpu_count(),3)
    alpha_range=np.array(list(range(11)))/10
    folder='Experiments/Benchmark_mSTreg/benchmark_smalltoydata/Data/n=%i/'%n

    os.makedirs(folder,True)

    partial_brute_force=partial(brute_force, n=n, alpha_range=alpha_range,save=True,folder=folder)
    for i in range(start_iterations,end_iterations):
        rs=partial_brute_force(i)
        print(rs)