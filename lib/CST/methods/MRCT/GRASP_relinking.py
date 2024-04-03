import numpy as np
import scipy.sparse as sp
try:
    from ...utils.utils import Wiener_index,centrality_weights_tree
    from ...utils.graphtools import ensure_connected_knn_graph
    from ...T_datacls.T_datacls import T_data
    from ..mSTreg.topology.prior_topology import karger_init

except:
    from utils.utils import Wiener_index, centrality_weights_tree
    from utils.graphtools import ensure_connected_knn_graph
    from T_datacls.T_datacls import T_data
    from methods.mSTreg.topology.prior_topology import karger_init

from scipy.spatial.distance import pdist, squareform
import itertools
import bisect #add elements to list while respecting order
import time
from tqdm import tqdm
try:
    import networkit as nk
    nk_available=True
except:
    nk_available=False
import networkx as nx

#%%
use_V1=True
class GRASP_relinking:

    def __init__(self,W,buffer_capacity_ratio=4,max_num_iterations=300,
                 max_it_localsearch=200,max_it_pathrelink=100,
                 alpha=1,max_time=600):
        self.W=W
        self._n=W.shape[0]
        self.alpha=alpha
        buffer_capacity = buffer_capacity_ratio * np.sqrt(W.shape[0])
        self.buffer = Ordered_Buffer(buffer_capacity=buffer_capacity)
        self._set_roots(W)
        self.max_time=max_time

        self.max_num_iterations=max_num_iterations
        self.max_it_localsearch=max_it_localsearch
        self.max_it_pathrelink=max_it_pathrelink
        
        self._set_limit_time_local_search_and_path_relinking(max_time)
        
    def _set_limit_time_local_search_and_path_relinking(self,limit_time):
        self.limit_time_local_search=2.5*limit_time
        self.limit_time_path_relinking=2.5*limit_time
    def _set_roots(self, W):
        if isinstance(W,np.ndarray):
            #assumes W euclidean distance matrix
            r1,r2=np.argpartition(W.sum(1),2)[:2]
            d2r1,d2r2=W[r1].flatten(),W[r2].flatten()
            T1=sp.lil_matrix((self._n,self._n))
            T1[r1]=W[r1]
            T1+=T1.T
            T2 = sp.lil_matrix((self._n, self._n))
            T2[r2] = W[r2]
            T2 += T2.T
            cost1=W[r1].sum()*(self._n-1)
            cost2 = W[r2].sum() * (self._n - 1)
        elif sp.isspmatrix(W):
            cost1 = np.inf
            cost2 = np.inf
            r1=r2=-1
            T1=T2=None
            for node in range(W.shape[0]):
                T_=SPT(W, node)
                cost=Wiener_index(T_,alpha=self.alpha)
                if cost >= cost2:
                    continue
                if cost<=cost1:
                    cost2=cost1
                    cost1=cost
                    r2=r1
                    r1 = node
                    T2=T1
                    T1=T_
                elif cost<cost2:
                    cost2=cost
                    r2 = node
                    T2 = T_
            d2r1,d2r2=sp.csgraph.dijkstra(T1,directed=False,unweighted=False,indices=r1),sp.csgraph.dijkstra(T2,directed=False,unweighted=False,indices=r2)


        self.best_TreeCost=TreeCost_cls(T1,cost1)
        self.buffer.add((TreeCost_cls(T1,cost1)))
        self.buffer.add((TreeCost_cls(T2, cost2)))
        self.r1,self.r2,self.d2r1,self.d2r2 = r1,r2,d2r1,d2r2

    def construct_tree(self):


        T = sp.lil_matrix((self._n, self._n))

        idx_r=np.random.choice([0,1],size=1)
        if idx_r==0:
            d2r=self.d2r1
            r=self.r1
        else:
            d2r = self.d2r2
            r=self.r2
        nodes_in_tree=[r]
        nodes_notin_tree=list(range(self._n))
        nodes_notin_tree.pop(r)
        for _ in range(self._n-1):
            if sp.isspmatrix(self.W):
                P=self.W[nodes_in_tree][:,nodes_notin_tree].multiply(np.tile(d2r[nodes_notin_tree],(len(nodes_in_tree),1)))
                P /= P.sum()
                p = P.data
                edges=list(map(lambda x,y:(nodes_in_tree[x],nodes_notin_tree[y]),*P.nonzero()))

            else:
                P = self.W[nodes_in_tree][:,nodes_notin_tree]*np.tile(d2r[nodes_notin_tree],(len(nodes_in_tree),1))
                P/=P.sum()
                p=P.flatten()
                edges = list(map(lambda x: (nodes_in_tree[x[0]], nodes_notin_tree[x[1]]), itertools.product(range(P.shape[0]),range(P.shape[1]))))

            u, v = edges[np.random.choice(len(edges), p=p)]
            nodes_in_tree.append(v)
            nodes_notin_tree.remove(v)
            T[u,v]=T[v,u]=self.W[u,v]

        return T

    def local_search_V1(self,T,cost):
        """
        Perform a local search on the given tree using strategy V1.

        Parameters:
        - T (sparse matrix): The input tree.
        - cost (float): The initial cost of the tree.

        Returns:
        - T (sparse matrix): The updated tree.
        - current_cost (float): The final cost of the updated tree.

        Strategy V1 explores all solutions during the iteration and accepts the one with the best cost improvement.
        """
        repeat=True
        current_cost=min_cost=cost
        edges_T=list(map(lambda x,y:(x,y), *sp.triu(T).nonzero()))
        edges_W = list(map(lambda x, y: (x, y), *sp.triu(self.W).nonzero()))
        centralities_T={e:centrality_e/2 for e,centrality_e in zip(edges_T,centrality_weights_tree(T.tocsr(),norm=False))}
        repeat_iterations = 0
        start_time=time.time()
        while repeat and repeat_iterations < self.max_it_localsearch:
            repeat_iterations += 1
            repeat = False
            for idx_e_T,e_T in enumerate(edges_T):
                cost_diff,candidate_T,candidate_centralities_T=self._find_minimizer_connection(e_T,T,centralities_T,edges_W)
                candidate_cost=current_cost+cost_diff



                if min_cost>candidate_cost:
                    repeat=True
                    min_cost = candidate_cost
                    minT=candidate_T.copy()
                    centralities_minT=candidate_centralities_T
            if repeat:
                T=minT.copy()
                current_cost=min_cost
                centralities_T=centralities_minT
                # edges_T=list(centralities_T.keys())
                edges_T = list(map(lambda x, y: (x, y), *sp.triu(T).nonzero()))
            

            if time.time()-start_time>self.limit_time_local_search:
                break



        return T, current_cost,repeat_iterations
    def local_search_V2(self,T,cost):
        """
        Perform a local search on the given tree using strategy V2.

        Parameters:
        - T (sparse matrix): The input tree.
        - cost (float): The initial cost of the tree.

        Returns:
        - T (sparse matrix): The updated tree.
        - current_cost (float): The final cost of the updated tree.

        Strategy V2 accepts the first solution that improves the cost during the iteration.
        """
        repeat=True
        current_cost=cost
        edges_T=list(map(lambda x,y:(x,y), *sp.triu(T).nonzero()))
        edges_W = list(map(lambda x, y: (x, y), *sp.triu(self.W).nonzero()))
        centralities_T={e:centrality_e/2 for e,centrality_e in zip(edges_T,centrality_weights_tree(T.tocsr(),norm=False))}
        repeat_iterations=0
        start_time=time.time()
        while repeat and repeat_iterations<self.max_it_localsearch:
            repeat_iterations += 1
            repeat = False
            for idx_e_T,e_T in enumerate(edges_T):
                cost_diff,candidate_T,candidate_centralities_T=self._find_minimizer_connection(e_T,T,centralities_T,edges_W)
                if cost_diff<0:
                    repeat=True
                    T = candidate_T.copy()
                    current_cost += cost_diff
                    centralities_T = candidate_centralities_T
                    # edges_T=list(centralities_T.keys())
                    edges_T = list(map(lambda x, y: (x, y), *sp.triu(T).nonzero()))

                    # # TODO remove assert
                    # assert np.isclose(current_cost , Wiener_index(T, self.alpha))

                    break

            if time.time()-start_time>self.limit_time_local_search:
                break


        return T,current_cost,repeat_iterations
    def _find_minimizer_connection(self,e_T,T,centralities_T,edges_W):
        #TODO: These lines were added after finding bug in compute_difference_cost_cycles
        ############################33
        current_cost=Wiener_index(T,self.alpha)
        new_centralities = centralities_T.copy()
        del new_centralities[e_T]
        ###############################33

        T_ = T.tolil().copy()
        #remove edge e_T
        T_[e_T[0], e_T[1]] = T_[e_T[1], e_T[0]] = 0

        # differentiate between nodes at the "left" and "right" side of eliminated edge
        _, label_components = sp.csgraph.connected_components(T_, directed=False, return_labels=True)
        min_cost_diff=0
        for e_W in edges_W:
            if e_W==e_T:
                continue
            u,v=e_W
            if label_components[u]==label_components[v]:
                continue
            #TODO: this function has bugs Fix it
            # cost_diff,new_centralities=self.compute_difference_cost_cycles(T,e_T,e_W,centralities_T,label_components)
            # line above replaced by the following ones. The dict of centralities does not play now any role, but removing
            # it would require to change the code in the main loop
            ##############################################
            T_[u,v]=T_[v,u]=self.W[u,v]
            new_centralities[e_W]=-1
            cost_diff=Wiener_index(T_,self.alpha)-current_cost
            T_[u, v] = T_[v, u] = 0
            
            ##############################################


            if cost_diff<min_cost_diff:
                min_cost_diff=cost_diff
                e_ins=e_W
                e_del=e_T
                best_new_centralities=new_centralities.copy()
                
            del new_centralities[e_W]
            
            
            
        if min_cost_diff<0:
            minT=T.copy().tolil()
            minT[e_del[0], e_del[1]] = minT[e_del[1], e_del[0]] = 0
            minT[e_ins[0], e_ins[1]] = minT[e_ins[1], e_ins[0]] = self.W[e_ins[0],e_ins[1]]

            centralities_minT=centralities_T.copy()
            del centralities_minT[e_del]
            for e in best_new_centralities.keys():
                centralities_minT[e]=best_new_centralities[e]

            return min_cost_diff,minT,centralities_minT
        else:
            return 0, T, centralities_T


    def compute_difference_cost_cycles(self,T, e_T, e_W,centralities_T,label_components):
        #TODO THIS FUNCTION DOES NOT WORK PROPERLY. FIX IT
        _,predecessors=sp.csgraph.dijkstra(T,directed=False,return_predecessors=True,indices=e_T[0])

        num_nodes_right_eT=np.count_nonzero(label_components)
        num_nodes_left_eT = (self._n-num_nodes_right_eT)

        if label_components[e_W[0]]==0:
            node_left=e_W[0]
            node_right=e_W[1]
        else:
            node_left = e_W[1]
            node_right = e_W[0]

        cost_cycle_old=0
        cost_cycle_new = 0

        if node_left<node_right:
            new_centralities = {(node_left,node_right):centralities_T[e_T]}
        else:
            new_centralities = {(node_right,node_left): centralities_T[e_T]}

        #copmpute cost left part cycle
        node_lcycle=node_left
        while predecessors[node_lcycle]>=0:
            pred=predecessors[node_lcycle]
            if e_T==(predecessors[node_lcycle],node_lcycle) or e_T==(node_lcycle,predecessors[node_lcycle]):
                cost_cycle_old += centralities_T[e_T]**self.alpha * self.W[e_T[0],e_T[1]]
                cost_cycle_new += centralities_T[e_T]**self.alpha * self.W[node_left,node_right]

            else:
                if node_lcycle < pred:
                    e = node_lcycle, pred
                else:
                    e = pred, node_lcycle
                try:
                    cost_cycle_old += centralities_T[e] ** self.alpha * self.W[node_lcycle,pred]
                except:
                    print('com?')
                    break
                cent_eW=centralities_T[e]
                k=np.sqrt(self._n**2-4*cent_eW)
                if (self._n-k)/2<num_nodes_left_eT and (self._n-k)/2>num_nodes_right_eT:
                    new_centrality = self._get_update_centrality_single_edge(T, e_T, e_W, e)
                elif (self._n-k)/2<num_nodes_left_eT:# and not((self._n-k)/2>num_nodes_right_eT):
                        new_num_nodes_right_eW=(self._n-k)/2+num_nodes_right_eT
                        new_num_nodes_left_eW=(self._n+k)/2-num_nodes_right_eT
                        new_centrality=new_num_nodes_left_eW*new_num_nodes_right_eW
                elif (self._n-k)/2>num_nodes_right_eT:# and not((self._n-k)/2<num_nodes_left_eT):
                        new_num_nodes_right_eW = (self._n + k) / 2 + num_nodes_right_eT
                        new_num_nodes_left_eW = (self._n - k) / 2 - num_nodes_right_eT
                        new_centrality=new_num_nodes_left_eW*new_num_nodes_right_eW
                else:
                    raise ValueError
                # #TODO remove assert
                # assert(new_centrality==self._get_update_centrality_single_edge(T,e_T,e_W,e))
                cost_cycle_new += new_centrality ** self.alpha * self.W[node_lcycle,pred]
                new_centralities[e]=new_centrality

            node_lcycle=pred

        # copmpute cost right part cycle
        node_rcycle = node_right
        while predecessors[node_rcycle] >= 0:
            pred = predecessors[node_rcycle]
            if e_T == (predecessors[node_rcycle], node_rcycle) or e_T == (node_rcycle, predecessors[node_rcycle]):
                cost_cycle_old += centralities_T[e_T] ** self.alpha * self.W[e_T[0], e_T[1]]
                cost_cycle_new += centralities_T[e_T]**self.alpha * self.W[node_left,node_right]

            else:
                if node_rcycle < pred:
                     e = node_rcycle, pred
                else:
                     e = pred, node_rcycle
                cost_cycle_old += centralities_T[e] ** self.alpha * self.W[
                    node_rcycle, pred]

                cent_eW = centralities_T[e]
                k = np.sqrt(self._n**2 - 4 * cent_eW)
                if (self._n - k) / 2 < num_nodes_right_eT and (self._n - k) / 2 > num_nodes_left_eT:
                    new_centrality = self._get_update_centrality_single_edge(T, e_T, e_W, e)
                elif (self._n - k) / 2 < num_nodes_right_eT:#and not((self._n - k) / 2 > num_nodes_left_eT):
                        new_num_nodes_left_eW = (self._n - k) / 2 + num_nodes_left_eT
                        new_num_nodes_right_eW = (self._n + k) / 2 - num_nodes_left_eT
                        new_centrality = new_num_nodes_left_eW * new_num_nodes_right_eW
                elif (self._n - k) / 2 > num_nodes_left_eT:# and not((self._n - k) / 2 < num_nodes_right_eT):
                        new_num_nodes_left_eW = (self._n + k) / 2 + num_nodes_left_eT
                        new_num_nodes_right_eW = (self._n - k) / 2 - num_nodes_left_eT
                        new_centrality = new_num_nodes_left_eW * new_num_nodes_right_eW
                else:
                    raise ValueError
                # #TODO remove assert
                # assert(new_centrality==self._get_update_centrality_single_edge(T,e_T,e_W,e))
                cost_cycle_new += new_centrality ** self.alpha * self.W[node_rcycle, pred]
                new_centralities[e] = new_centrality

            node_rcycle = pred


        return cost_cycle_new-cost_cycle_old,new_centralities

    def _get_update_centrality_single_edge(self,T,e_T,e_W,e):
        T_=T.copy()
        T_[e_T[0],e_T[1]]=T_[e_T[1],e_T[0]]=T_[e[0],e[1]]=T_[e[1],e[0]]=0
        T_[e_W[0], e_W[1]] = T_[e_W[1], e_W[0]] = 1

        _,labels=sp.csgraph.connected_components(T_,directed=False,return_labels=True)


        nodes_left=np.count_nonzero(labels)
        return nodes_left*(self._n-nodes_left)


    def Path_reliking_v1(self,T1,costT1,T2,costT2):
        repeat=True
        assert costT1<=costT2
        bestT=T1.copy()
        min_cost=current_cost=costT1
        edgesT1 = list(map(lambda x, y: (x, y), *sp.triu(T1).nonzero()))
        edgesT2 = list(map(lambda x, y: (x, y), *sp.triu(T2).nonzero()))

        edges2remove = set(edgesT1).difference(edgesT2)
        edges2add = set(edgesT2).difference(edgesT1)

        centralities_bestT = {e: centrality_e / 2 for e, centrality_e in
                              zip(edgesT1, centrality_weights_tree(bestT.tocsr(), norm=False))}
        repeat_iterations = 0
        start_time=time.time()
        while repeat and repeat_iterations < self.max_it_pathrelink:
            repeat_iterations += 1
            repeat = False
            for e_T in edges2remove:
                cost_diff, candidate_T, candidate_centralities_T = self._find_minimizer_connection(e_T, bestT,
                                                                                                   centralities_bestT,
                                                                                                   edges2add)
                candidate_cost = current_cost + cost_diff

                # # TODO remove assert
                # assert np.isclose(candidate_cost , Wiener_index(candidate_T,self.alpha))

                if min_cost > candidate_cost:
                    repeat = True
                    min_cost = candidate_cost
                    minT = candidate_T.copy()
                    centralities_minT = candidate_centralities_T
                    e_T_del=e_T
            if repeat:
                bestT = minT.copy()
                current_cost = min_cost
                centralities_bestT = centralities_minT
                edges2remove.remove(e_T_del)
                edges2add.difference(centralities_bestT.keys())
            

            if time.time()-start_time>self.limit_time_path_relinking:
                break
            
        return bestT, current_cost,repeat_iterations

    def Path_reliking_v2(self, T1, costT1, T2, costT2):
        repeat=True
        assert costT1 <= costT2
        bestT = T1.copy()
        min_cost = current_cost = costT1
        edgesT1 = list(map(lambda x, y: (x, y), *sp.triu(T1).nonzero()))
        edgesT2 = list(map(lambda x, y: (x, y), *sp.triu(T2).nonzero()))

        edges2remove = set(edgesT1).difference(edgesT2)
        edges2add = set(edgesT2).difference(edgesT1)

        centralities_bestT = {e: centrality_e / 2 for e, centrality_e in
                              zip(edgesT1, centrality_weights_tree(bestT.tocsr(), norm=False))}
        repeat_iterations = 0
        start_time=time.time()
        while repeat and repeat_iterations < self.max_it_pathrelink:
            repeat_iterations += 1
            repeat=False
            for e_T in edges2remove:
                cost_diff, candidate_T, candidate_centralities_T = self._find_minimizer_connection(e_T, bestT,
                                                                                                   centralities_bestT,
                                                                                                   edges2add)
                candidate_cost = current_cost + cost_diff

                # # TODO remove assert
                # assert np.isclose(candidate_cost , Wiener_index(candidate_T, self.alpha))

                if min_cost > candidate_cost:
                    repeat = True
                    current_cost = candidate_cost.copy()
                    bestT = candidate_T
                    centralities_bestT = candidate_centralities_T
                    edges2remove.remove(e_T)
                    edges2add.difference(centralities_bestT.keys())
                    break

            
            if time.time()-start_time>self.limit_time_path_relinking:
                break
            
        return bestT, current_cost,repeat_iterations

    def run (self,return_history=False):
        num_iterations=0
        pbar=tqdm(total=self.max_num_iterations)
        if return_history:
            cost_LS_history=[]
            cost_PR_history=[]
            cost_best_history=[(self.best_TreeCost.cost,0)]
            iter_LS_history = []
            iter_PR_history=[]
        start_time=time.time()
        while num_iterations<self.max_num_iterations:
            pbar.update(1)
            num_iterations+=1
            T1=self.construct_tree()
            cost_T1=Wiener_index(T1,alpha=self.alpha)
            if use_V1:
                T1,costT1,local_search_iter=self.local_search_V1(T1,cost_T1)
            else:
                T1, costT1,local_search_iter = self.local_search_V2(T1, cost_T1)
            if return_history:
                iter_LS_history.append(local_search_iter)
                cost_LS_history.append((costT1,time.time()-start_time))
            TreeCost=self.buffer.sample()
            T2=TreeCost.T
            costT2 = TreeCost.cost

            if costT1<costT2:
                if use_V1:
                    T,costT,PR_iter=self.Path_reliking_v1( T1, costT1, T2, costT2)
                else:
                    T, costT,PR_iter = self.Path_reliking_v2(T1, costT1, T2, costT2)
            else:
                if use_V1:
                    T, costT,PR_iter = self.Path_reliking_v1(T2, costT2, T1, costT1)
                else:
                    T, costT,PR_iter = self.Path_reliking_v2(T2, costT2, T1, costT1)
            
            treecost = TreeCost_cls(T.copy(), costT)
            self.buffer.add(treecost)
            
            if return_history:
                iter_PR_history.append(PR_iter)
                cost_PR_history.append((costT,time.time()-start_time))
            if costT<self.best_TreeCost.cost:
                self.best_TreeCost.cost=costT
                self.best_TreeCost.T = T.copy()
                
                
            if return_history:
                cost_best_history.append((self.best_TreeCost.cost,time.time()-start_time))
            if time.time()-start_time>self.max_time:
                print('max time %0.2f reached: actual time %0.2f'%(self.max_time,time.time()-start_time))
                break
        pbar.close()
        if return_history:
            return self.best_TreeCost.T,self.best_TreeCost.cost,cost_LS_history,iter_LS_history,cost_PR_history,iter_PR_history,cost_best_history
        return self.best_TreeCost.T,self.best_TreeCost.cost

class GRASP_relinking_general_alf(GRASP_relinking):
    def __init__(self,coords,buffer_capacity_ratio=4,max_num_iterations=300,
                 max_it_localsearch=200,max_it_pathrelink=100,
                 alpha=1,max_time=600,):
        self.tdata=T_data(coords,verbose=False)
        W=squareform(pdist(coords))
        self.W=W
        self._n=W.shape[0]
        self.alpha=alpha
        self.max_time=max_time

        buffer_capacity = buffer_capacity_ratio * np.sqrt(W.shape[0])
        self.buffer = Ordered_Buffer(buffer_capacity=buffer_capacity)

        if alpha>=0.7:
            self._set_roots(W)
        else:
            self._set_karger_base_graph(coords)
            self._define_init_bestT()

        self.max_num_iterations=max_num_iterations
        self.max_it_localsearch=max_it_localsearch
        self.max_it_pathrelink=max_it_pathrelink
        self._set_limit_time_local_search_and_path_relinking(max_time)


    def _define_init_bestT(self,):
        lamda_karger_temperature = np.quantile(self.karger_base_graph.data, np.random.uniform(0, 0.4))
        karger_temperature = np.random.exponential(1 / lamda_karger_temperature)
        T=karger_init(self.karger_base_graph, karger_temperature)
        cost=Wiener_index(T,alpha=self.alpha)
        self.best_TreeCost=TreeCost_cls(T,cost)
        self.buffer.add(TreeCost_cls(T,cost))

    def _set_karger_base_graph(self,coords):
        self.karger_base_graph = ensure_connected_knn_graph(coords,num_neighs=int(2*np.log(len(coords))))
    def construct_tree(self):
        if self.alpha>=0.7:
            return super().construct_tree()

        lamda_karger_temperature=np.quantile(self.karger_base_graph.data,np.random.uniform(0,0.4))
        karger_temperature=np.random.exponential(1/lamda_karger_temperature)

        return karger_init(self.karger_base_graph,karger_temperature)

class GRASP_relinking_CST(GRASP_relinking):
    def __init__(self,P,buffer_capacity_ratio=4,max_num_iterations=300,
                 max_it_localsearch=200,max_it_pathrelink=100,
                 max_time=600,
                 alpha=1,maxfreq_mSTreg=3, maxiter_mSTreg=20):
        self.tdata=T_data(P,verbose=False)
        W=squareform(pdist(P))
        self.W=W
        self._n=W.shape[0]
        self.alpha=alpha
        self.max_time = max_time

        buffer_capacity = buffer_capacity_ratio * np.sqrt(W.shape[0])
        self.buffer = Ordered_Buffer(buffer_capacity=buffer_capacity)
        
        self.maxfreq_mSTreg=maxfreq_mSTreg
        self.maxiter_mSTreg=maxiter_mSTreg

        self._define_init_bestT()

        self.max_num_iterations=max_num_iterations
        self.max_it_localsearch=max_it_localsearch
        self.max_it_pathrelink=max_it_pathrelink
        self._set_limit_time_local_search_and_path_relinking(max_time)



    def _define_init_bestT(self,):
        self.tdata.compute_BCST(alpha=self.alpha,
        maxfreq_mSTreg = self.maxfreq_mSTreg, maxiter_mSTreg=self.maxiter_mSTreg,return_topo_CST=True)

        T=self.tdata.trees['CST_%0.2f' % self.alpha].T
        cost=Wiener_index(T,alpha=self.alpha)

        self.best_TreeCost = TreeCost_cls(T, cost)
        self.buffer.add(TreeCost_cls(T, cost))
        

    def construct_tree(self):
        criterium_BP_position_update=np.random.choice(['median','no_update'])
        merging_criterium=np.random.choice(['tryall','closest'])
        init_topo=np.random.choice(['karger','random'],p=[0.9,0.1])
        
        maxfreq_mSTreg=np.random.randint(low=0,high=self.maxfreq_mSTreg)
        maxiter_mSTreg=np.random.randint(low=-1,high=self.maxiter_mSTreg)
        
        if init_topo=='karger':
            lamda_karger_temperature=np.quantile(self.tdata._knn_graph.data,np.random.uniform(0,0.4))
            karger_temperature=np.random.exponential(1/lamda_karger_temperature)
        else:
            karger_temperature=None

        self.tdata.compute_BCST(alpha=self.alpha,init_topo=init_topo,
                           maxfreq_mSTreg=maxfreq_mSTreg, maxiter_mSTreg=maxiter_mSTreg,
                           return_topo_CST=True,order_criterium = 'random', merging_criterium = merging_criterium,
                           criterium_BP_position_update = criterium_BP_position_update,
                           karger_graph = self.tdata._knn_graph,karger_temperature=karger_temperature,
                                filter_BP_from_solution=False)
        return self.tdata.trees['CST_%0.2f'%self.alpha].T
class TreeCost_cls:
    def __init__(self,T,cost):
        self.T=T.copy()
        self.cost=cost
    def __lt__(self, other):
        return self.cost<other.cost
    def __le__(self, other):
        return self.cost<=other.cost
    def __gt__(self, other):
        return not self.__le__(self,other)
    def __ge__(self, other):
        return not self.__lt__(self,other)

    def __repr__(self):
        return "Cost=%0.3f"%self.cost
class Ordered_Buffer:
    def __init__(self, buffer_capacity, throw_out_policy="worst",inv_temperature=0,
                 add_if_better=True):
        self.buffer_capacity = buffer_capacity
        self.throw_out_policy = throw_out_policy
        self._inv_temperature = inv_temperature

        self.add_if_better=add_if_better

        self.sample_list = []
        self.prob_sampling=[]



    def update_inv_temperature(self,new_inverse_temperature):
        self._inv_temperature = new_inverse_temperature
        self._define_prob_sampling()

    def define_prob_sampling(self):
        if self._inv_temperature==0:
            self.prob_sampling=None
        else:
            # probability of sampling increases with proximity of its sorted position to the first (argmin).
            # That is, lower cost  trees are more likely to be sampled
            self.prob_sampling = np.exp(-self._inv_temperature/len(self.sample_list) * np.arange(len(self.sample_list)))
            self.prob_sampling /= self.prob_sampling.sum()



    def add(self,TreeCost):
        """add instance of MemorySample class"""

        if len(self.sample_list) >= self.buffer_capacity:
            if self.add_if_better and self.sample_list[-1] < TreeCost:
                return

            if self.throw_out_policy == "random":
                self.sample_list.pop(np.random.randint(0, len(self.sample_list)))
            elif self.throw_out_policy == "worst":
                self.sample_list.pop()
            else:
                raise NotImplementedError(f"{self.throw_out_policy} is not available.")

        bisect.insort(self.sample_list,TreeCost)


    def sample(self):

        if self._inv_temperature==0 or len(self.sample_list) != len(self.prob_sampling):
            self.define_prob_sampling()


        idx = np.random.choice(np.arange(len(self.sample_list)), replace=False,p=self.prob_sampling)

        return self.sample_list[idx]

def SPT(A,center):

    if nk_available:
        if 'scipy.sparse' in str(type(A)):
            G = nk.nxadapter.nx2nk(nx.from_scipy_sparse_array(A, create_using=nx.Graph()), weightAttr='weight')
        elif 'networkx' in str(type(A)):
            G = nk.nxadapter.nx2nk(A, weightAttr='weight')
        else:
            G = A
        num_nodes=G.numberOfNodes()
        dijkstra=nk.distance.Dijkstra(G=G, source=center, storePaths=True, storeNodesSortedByDistance=True)
        dijkstra.run()
        T = sp.lil_matrix((num_nodes, num_nodes))
        for target in dijkstra.getNodesSortedByDistance()[::-1]:
            path=dijkstra.getPath(target)[::-1]
            for i in range(len(path[:-1])):
                u,v=path[i],path[i+1]
                if T[u,v]!=0:
                    break
                T[u,v]=T[v,u]=G.weight(u,v)

            if T.count_nonzero()>2*(num_nodes-1):
                break

    else:
        if 'scipy.sparse' in str(type(A)):
            pass
        else:
            A = nx.adjacency_matrix(A.to_undirected())

        D,predecessors = sp.csgraph.dijkstra(A, directed=False,indices=center, unweighted=False,return_predecessors=True)
        num_nodes=len(D)
        T = sp.lil_matrix((num_nodes, num_nodes))
        for node in np.argsort(-D):
            pred=predecessors[node]
            while pred>=0:
                if T[pred,node]!=0:
                    break
                T[pred,node]=T[node,pred]=A[pred,node]
            if T.count_nonzero()>2*(num_nodes-1):
                break

    return T

if __name__=='__main__':
    from sklearn.metrics import pairwise_distances
    from utils.visualization import  plot_graph
    import matplotlib.pyplot as plt
    np.random.seed(15)
    P=np.random.uniform(0,1,size=(50,2))

    D=pairwise_distances(P)
    # mask = np.random.uniform(0, 1, D.shape) > 0.97
    # D = spp.lil_matrix(mask * D)
    # D = D.maximum(D.T)



    #EXAMPLE PAPER
    P=np.array([[0,1],[-0.5,.5],[0,0],[.5,.5],[0.5,-0.5],[-1,0],[-0.5,-0.5],[0,-1]])
    D=sp.lil_matrix((8,8))
    D[0,1]=D[0,3]=D[6,7]=1
    D[1,2]=D[2,4]=D[5,6]=3
    D[1,5]=D[2,3]=D[2,6]=D[3,4]=D[4,7]=2
    D = D.maximum(D.T)

    k1=1
    k2=3

    plot_graph(sp.lil_matrix(D),P,title='G',figsize=(12,6),counter_plot=0,k1=k1,k2=k2)

    grasp=GRASP_relinking(D,4,500,500,500)
    T,cost=grasp.run()
    widths=centrality_weights_tree(T.tocsr(),max_width=8)
    plot_graph(T,P,widths=widths,title='Cost=%0.2f'%cost,counter_plot=1,k1=k1,k2=k2)
    # plt.tight_layout()
    # plt.show()

    from T_datacls.T_datacls import T_data
    edge_restriction=list(map(lambda x,y:(x,y),*sp.tril(D).nonzero()))
    tdata=T_data(P)
    tdata.compute_BCST(alpha=1,
                       maxfreq_mSTreg=3, maxiter_mSTreg=30)
    cost_mSTREG=Wiener_index(tdata.trees['CST_1.00'].T,alpha=1) / (len(P) ** 2)
    plot_graph(tdata.trees['CST_1.00'].T, P, widths=8*tdata.trees['CST_1.00'].widths,
               title='Cost=%0.5f' % cost_mSTREG, counter_plot=2, k1=k1, k2=k2)
    plt.tight_layout()
    plt.show()