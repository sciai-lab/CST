import numpy as np
import networkx as nx
from queue import PriorityQueue
from random import shuffle
from pqdict import pqdict
from copy import deepcopy
def get_orderBPcls(order_criterium,T_dict,coords, num_terminals):
    '''
    Determine order in which BPs are merged
    :param order_criterium:
        - 'closest'-> the one with closest neightbor goes first
        - 'furthest'-> the one with furthest neightbor goes first
        - 'default'-> merges them in increasing order of index
        - 'random'-> merges them randomly
        - 'closestterminals'-> the one closest to a terminal which is its neighbor goes first
        - 'fewest_neigh'-> the one with fewer neighbors goes first
    :param T_nx:
    :param coords:
    :param num_terminals:
    :return:
    '''
    if order_criterium.lower()== 'closest':
        return OrderBP4removal_closest(T_dict,coords,num_terminals)
    elif order_criterium.lower()== 'furthest':
        return OrderBP4removal_furthest(T_dict,coords,num_terminals)
    elif order_criterium.lower() == 'default':
        return OrderBP4removal_default(T_dict, num_terminals)
    elif order_criterium.lower() == 'random':
        return OrderBP4removal_random(T_dict, num_terminals)
    elif order_criterium.lower() == 'closestterminals':
        return OrderBP4removal_closestterminals(T_dict,coords,num_terminals)
    elif order_criterium.lower() == 'lowestdegree':
        return OrderBP4removal_lowestdegree(T_dict,num_terminals)
    else:
        raise NotImplementedError('order criterium to remove BPs "%s" not implemented'%order_criterium)
    

class OrderBP4removal():

    def init_queue(self, **kwargs):
        pass

    def put(self, **kwargs):
        pass

    def __getitem__(self, i):
        return self.Q[i]

    def pop(self):
        return self.Q.pop()

    def popitem(self):
        return self.Q.popitem()
    def restore_values(self,dict2restore):
        for k,v in dict2restore.items():
            self.Q[k]=v


    def empty(self):
        return len(self.Q)==0
    def __len__(self):
        return len(self.Q)


    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


class OrderBP4removal_closest(OrderBP4removal):
    '''
    Determine order in which BPs are merged. The one with closest neightbor goes first.

    The queue is a priority queue, where the priority is given by the distance to the closest terminal, in case of a
    tie, the one with fewer neighbors goes first. Each value of the priority queue is a tuple (distance, number of neighbors)
    The keys are the branching points.
    '''
    
    def __init__(self, T_dict, coords, num_terminals):
        self.n = num_terminals
        fulltopo = len(T_dict) == (num_terminals * 2 - 2)
        self.init_queue(coords, T_dict, fulltopo=fulltopo)
    
    def init_queue(self, coords, T_dict, fulltopo=True):
        self.Q = pqdict({})
        self.closest_neigbor = {}
        if fulltopo:
            node_indices = np.arange(self.n, len(coords))
            neighbors = [T_dict[BP] for BP in node_indices]
            coord_diff = coords[node_indices, np.newaxis] - coords[neighbors]
            norm_diff = np.linalg.norm(coord_diff, axis=2)
            min_indices = np.argmin(norm_diff, axis=1)
            for i, BP in enumerate(node_indices):
                # in fulltopo each BP has degree 3
                self.Q[BP] = (norm_diff[i, min_indices[i]], 3)
                self.closest_neigbor[BP] = neighbors[i][min_indices[i]]
        else:
            for BP in range(self.n, len(coords)):
                if BP not in T_dict:
                    continue
                neighbors_BP = T_dict[BP]
                D2BP = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1)
                
                idx = np.argmin(D2BP)
                self.Q[BP] = (D2BP[idx], len(neighbors_BP))
                self.closest_neigbor[BP] = neighbors_BP[idx]
    
    def put(self, coords, T_dict, merging_node, removed_node, **kwargs):
        D2node = np.linalg.norm(coords[merging_node] - coords[T_dict[merging_node]], axis=1)
        updated = False
        for node, d in zip(T_dict[merging_node], D2node):
            if node >= self.n:
                if self.closest_neigbor[node] == removed_node:
                    D2node_aux = np.linalg.norm(coords[node] - coords[T_dict[node]], axis=1)
                    idx = np.argmin(D2node_aux)
                    self.Q[node] = (D2node_aux[idx], len(T_dict[node]))
                    self.closest_neigbor[node] = T_dict[node][idx]
                else:
                    if self.Q[node] > (d, len(T_dict[node])):
                        self.Q[node] = (d, len(T_dict[node]))
                        self.closest_neigbor[node] = merging_node
                if merging_node >= self.n:
                    if updated:
                        if self.Q[merging_node] > (d, len(T_dict[merging_node])):
                            self.Q[merging_node] = (d, len(T_dict[merging_node]))
                            self.closest_neigbor[merging_node] = node
                    else:
                        self.Q[merging_node] = (d, len(T_dict[merging_node]))
                        self.closest_neigbor[merging_node] = node
                        updated = True


class OrderBP4removal_furthest(OrderBP4removal):
    def __init__(self, T_dict, coords, num_terminals):
        self.n = num_terminals
        fulltopo = len(T_dict) == (num_terminals * 2 - 2)
        self.init_queue(coords, T_dict, fulltopo=fulltopo)
    
    def init_queue(self, coords, T_dict, fulltopo=True):
        self.Q = pqdict({})
        self.furthest_neighbor = {}
        if fulltopo:
            node_indices = np.arange(self.n, len(coords))
            neighbors = [T_dict[BP] for BP in node_indices]
            coord_diff = coords[node_indices, np.newaxis] - coords[neighbors]
            norm_diff = np.linalg.norm(coord_diff, axis=2)
            max_indices = np.argmax(norm_diff, axis=1)
            for i, BP in enumerate(node_indices):
                self.Q[BP] = (norm_diff[i, max_indices[i]], 3)
                self.furthest_neighbor[BP] = neighbors[i][max_indices[i]]
        else:
            for BP in range(self.n, len(coords)):
                if BP not in T_dict:
                    continue
                neighbors_BP = T_dict[BP]
                D2BP = np.linalg.norm(coords[BP] - coords[neighbors_BP], axis=1)
                
                idx = np.argmax(D2BP)
                self.Q[BP] = (-D2BP[idx], len(neighbors_BP))
                self.furthest_neighbor[BP] = neighbors_BP[idx]
    
    def put(self, coords, T_dict, merging_node, removed_node, **kwargs):
        D2node = np.linalg.norm(coords[merging_node] - coords[T_dict[merging_node]], axis=1)
        updated = False
        for node, d in zip(T_dict[merging_node], -D2node):
            if node >= self.n:
                
                if self.furthest_neighbor[node] == removed_node:
                    D2node_aux = np.linalg.norm(coords[node] - coords[T_dict[node]], axis=1)
                    idx = np.argmax(D2node_aux)
                    self.Q[node] = (-D2node_aux[idx], len(T_dict[node]))
                    self.closest_neigbor[node] = T_dict[node][idx]
                else:
                    if self.Q[node] > (d, len(T_dict[node])):
                        self.Q[node] = (d, len(T_dict[node]))
                        self.furthest_neighbor[node] = merging_node
                if merging_node >= self.n:
                    if updated:
                        if self.Q[merging_node] > (d, len(T_dict[merging_node])):
                            self.Q[merging_node] = (d, len(T_dict[merging_node]))
                            self.furthest_neighbor[merging_node] = node
                    else:
                        self.Q[merging_node] = (d, len(T_dict[merging_node]))
                        self.furthest_neighbor[merging_node] = node
                        updated = True

class OrderBP4removal_default(OrderBP4removal):
    def __init__(self, T_dict, num_terminals,**kwargs):
        self.n = num_terminals
        
        self.Q = [BP for BP in T_dict.keys() if BP >= self.n]



class OrderBP4removal_random(OrderBP4removal):
    def __init__(self,T_dict, num_terminals,**kwargs):
        self.n = num_terminals

        self.Q=[BP for BP in T_dict.keys() if BP>=self.n]
        shuffle(self.Q)


class OrderBP4removal_closestterminals(OrderBP4removal):
    '''
    Determine order in which BPs are merged. The one closest to a terminal which is its neighbor goes first.
    In case of a tie, the one with fewer neighbors goes first.

    The queue is a priority queue, where the priority is given by the distance to the closest terminal, in case of a
    tie, the one with fewer neighbors goes first. Each value of the priority queue is a tuple (distance, number of neighbors)
    The keys are the branching points.
    '''
    
    def __init__(self, T_dict, coords, num_terminals):
        self.n = num_terminals
        
        fulltopo = len(T_dict) == (num_terminals * 2 - 2)
        self.init_queue(coords, T_dict, fulltopo=fulltopo)
    
    def init_queue(self, coords, T_dict, fulltopo=True):
        # init pqdict with inf values. First element value represents distance to closest terminal, second element represents number of neighbors
        self.Q = pqdict({i: (np.inf, np.inf) for i in range(self.n, len(coords))})
        if fulltopo:
            neighbors = [T_dict[terminal][0] for terminal in range(self.n)]
            norm_diff = np.linalg.norm(coords[:self.n] - coords[neighbors], axis=1)
            for i, BP in enumerate(neighbors):
                self.Q[BP] = min(self.Q[BP], (norm_diff[i], 3))
        else:
            for node in range(self.n):
                for neighbor in T_dict[node]:
                    if neighbor >= self.n:
                        self.Q[neighbor] = (np.linalg.norm(coords[node] - coords[neighbor]), len(T_dict[neighbor]))
    
    def put(self, coords, T_dict, merging_node, **kwargs):
        neighbors_idx_closest_neighbor = T_dict[merging_node] + [merging_node]
        for neighbor_merging_node in neighbors_idx_closest_neighbor:
            if neighbor_merging_node >= self.n:
                updated = False
                for neighbor in T_dict[neighbor_merging_node]:
                    if neighbor < self.n:
                        if updated:
                            distance = np.linalg.norm(coords[neighbor_merging_node] - coords[neighbor])
                            num_neighbors = len(T_dict[neighbor_merging_node])
                            self.Q[neighbor_merging_node] = min(self.Q[neighbor_merging_node],
                                                                (distance, num_neighbors))
                        else:
                            distance = np.linalg.norm(coords[neighbor_merging_node] - coords[neighbor])
                            num_neighbors = len(T_dict[neighbor_merging_node])
                            self.Q[neighbor_merging_node] = (distance, num_neighbors)
                            updated = True



class OrderBP4removal_lowestdegree(OrderBP4removal):
    def __init__(self, T_dict, num_terminals):
        self.n = num_terminals
        self.Q = pqdict({})
        BPs_ls = list(T_dict.keys())
        shuffle(BPs_ls)
        for BP in T_dict.keys():
            if BP>=self.n:
                self.Q[BP] = len(T_dict[BP])
    
    def put(self, T_dict, merging_node, **kwargs):
        if merging_node >= self.n:
            self.Q[merging_node] = len(T_dict[merging_node])