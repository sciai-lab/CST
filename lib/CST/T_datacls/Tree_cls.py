from .utilities.graphtools import centrality_weights_tree, Wiener_index
import numpy as np

from ..methods.mSTreg.heuristics import remove_collapsedBP_from_solution
from ..methods.mSTreg.topology import adj_to_adj_sparse
from .utilities.visualisation import vis_data_coords4fulltopo


import ctypes
from numpy import ctypeslib as npct
import os


file_path = os.path.abspath(__file__)
# Extract the directory path
directory_file = os.path.dirname(file_path)


# load shared C-library for optimizer
opt_lib = npct.load_library("fast_optimizer", "%s/../methods/mSTreg/geometry_optimization/lib"%directory_file)

# define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

opt_lib.calculate_EW_BCST.argtypes = [ctypes.c_int,array_2d_double,array_2d_int,array_1d_double,ctypes.c_double,ctypes.c_double]

import scipy.sparse as spp
from copy import deepcopy

class Tree_():
	
	def shape(self):
		return self.T.shape
	
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


class Tree(Tree_):
	'''
	Class for storing the data of a CST solution
	'''
	
	def __init__(self, T, coords=None, widths=None, coords_vis=None, cost=None):
		self.T = T
		self.coords = coords
		# Upon plotting the tree, the edge widths are proportional to the centrality of the edges
		if widths is None:
			self.compute_widths()
		else:
			self.widths = widths
		if coords_vis is not None:
			self.coords_vis = coords_vis
		if cost is not None:
			self.cost = cost
	
	def get_Wiener_index(self, alpha=1, normalize=True):
		if normalize:
			return Wiener_index(self.T, alpha=alpha) / (self.T.shape[0] ** (2 * alpha))
		return Wiener_index(self.T, alpha=alpha)
	
	def compute_widths(self):
		self.T.sort_indices()
		self.widths = centrality_weights_tree(self.T, max_width=1)
	
	def get_T_weighted_by_widths(self):
		k = 0
		T_aux = spp.lil_matrix(self.T.shape)
		for i, j in zip(*spp.triu(self.T).nonzero()):
			T_aux[i, j] = T_aux[j, i] = self.widths[k]
			k += 1
		return T_aux.tocsr()
	
	def __repr__(self):
		txt2print = 'Tree: \n'
		txt2print += 'T: %ix%i \n' % self.T.shape
		
		if self.coords is not None:
			txt2print += 'P: %ix%i \n' % self.coords.shape
		if hasattr(self, 'coords_vis'):
			txt2print += 'coords_vis: %ix%i \n' % self.coords_vis.shape
		if hasattr(self, 'widths'):
			txt2print += 'widths\n'
		if hasattr(self, 'cost'):
			txt2print += 'cost: %f \n' % self.cost
		
		return txt2print


class Tree_SP(Tree):
	'''
	Class for storing the data of a BCST solution
	'''
	
	def __init__(self, adj, adj_flows, coords, cost, coords_vis=None, alpha=None,demands=None):
		self.adj = adj
		self.adj_flows = adj_flows
		self.coords = coords
		self.cost = cost
		self.num_terminals = len(coords) // 2 + 1
		
		self.demands=demands
		
		self.T = adj_to_adj_sparse(adj, coords=coords)
		
		# Order .nonzero() output in lexographical order (i.e. first by row, then by column)
		# This is more convenient for ordering the edges in other functions.
		self.T.sort_indices()
		
		# Store BCST centralities for visualization
		self.widths = spp.triu(adj_to_adj_sparse(adj, flows=adj_flows)).data
		self.widths /= np.max(self.widths)
		
		# get projection for visualization for the BPs
		if coords_vis is not None and alpha is not None:
			self.set_coords_vis(coords_vis, alpha=alpha)
	
	def compute_coords_vis(self, Pvis, alpha):
		return vis_data_coords4fulltopo(self.adj, Pvis, alpha)
	
	def set_coords_vis(self, Pvis, alpha):
		self.coords_vis = self.compute_coords_vis(Pvis, alpha=alpha)
	
	def filter_BP(self, threshold_filter_collapse=1e-5):
		
		self.T_bp_filtered, self.coords_filtered, self.widths_filtered, self.idxs_filtered = remove_collapsedBP_from_solution(
			self.adj,
			self.adj_flows,
			self.coords,
			threshold_filter_collapse)
		self.widths_filtered = np.array(self.widths_filtered)
		self.widths_filtered /= np.max(self.widths_filtered)
		if hasattr(self, 'coords_vis'):
			self.coords_vis_filtered = self.coords_vis[self.idxs_filtered]
	
	def get_Wiener_index(self, alpha=1):
		# widths are normalized to 1 in init, here we recompute the inverse normalization factor
		inv_norm_factor = ((self.num_terminals - 1) / (self.num_terminals ** 2)) / min(self.widths)
		return np.dot(spp.triu(self.T).data, (self.widths * inv_norm_factor) ** alpha)
	
	def get_T_weighted_by_flows(self, filtered=False):
		if not filtered:
			return adj_to_adj_sparse(self.adj, flows=self.adj_flows)
		else:
			if not hasattr(self, 'T_bp_filtered'):
				self.filter_BP()
			k = 0
			T_aux = spp.lil_matrix(self.T_bp_filtered.shape)
			max_flows = np.max(self.adj_flows)
			flows = max_flows * self.widths_filtered
			for i, j in zip(*spp.triu(self.T_bp_filtered).nonzero()):
				T_aux[i, j] = T_aux[j, i] = flows[k]
				k += 1
			return T_aux.tocsr()
		
	def update_flows(self, demands=None):
		if demands is None:
			if self.demands is None:
				supply_array = np.array([1 / self.num_terminals])  # np.array([(nsites-1)/nsites])
				demand_array = -np.ones(self.num_terminals - 1) / self.num_terminals
				demands = np.append(supply_array, demand_array)
			else:
				demands = self.demands
		
		
		opt_lib.calculate_EW_BCST(self.num_terminals,self.adj_flows, self.adj, demands, 1, 1)
		
		#update widths
		self.widths = spp.triu(adj_to_adj_sparse(self.adj, flows=self.adj_flows)).data
		self.widths /= np.max(self.widths)
		
		#update coords_vis
		if hasattr(self, 'coords_vis'):
			self.set_coords_vis(self.coords_vis, alpha=1)
		
		#update filtered
		if hasattr(self, 'T_bp_filtered'):
			self.filter_BP()
		
		#update cost
		if self.alpha!=None:
			self.cost = self.get_Wiener_index(alpha=self.alpha)
	
	def __repr__(self):
		txt2print = '%s: \n' % self.__class__.__name__
		txt2print += 'T: %ix%i \n' % self.T.shape
		txt2print += 'num_terminals: %i \n' % self.num_terminals
		txt2print += 'adj: %ix%i \n' % self.adj.shape
		txt2print += 'adj_flows: %ix%i \n' % self.adj_flows.shape
		
		if self.coords is not None:
			txt2print += 'coords: %ix%i \n' % self.coords.shape
		if hasattr(self, 'coords_vis'):
			txt2print += 'coords_vis: %ix%i \n' % self.coords_vis.shape
		if self.widths is not None:
			txt2print += 'widths \n'
		if self.cost is not None:
			txt2print += 'cost: %f \n' % self.cost
		
		if hasattr(self, 'T_bp_filtered'):
			txt2print += 'T_bp_filtered: %ix%i \n' % self.T_bp_filtered.shape
		if hasattr(self, 'coords_filtered'):
			txt2print += 'coords_filtered: %ix%i \n' % self.coords_filtered.shape
		if hasattr(self, 'idxs_filtered'):
			txt2print += 'idxs_filtered: len=%i \n' % len(self.idxs_filtered)
		if hasattr(self, 'widths_filtered'):
			txt2print += 'widths_filtered \n'
		
		return txt2print
