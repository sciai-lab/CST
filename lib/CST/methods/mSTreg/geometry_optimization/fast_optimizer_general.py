import ctypes
import numpy as np
from numpy import ctypeslib as npct
import scipy.sparse as sp
import os
import warnings

file_path = os.path.abspath(__file__)
# Extract the directory path
directory_file = os.path.dirname(file_path)

# load shared C-library for optimizer
opt_lib = npct.load_library("fast_optimizer_general", "%s/lib"%directory_file)

# define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')
array_1d_int = npct.ndpointer(dtype=np.intc, ndim=1, flags='CONTIGUOUS')

opt_lib.iterations.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_1d_int, array_1d_int,
                               array_1d_double, array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_int,
                               ctypes.c_double,
                               ctypes.c_bool, ctypes.c_double, ctypes.c_bool]
opt_lib.iterations.restype = ctypes.c_double


def fast_optimize_general(T, coords_terminals, al, improv_threshold=1e-7,
                          demands=None, init_BP_coords=None, EW=None, beta=1, ):
	'''
	Optmizes the coordinates of the Branching/Steiner points of topologies which are not necessarily full tree topologies,
	i.e. where the BPs may have degree hihger than 3.
	:param T: sparse adjacency matrix of the topology
	:param coords_terminals: numpy array of coordinates of the terminals
	:param al: alpha value
	:param improv_threshold:
	:param demands:
	:param init_BP_coords:
	:param EW:
	:param beta:
	:return:
	'''
	
	num_terminals = len(coords_terminals)
	NUM_BPS = T.shape[0] - num_terminals
	
	if demands is None:
		supply_array = np.array([1 / num_terminals])  # np.array([(nsites-1)/nsites])
		demand_array = -np.ones(num_terminals - 1) / num_terminals
		demands = np.append(supply_array, demand_array)
	
	subT = T[:num_terminals][:, :num_terminals]
	# check terminals are leaves. If one of them is not adapt demands
	if subT.sum() != 0:
		warnings.warn('Terminals are not leaves. Demands will be adapted. This may lead to unexpected results. NOT WORKING PROPERLY')
		T_ = T.tolil(copy=True)
		T_aux = T.tolil(copy=True)
		for i, j in subT.nonzero():
			if i > j:
				continue
			# disconnect T_
			T_[i, j] = 0
			T_[j, i] = 0
			
			# disconnect T_aux
			T_aux[i, j] = 0
			T_aux[j, i] = 0
			
			# compute connected components
			cc_ = sp.csgraph.connected_components(T_aux, directed=False)[1][j]
			
			# compute demands on each side of the cut
			demands[i] = demands[np.where(cc_ == cc_[j])].sum()
			demands[j] = demands[np.where(cc_ == cc_[i])].sum()
			# reconnect T_aux
			T_aux[i, j] = T_aux[j, i] = 1
		
		T_ = T_.tocsr()
		indptr = T_.indptr[num_terminals:].astype(np.intc) - num_terminals
		indices = T_.indices[num_terminals:].astype(np.intc)
	else:
		indptr = T.indptr[num_terminals:].astype(np.intc) - num_terminals
		indices = T.indices[num_terminals:].astype(np.intc)
	
	# dimension euclidean space
	dim = len(coords_terminals[0])
	# number of terminals
	
	# assign initial BP positions
	if init_BP_coords is not None:
		assert (init_BP_coords.shape == (NUM_BPS, dim))
		coords_arr = np.vstack((coords_terminals, init_BP_coords))
		use_init = False
	else:
		coords_arr = np.vstack((coords_terminals, np.random.rand(NUM_BPS, dim)))
		use_init = True
	
	coords_arr = coords_arr.flatten()
	
	# construct data arrays for optimizer (Edge Weights
	if EW is None:
		EW = np.array(np.zeros(2 * (NUM_BPS - 1) + num_terminals)).astype(np.double)
		EW_given = False
	else:
		EW_given = True
	
	# construct data arrays for optimizer (demands)
	
	# additional output variables
	iter = ctypes.c_int(0)
	
	# run optimization
	cost = opt_lib.iterations(ctypes.byref(iter), dim, num_terminals, indices, indptr, EW, demands, coords_arr, al,
	                          NUM_BPS, improv_threshold,
	                          EW_given, beta, use_init)
	
	# reshape P_Zheng array into original shape
	coords_arr = coords_arr.reshape((-1, dim))
	
	T_W = sp.csr_matrix((EW, indices, np.concatenate((np.zeros(num_terminals), indptr))),
	                    shape=(num_terminals + NUM_BPS, num_terminals + NUM_BPS))
	T_W = T_W.maximum(T_W.T)
	widths = sp.triu(T_W).data
	return T, cost, coords_arr, widths, EW, iter.value
