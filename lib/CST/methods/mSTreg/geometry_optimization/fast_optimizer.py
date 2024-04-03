import ctypes
import numpy as np
from numpy import ctypeslib as npct

from ..topology import topology
import os


file_path = os.path.abspath(__file__)
# Extract the directory path
directory_file = os.path.dirname(file_path)


# load shared C-library for optimizer
opt_lib = npct.load_library("fast_optimizer", "%s/lib"%directory_file)

# define input and output types for C-function
array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_2d_int = npct.ndpointer(dtype=np.intc, ndim=2, flags='CONTIGUOUS')

opt_lib.iterations.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_2d_int, array_2d_double,
                               array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_bool, ctypes.c_double, ctypes.c_bool,ctypes.c_double]
opt_lib.iterations.restype = ctypes.c_double

opt_lib.calculate_EW_BCST.argtypes = [ctypes.c_int, array_2d_double, array_2d_int,array_2d_double,ctypes.c_double, ctypes.c_double]

opt_lib.calculate_EW_BOT.argtypes = [ctypes.c_int, array_2d_double, array_2d_int,array_2d_double,ctypes.c_double]



opt_lib.iterations_BOT.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, array_2d_int, array_2d_double,
                               array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double,
                               ctypes.c_bool]
opt_lib.iterations_BOT.restype = ctypes.c_double


def fast_optimize(itopo: topology, coords_terminals, al, improv_threshold=1e-7,
                  demands=None, init_BP_coords=None, EW=None,beta=1,factor_terminal=1.):
    if isinstance(itopo, topology):
        adj = itopo.adj
    elif isinstance(itopo, np.ndarray):
        adj = itopo

    # dimension euclidean space
    dim = len(coords_terminals[0])
    # number of terminals
    nsites = len(coords_terminals)

    # assign initial BP positions
    if init_BP_coords is not None:
        assert (init_BP_coords.shape == (nsites - 2, dim))
        coords_arr = np.vstack((coords_terminals, init_BP_coords))
        not_use_given_init=False
    else:
        coords_arr = np.vstack((coords_terminals, np.random.rand(nsites - 2, dim)))
        not_use_given_init=True
    coords_arr = coords_arr.flatten()

    # construct data arrays for optimizer (Edge Weights
    if EW is None:
        EW = np.array(np.zeros((nsites - 2, 3))).astype(np.double)
        EW_given = False
    else:
        EW_given = True

    # construct data arrays for optimizer (demands)
    if demands is None:
        supply_array = np.array([1/nsites])#np.array([(nsites-1)/nsites])
        demand_array = -np.ones(nsites-1)/nsites
        demands=np.append(supply_array,demand_array)


    # additional output variables
    iter = ctypes.c_int(0)

    # run optimization
    cost = opt_lib.iterations(ctypes.byref(iter), dim, nsites, adj, EW, demands, coords_arr, al, improv_threshold,
                              EW_given,beta,not_use_given_init,factor_terminal)

    # reshape P array into original shape
    coords_arr = coords_arr.reshape((-1, dim))

    return itopo, cost, coords_arr, EW, iter.value




def fast_optimize_bot(itopo: topology, coords_terminals, al, improv_threshold=1e-7,
                  demands=None, init_BP_coords=None, EW=None,beta=1,):
    if isinstance(itopo, topology):
        adj = itopo.adj
    elif isinstance(itopo, np.ndarray):
        adj = itopo

    # dimension euclidean space
    dim = len(coords_terminals[0])
    # number of terminals
    nsites = len(coords_terminals)

    # assign initial BP positions
    if init_BP_coords is not None:
        assert (init_BP_coords.shape == (nsites - 2, dim))
        coords_arr = np.vstack((coords_terminals, init_BP_coords))
    else:
        coords_arr = np.vstack((coords_terminals, np.random.rand(nsites - 2, dim)))

    coords_arr = coords_arr.flatten()

    # construct data arrays for optimizer (Edge Weights
    if EW is None:
        EW = np.array(np.zeros((nsites - 2, 3))).astype(np.double)
        EW_given = False
    else:
        EW_given = True



    # additional output variables
    iter = ctypes.c_int(0)

    # run optimization
    cost = opt_lib.iterations_BOT(ctypes.byref(iter), dim, nsites, adj, EW, demands, coords_arr, al, improv_threshold,
                              EW_given)

    # reshape P array into original shape
    coords_arr = coords_arr.reshape((-1, dim))

    return itopo, cost, coords_arr, EW, iter.value



def compute_EW_BCST(topology,demands=None,beta=1):
    import scipy.sparse as sp
    if isinstance(topology, topology):
        adj = topology.adj
    elif isinstance(topology, np.ndarray):
        adj = topology
        assert (adj.shape[1] == 3)
    elif sp.issparse(topology):
        from ..topology.prior_topology import sp_to_adj
        adj = sp_to_adj(topology)
        
    # number of terminals
    nsites = len(adj)+2

    # construct data arrays for optimizer (Edge Weights
    EW = np.array(np.zeros((nsites - 2, 3))).astype(np.double)

    # construct data arrays for optimizer (demands)
    if demands is None:
        supply_array = np.array([1/nsites])#np.array([(nsites-1)/nsites])
        demand_array = -np.ones(nsites-1)/nsites
        demands=np.append(supply_array,demand_array)


    # Compute EW
    opt_lib.calculate_EW_BCST(nsites, EW, adj, demands, 1,beta)
    return EW



def compute_EW_BOT(topology, demands=None):
    import scipy.sparse as sp
    if isinstance(topology, topology):
        adj = topology.adj
    elif isinstance(topology, np.ndarray):
        adj = topology
        assert (adj.shape[1] == 3)
    elif sp.issparse(topology):
        from ..topology.prior_topology import sp_to_adj
        adj = sp_to_adj(topology)
    
    # number of terminals
    nsites = len(adj) + 2
    
    # construct data arrays for optimizer (Edge Weights
    EW = np.array(np.zeros((nsites - 2, 3))).astype(np.double)
    
    # construct data arrays for optimizer (demands)
    if demands is None:
        supply_array = np.array([1 / nsites])  # np.array([(nsites-1)/nsites])
        demand_array = -np.ones(nsites - 1) / nsites
        demands = np.append(supply_array, demand_array)
    
    # Compute EW
    opt_lib.calculate_EW_BOT(nsites, EW, adj, demands, 1)
    return EW

