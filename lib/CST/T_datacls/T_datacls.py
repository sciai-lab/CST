import os
import logging
logging.basicConfig(level = logging.INFO)
from ..utils.utils import factor_number
from .utilities import plot_graph
import matplotlib.pyplot as plt
import numpy as np
import pickle
from .utilities.graphtools import ensure_connected_knn_graph

from ..methods.mSTreg.heuristics import compute_BCST,sparsemat2fulltopoadj
from ..methods.mSTreg.topology.prior_topology import random_bin_tree,incremental_star,sp_to_adj
from ..methods.mSTreg.topology import topology as FullTopology_cls

from functools import wraps

import scipy.sparse as spp

from scipy.spatial.distance import pdist, squareform
import time

from .Tree_cls import Tree,Tree_SP

def timeit(func):
    """
		Decorator function to measure the execution time of the wrapped function.

		Parameters:
			func (callable): The function to be wrapped and measured.

		Returns:
			callable: A wrapped function that measures the execution time of the original function.

		Usage:
			- Decorate the desired function with `@timeit` to measure its execution time.
			- The wrapped function will log the execution time using the logger instance `self.logger.info()`.
			- The original function's result will be returned as-is.
	"""

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        """
			Wrapper function that measures the execution time of the wrapped function.

			Parameters:
				*args: Positional arguments to be passed to the wrapped function.
				**kwargs: Keyword arguments to be passed to the wrapped function.

			Returns:
				Any: The result returned by the wrapped function.

			Notes:
				The wrapper function logs the execution time using the logger instance `self.logger.info()`.
				It measures the time taken by the wrapped function to execute and returns the result.
		"""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        args[0].logger.info(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


class T_data():
    def __init__(self,X,X_vis=None,labels=None,verbose=True):
        self.verbose=verbose
        self.logger = logging.getLogger('Tdata_logger')
        if verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)


        self.X=X
        self.logger.info("Added  'X' to data\n")
        if X.shape[1]!=2:
            self.X_vis = X_vis
            if self.X_vis is not None:
                self.logger.info("Added  'X_vis' to data\n")
            self.BCST_coords_vis = {}
        else:
            self.X_vis=None
            self.BCST_coords_vis = None


        self.trees={}
        self.labels=labels
        self.widths_trees={}
        self.costs={}
        self.BCST_coords={}


        self._max_width_edge_tree = 8



    def set_logging_level(self,level):
        """
           Set the logging level for the logger instance.

           Parameters:
               level (int or str): The logging level to be set. It can be specified as an integer or a string.
                   Possible logging levels (in increasing order of severity):
                       - DEBUG: Detailed information, typically useful only for diagnosing problems. (Value: 10)
                       - INFO: Confirmation that things are working as expected. (Value: 20)
                       - WARNING: Indication of a potential issue or a deprecated feature usage. (Value: 30)
                       - ERROR: An error occurred that may prevent the application from functioning properly. (Value: 40)
                       - CRITICAL: A critical error occurred that may result in the application's termination. (Value: 50)
                   You can also use the corresponding string values instead of the integer values.

           Returns:
               None
           """
        self.logger.setLevel(level)


    def __repr__(self):

        txt2print ='X: %ix%i \n'%(self.X.shape[0],self.X.shape[1])
        if self.X_vis is not None:
            txt2print += 'X_vis: %ix%i \n' % (self.X_vis.shape[0], self.X_vis.shape[1])
        for k,v in self.__dict__.items():
            if isinstance(v, dict):
                if len(v.keys())>0:
                    txt2print+=f"\n    {k}: {str(list(v.keys()))[1:-1]}"

        return txt2print

    def __str__(self):
        return self.__repr__()

    
    @timeit
    def compute_BCST(self, alpha, maxiter_mSTreg=10,
                     maxfreq_mSTreg=3, return_topo_CST=True, init_topo='mST', demands=None,
                     order_criterium = 'closest', merging_criterium = 'tryall',
                     criterium_BP_position_update = 'median', compute_CST_each_iter=True,
                     mST_fromknn=True,filter_BP_from_solution=True,threshold_filter_collapse=1e-5,
                     ori_input_graph=None,beta=1,factor_terminal=1,karger_graph=None,karger_temperature=1):
        '''
		Compute the BCST (Branched Central Spanning Tree) and CST (Central Spanning Tree) for a given alpha.

        Parameters:
            alpha (float): Parameter for the computation.
            maxiter_mSTreg (int): Maximum number of iterations for the heuristic. Default is 10.
            maxfreq_mSTreg (int): -> maximum number of extra points added to an edge.
                                    The extermes of the nodes of the edge are counted as nodes to add, though these always are part. That is,
                                    If maxfreq is equal to k, then k-2 points equidistant points are sampled from the edge.
                                  Default is 5.
            return_topo_CST (bool): If True, compute the CST as well. Default is True.
            init_topo (str, np.ndarray, scipy.sparse array): Initial topology guess.
                - If str, it can be 'mST' (minimum spanning tree), 'random' or 'incremental_star'.
                - If np.ndarray, a 2D array of shape (n-2, 3), where n is the number of terminals.
                  Each row represents a branching point and the columns are neighboring nodes.
                - If scipy.sparse array, it is the adjacency matrix of the initial topology (without BPs).
                  Default is 'mST'.
            demands (np.ndarray): Demands of each terminal. If None, all terminals have demand 1, as is the case for the (B)CST.
                                  Default is None.
            order_criterium (str): Criteria for ordering the branching points for merging when computing a CST topology from
                                    a BCST topology.
                Options:
                    - 'closest': Merge the one with the closest neighbor first.
                    - 'furthest': Merge the one with the furthest neighbor first.
                    - 'default': Merge them in increasing order of index.
                    - 'random': Merge them randomly.
                    - 'closestterminals': Merge the one closest to a terminal neighbor first.
                                  Default is 'closestterminals'.
            merging_criterium (str): Criteria for merging the branching points to a neighbor when computing a CST topology from
                                    a BCST topology.
                Options:
                    - 'closest': Merge to the closest neighbor.
                    - 'minflowcost': Merge to the neighbor connected by the edge with the lowest cost*flow**alpha.
                    - 'tryall': Try all possibilities and choose the one with the lowest cost
                                given the current branching point positions.
                Default is 'tryall'.
            criterium_BP_position_update (str): Criteria for updating the positions of the branching points  when computing a CST topology from
                                    a BCST topology.
                Options:
                    - 'median': Assign the geometric median as the merged branching point's position
                                with respect to the neighboring nodes.
                    - 'no_update': Do not update the position of the branching point.
                Default is 'median'.
            compute_CST_each_iter (bool): If True and return_topo_CST is True, compute the CST at each iteration
                                          of the heuristic. Default is True.
            mST_fromknn (bool): If True, approximate the mST computed in the iterations of the mSTreg heuristic
                                from the k-nearest neighbors graph. Default is True.
            filter_BP_from_solution (bool): If True, filter out branching points that are too close to each other.
                                            Default is True.
            threshold_filter_collapse (float): Threshold for filtering out branching points that are too close to each other.
        Returns:
            The computed BCST and CST if return_topo_CST is True.
        '''

        BCST_txt='BCST_%0.2f' % alpha
        CST_txt = 'CST_%0.2f' % alpha

        init_CST_cost = np.inf

        # Logic for initializing the initial topology based on the provided init_topo parameter
        if isinstance(init_topo,str):
            # If init_topo is a string, compute the initial topology based on the specified method ('mST' or 'random')
            self.logger.info("Computing initial topology BCST %s" %init_topo)
            if init_topo =='mST':
                if ori_input_graph is not None:
                    init_topo = spp.csgraph.minimum_spanning_tree(ori_input_graph)
                    init_topo = init_topo.maximum(init_topo.T)
                else:
                    if not 'mST' in self.trees.keys():
                        self.minimum_spanning_tree()
                    if return_topo_CST:
                        init_topo_CST = self.trees['mST'].T.copy()
                        init_CST_cost = self.trees['mST'].get_Wiener_index(alpha=alpha)
                    init_topo=self.trees['mST'].T
                init_topo = sparsemat2fulltopoadj(init_topo, coords=self.X)
            elif init_topo=='random':
                #returns adj
                init_topo = random_bin_tree(len(self.X))
            elif init_topo=='incremental_star':
                init_topo=incremental_star(coords_terminals=self.X, alpha=alpha, demands=demands, return_intermediate_trees=False)
                init_topo = sp_to_adj(init_topo)
            elif init_topo == 'karger':
                # get IMST initialization for given bot_problem, optional interpolation parameter beta (e.g. beta = 1: MST)
                if karger_graph is None:
                    if hasattr(self,'_knn_graph'):
                        karger_graph = self._knn_graph
                    else:
                        self._knn_graph = ensure_connected_knn_graph(self.X, num_neighs=int(2 * np.log(len(self.X))))
                        karger_graph= self._knn_graph


                init_topo = karger_init(A=karger_graph,
                                        temperature=karger_temperature)
                if return_topo_CST:
                    init_CST_cost= Wiener_index(init_topo, alpha=alpha)/(self.X.shape[0]**(2*alpha))
                    init_topo_CST = init_topo.copy()
                init_topo = sparsemat2fulltopoadj(init_topo, coords=self.X)
            else:
                raise NotImplementedError('%s init topo not implemented'%init_topo)

        elif spp.isspmatrix(init_topo):
            # If init_topo is a sparse matrix, use it as the initial topology (without BPs)
            self.logger.info("Precomputed initial topology")
            assert (init_topo.shape == (len(self.X),len(self.X)))
            if return_topo_CST:
                edges=init_topo.nonzero()
                for i,j in zip(*edges):
                    if i>j:
                        init_topo[j,i]=init_topo[i,j]=np.linalg.norm(self.X[i]-self.X[j])
                init_topo_CST=init_topo.copy()
                init_CST_cost = Wiener_index(init_topo, alpha=alpha) / (self.X.shape[0] ** (2 * alpha))
            init_topo = sparsemat2fulltopoadj(init_topo, coords=self.X)

        elif isinstance(init_topo,np.ndarray):
            # If init_topo is a numpy array of size num_terminals x 3, use it as the initial full tree topology.
            # Each row indicates the neighbors of a branching point.
            assert(init_topo.shape==(len(self.X)-2,3))
            self.logger.info("Precomputed initial topology")
            init_topo=FullTopology_cls(adj=init_topo)
        
        
        
        # run local search improvment
        # maxiter: maximum number of topologies looked at overall
        # num_tries: maximum number of topologies looked at for every sampled edge
        
        scale=np.max(np.max(self.X,axis=1)-np.min(self.X,axis=1))

        if return_topo_CST:
            self.logger.info("Computing %s and %s" % (BCST_txt, CST_txt))
            # Compute BCST and CST topologies, along with other related information
            BCST_topo, CST_topo, BCST_cost, CST_cost, BCST_coords_, BCST_EW = compute_BCST(init_topo, alpha,
                                                                                 coords_terminals=self.X/scale,
                                                                                 maxiter_mSTreg=maxiter_mSTreg,
                                                                                 maxfreq_mSTreg=maxfreq_mSTreg,
                                                                                 return_CST=True,
                                                                                 order_criterium=order_criterium,
                                                                                 merging_criterium=merging_criterium,
                                                                                 criterium_BP_position_update=criterium_BP_position_update,
                                                                                 Compute_CST_each_iter=compute_CST_each_iter, demands=demands,
                                                                                 verbose=self.verbose,
                                                                                 mST_fromknn=mST_fromknn,ori_input_graph=ori_input_graph,
                                                                                           beta=beta,
                                                                                           factor_terminal=factor_terminal)
            # Store CST topology and related information
            if init_CST_cost<CST_cost*scale:
                self.trees[CST_txt] = Tree(init_topo_CST.tocsr(), coords=None, widths=None, coords_vis=self.X_vis,
                                           cost=init_CST_cost)
            else:
                self.trees[CST_txt] = Tree(CST_topo*scale,coords=None,widths=None,coords_vis=self.X_vis,cost=CST_cost*scale)
            self.logger.info("Added  %s to trees\n" % CST_txt)



        else:
            self.logger.info("Computing %s" % BCST_txt)
            # Compute BCST topology without CST
            BCST_topo, BCST_cost, BCST_coords_, BCST_EW = compute_BCST(init_topo,alpha,
                                                                      coords_terminals=self.X/scale,
                                                                      maxiter_mSTreg=maxiter_mSTreg,
                                                                      maxfreq_mSTreg=maxfreq_mSTreg,
                                                                      return_CST=False,demands=demands,
                                                                      verbose=self.verbose,
                                                                       mST_fromknn=mST_fromknn,ori_input_graph=ori_input_graph,
                                                                       beta=beta,
                                                                       factor_terminal=factor_terminal,square_norm=square_norm)

        # Store BCST topology and related information
        self.trees[BCST_txt]=Tree_SP(adj=BCST_topo.adj,adj_flows=BCST_EW,coords=BCST_coords_*scale,
                                     cost=BCST_cost*scale,coords_vis=self.X_vis,alpha=alpha,demands=demands)
        if filter_BP_from_solution:
            self.trees[BCST_txt].filter_BP(threshold_filter_collapse=threshold_filter_collapse*scale)

        self.logger.info("Added  %s to trees\n" % BCST_txt)
    
    
    def minimum_spanning_tree(self,from_knn=True,num_neighs=None):
        """
           Compute the minimum spanning tree (mST) of the data points.

           Parameters:
               from_knn (bool): If True, construct the minimum spanning tree from the k-nearest neighbors graph.
                                If False, construct it directly from pairwise distances between data points.
                                Default is True.

           Returns:
               None

           Notes:
               - If the mST already exists in `self.trees`, the function does nothing.
               - Otherwise, if `from_knn` is True, the k-nearest neighbors graph is constructed (if not already available)
                 and used to compute the mST.
               - If `from_knn` is False, the mST is directly computed from pairwise distances between data points.
               - The resulting mST is stored in `self.trees` and the centrality weights are computed and stored in `self.widths_trees`.
       """

        tree_txt = 'mST'

        if tree_txt in self.trees.keys():
            # If the mST already exists, do nothing
            return
        
        if num_neighs is None:
            num_neighs = int(2 * np.log(len(self.X)))
        else:
            if num_neighs>=len(self.X)-1:
                from_knn=False
        if from_knn:
            # Construct the k-nearest neighbors graph or use pairwise distances
            try:
                G = self._knn_graph
            except:
                
                self._knn_graph=ensure_connected_knn_graph(self.X,num_neighs=num_neighs)
                G = self._knn_graph
        else:
            G=squareform(pdist(self.X))

        self.logger.info("Computing %s \n" % (tree_txt))

        # Compute the minimum spanning tree
        mST= spp.csgraph.minimum_spanning_tree(G)

        # Make the tree undirected by adding its transpose
        mST+=mST.T

        # store mST in trees
        self.trees[tree_txt] = Tree(T=mST,coords=None,widths=None,coords_vis=self.X_vis,cost=mST.sum()/2)


        self.logger.info("Added  %s to trees\n" % tree_txt)





    def plot_graph(self,on_structure='mST',node_colors=None,
                   cmap_nodes='bwr',node_size=5,suptitle='',file_path=None,show=True,
                   reverse_k1k2=False,custom_titles=None,fast_plot=True,
                   plot_filtered_BP=False):

        """
            Plot the graph representation of the data points.

            Parameters:
                on_structure (str or list): The structure(s) to plot the graph on.
                                           It can be a single structure or a list of structures.
                                           Default is 'mST'. They must be strings of self.trees.keys().
                node_colors (None or str or list): The color(s) to assign to the nodes.
                                                   It can be None, a single color, or a list of colors.
                                                   Default is None.
                cmap_nodes (str): The colormap for the node colors.
                                  Default is 'bwr'.
                node_size (int or list): The size(s) of the nodes.
                                         It can be an integer or a list of sizes.
                                         Default is 5.
                suptitle (str): The super title of the plot.
                                Default is ''.
                file_path (str): The file path to save the plot as an image.
                                 Default is None (no saving).
                show (bool): Whether to display the plot.
                             Default is True.
                reverse_k1k2 (bool): Whether to reverse the aspect ratio of the plot.
                                     Default is False.
                custom_titles (None or str or list): The custom title(s) for each plot.
                                                     It can be None, a single title, or a list of titles.
                                                     Default is None.

            Returns:
                None
            """
        # Handle input arguments as lists if necessary
        if isinstance(on_structure,list):
            on_structure_ls=on_structure
        else:
            on_structure_ls = [on_structure]
        if isinstance(node_colors,list):
            node_colors_ls=node_colors
        else:
            node_colors_ls = [node_colors]

        # Set node_size as a list if it's not already
        if not isinstance(node_size,list):
            node_size = [node_size]*(2*len(self.X)-2)

        # Set custom_titles as on_structure_ls if it's None
        if custom_titles is None:
            custom_titles=on_structure_ls

        # Calculate the k1 and k2 values for subplot layout
        k1,k2=factor_number(len(on_structure_ls))

        # Adjust figsize based on reverse_k1k2 and number of plots
        if reverse_k1k2:
            k2,k1=k1,k2
            figsize=(16* k1 / k2, 16) if len(on_structure_ls)!=1 else None
        else:
            figsize = (16, 16 * k1 / k2) if len(on_structure_ls) != 1 else None

        for counter_plot,on_structure in enumerate(on_structure_ls):
            if on_structure in self.trees.keys():
                # Get the adjacency matrix and widths for the specified structure
                A = self.trees[on_structure].T
                widths = self.trees[on_structure].widths*self._max_width_edge_tree
            else:
                raise NotImplementedError(
                    '%s not in trees or graph_structure. The possibilities are \n graph_strucuture: %s\n trees: %s\n'%(
                    on_structure,str(list(self.graph_structure.keys()))[1:-1] ,str(list(self.trees.keys()))[1:-1]))



            if node_colors_ls[0] is not None:
                # Assign node_colors based on node_colors_ls
                if len(node_colors_ls)==1:
                    node_colors=self._check_node_colors(node_colors_ls[0])
                else:
                    node_colors = self._check_node_colors(node_colors_ls[counter_plot])

            if 'BCST' in on_structure or 'BCSF' in on_structure:
                #if the structure is a BCST, plot the BCST and adjust plotting parameters for the BPs
                if self.X.shape[1]==2:
                    if plot_filtered_BP:
                        if not hasattr(self.trees[on_structure],"coords_filtered"):
                            self.trees[on_structure].filter_BP(threshold_filter_collapse=1e-5)
                        P=self.trees[on_structure].coords_filtered
                        widths=self.trees[on_structure].widths_filtered*self._max_width_edge_tree
                        A=self.trees[on_structure].T_bp_filtered
                    else:
                        P=self.trees[on_structure].coords
                else:
                    if plot_filtered_BP:
                        if not hasattr(self.trees[on_structure],"coords_filtered"):
                            self.trees[on_structure].filter_BP(threshold_filter_collapse=1e-5)
                        P=self.trees[on_structure].coords_vis[self.trees[on_structure].idxs_filtered]
                        widths=self.trees[on_structure].widths_filtered*self._max_width_edge_tree
                        A=self.trees[on_structure].T_bp_filtered
                    else:
                        P = self.trees[on_structure].coords_vis

                if node_colors is None:
                    node_colors_=['b']*(len(self.X))+ ['r']*(len(P)-len(self.X))
                else:
                    if isinstance(node_colors,list):
                        node_colors_=node_colors+[1+max(node_colors)]*(len(P)//2-1)
                    else:
                        node_colors_ = node_colors.tolist() + [1+np.max(node_colors)] * (
                                    len(P) // 2 - 1)

                plot_graph(G=A, p=P, k1=k1, k2=k2, counter_plot=counter_plot,
                           title=custom_titles[counter_plot], node_colors=node_colors_, cmap_nodes=cmap_nodes,
                           node_size=node_size[:A.shape[0]], widths=widths, figsize=figsize, axis=True,
                           fast_plot=fast_plot)
            else:
                #plot tree
                if self.X.shape[1]==2:
                    P=self.X
                else:
                    P = self.X_vis
                plot_graph(G=A,p=P,k1=k1,k2=k2,counter_plot=counter_plot,
                           title=custom_titles[counter_plot],node_colors=node_colors,cmap_nodes=cmap_nodes,
                           node_size=node_size[:A.shape[0]],widths=widths,figsize=figsize,axis=True,
                           fast_plot=fast_plot)

            idx_node_colors_ls=min(counter_plot,len(node_colors_ls)-1)
            if node_colors_ls[idx_node_colors_ls] is not None and node_colors_ls[idx_node_colors_ls] != "labels":
                sm = plt.cm.ScalarMappable(cmap=cmap_nodes,
                                           norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
                sm._A = []
                plt.colorbar(sm, fraction=0.03, pad=0.04)
        plt.suptitle(suptitle,fontsize=20)

        if file_path is not None:
            os.makedirs('/'.join(file_path.split('/')[:-1])+'/',True)
            plt.savefig(file_path,dpi=300)
        if show:
            plt.tight_layout()
            plt.show()

    def _check_node_colors(self,node_colors):
        if node_colors is None:
            pass
        elif node_colors=='labels':
            node_colors=self.labels
        else:
            raise NotImplementedError('node_colors can be None or labels')
        return node_colors

def save_object(obj, filename):
    with open(filename+'.pkl', 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    filename=filename+'.pkl' if '.pkl' not in filename else filename
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj




