import numbers
try:
    import networkit as nk
    nk_imported=True
except:
    nk_imported=False

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from ...methods.mSTreg.heuristics.mSTreg import compute_BCST


def vis_data_coords4fulltopo(adj, vis_coords,alpha=None):
    '''
    Compute positions of Steiner points given the adjacency matrix and the positions of the terminals

    :param adj: np.array where row i indicates the neighbors of steiner point indexed by num_terminals + i. Each Steiner
    point has 3 neighbors. There are num_terminals + 2  Steiner points.
    :param vis_coords: coordinates of the terminals
    :param alpha:
    :return:
    '''

    if alpha is None:
        return vis_data_coords4fulltopo_approx(adj, vis_coords)
    else:
        return vis_data_coords4fulltopo_exact(adj, vis_coords,alpha)


@jit(nopython=True)
def vis_data_coords4fulltopo_approx(adj, vis_coords):
    num_bp = len(vis_coords) - 2
    visfulltopo_coords = np.zeros((len(vis_coords) + num_bp, 2))

    # Assign values from vis_coords to visfulltopo_coords
    for i in range(len(vis_coords)):
        visfulltopo_coords[i] = vis_coords[i]

    # Sort BPs based on their neighbor with lowest value. BPs adjacent to terminals will be first.
    n = len(vis_coords)
    min_vals = np.empty(adj.shape[0], dtype=adj.dtype)  # Create an empty array for minimum values
    for i in range(adj.shape[0]):
        min_vals[i] = np.min(adj[i, :])  # assign minimum value of each row to min_vals

    sorted_idxs = np.argsort(min_vals)
    for j in range(10):
        for i, a in enumerate(adj[sorted_idxs]):
            na = sorted_idxs[i] + n
            if j == 0:
                total = max(np.count_nonzero(np.abs(visfulltopo_coords[a]).sum(1)), 1)
            else:
                total = 3

            visfulltopo_coords[na] = np.sum(visfulltopo_coords[a], axis=0) / total

    return visfulltopo_coords


def vis_data_coords4fulltopo_exact(adj, vis_coords,alpha):
    _, _, visfulltopo_coords, _=compute_BCST(adj, alpha,coords_terminals=vis_coords,
                                             maxiter_mSTreg=-1,verbose=False)

    return visfulltopo_coords



def compute_figsize(x_lim,y_lim,k1=1,k2=1):
    if k1==k2 and k1==1:
        x_diff=x_lim[1]-x_lim[0]
        y_diff=y_lim[1]-y_lim[0]
        scale=x_diff/y_diff
        if scale<1:
            figsize=(np.round(16*scale).astype(np.int8),16)
        else:
            figsize=(16,np.round(16/scale).astype(np.int8))
    else:
        if k2>k1:
            figsize = (16, k1/k2* 16)
        else:
            figsize = (16*k2 / k1,   16)
    return figsize


def compute_xy_lim(P,margin_perc=0.05):
    if isinstance(P,dict):
        P=np.array(list(P.values()))
    else:
        assert(isinstance(P,np.ndarray))

    X_offset=(np.max(P[:,0])-np.min(P[:,0]))*margin_perc
    Y_offset=(np.max(P[:,1])-np.min(P[:,1]))*margin_perc
    x_lim=(np.min(P[:,0])-X_offset,np.max(P[:,0])+X_offset)
    y_lim=(np.min(P[:,1])-Y_offset,np.max(P[:,1])+Y_offset)

    return x_lim,y_lim


def plot_graph(G,p,k1=1,k2=1,counter_plot=0,x_lim=None,y_lim=None,widths=None,title='',
               edge_colors=None,cmap_edges=None,node_colors=None,cmap_nodes=None,node_size=1,return_ax=False,
               new=True,label_nodes=None,label_edges=False,axis=True,vmin_edge=None,
               vmax_edge=None,figsize=None,node_shape="o",edgelist=None,return_fig=False,
               nodelist=None,alpha_node=None,alpha_edge=None,fontsize=None,ax=None,
               fast_plot=True):
    '''


    Parameters
    ----------
    G : nx or nk graph
        DESCRIPTION.
    p : dict
        coordinates of the points. key =node ID, value=coordinate
    k1 : int, optional
        number of rows for the subplot. The default is 1.
    k2 : int, optional
        number of columns for the subplot. The default is 1.
    counter_plot : TYPE, optional
        DESCRIPTION. The default is 0.
    x_lim : tuple
        range of the values of the points in the x coordinate.
    y_lim : tuple
        range of the values of the points in the y coordinate.
    widths : list, optional
        widths of the edges. It must have the same order as the edges in the graph G.
        The default is None.
    title : str, optional
        Title of the plot. The default is ''.
    edge_colors : list, optional
        color values of the edges. It must have the same order as the edges in the graph G.
        The default is None.
    cmap_edges : plt.colormap, optional
        colormap of the edges. The default is None.
    node_colors : TYPE, optional
        color values of the nodes. It must have the same order as the nodes in the graph G.
    cmap_nodes : plt.colormap, optional
        colormap of the nodes. The default is None.
    node_size : int, optional
        size of the nodes. The default is 1.
    return_ax : bool, optional
        if True returns axis of the figure. The default is False.
    new : bool, optional
        if True, a new figure is created. The default is True.
    label_nodes : bool, optional
        if True, ID of nodes is shown. The default is False.
    label_edges :bool, optional
        if True, weight of the edges is shown. The default is False.
    axis : bool, optional
        If True, the axis coordinate is shown. The default is True.
    vmin_edge : float, optional
        minimum value of the colormap of the edges. The default is None.
    vmax_edge : float, optional
        maximum value of the colormap of the edges. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is None.
    edgelist : list, optional
        list with edges to be plotted. The default is None.
    return_fig : bool, optional
        returns figure if True. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if x_lim is None or y_lim is None:
        x_lim,y_lim=compute_xy_lim(p)
    if figsize is None:
        figsize=compute_figsize(x_lim,y_lim,k1=k1,k2=k2)

    if 'scipy.sparse' in str(type(G)):
        try:
            G=nx.from_scipy_sparse_matrix(G,create_using=nx.Graph())
        except AttributeError:
            G=nx.from_scipy_sparse_array(G,create_using=nx.Graph())


    if isinstance(p,np.ndarray):
        p=dict(enumerate(p))

    try:
        G=nk.nxadapter.nk2nx(G)
    except:
        pass

    if edgelist is None:
        edgelist=G.edges()
    if nodelist is None:
        nodelist=G.nodes()

    if counter_plot==0 and new:
        if return_ax:
            fig=plt.figure(figsize=figsize,dpi=100)
        else:
            plt.figure(figsize=figsize,dpi=100)
    if ax is None:
        if k1!=1 or k2!=1:
            ax=plt.subplot(int(k1),int(k2),counter_plot+1,axisbelow=True)
        else:
            ax=plt.axes()
    plt.title(title,fontsize=fontsize)

    # max_width=max(widths) if widths is not None else 1

    if G.number_of_nodes()<=30 and label_nodes != False:

        #node_size=1

        nx.draw_networkx_labels(G,p)


        if isinstance(node_shape,list):
            for s  in np.unique(node_shape):
                node_list=np.where(node_shape==s)[0]
                nx.draw_networkx_nodes(G,p,nodelist=node_list, node_size=node_size,ax=ax,node_color=node_colors,cmap=cmap_nodes
                                   ,node_shape=s)
        else:
            nx.draw_networkx_nodes(G,p, node_size=node_size,ax=ax,node_color=node_colors,cmap=cmap_nodes
                               ,node_shape=node_shape,nodelist=nodelist)

        nx.draw_networkx_edges(G, pos=p, width=widths, edge_color=edge_colors, edge_cmap=cmap_edges,
                               edge_vmin=vmin_edge, edge_vmax=vmax_edge, edgelist=edgelist,
                               connectionstyle='arc3,rad=1000000',alpha=alpha_edge)

        if label_edges:
            E_labels = dict([((u,v,), f"{d['weight']:.3f}") for u,v,d in G.edges(data=True)])
            nx.draw_networkx_edge_labels(G,p,edge_labels=E_labels)
        # nx.draw_networkx_nodes(G,p,with_labels=True)
    else:

        # else:
        #     nx.draw_networkx_edges(G,pos=p,edge_color=edge_colors,edge_cmap=cmap_edges)#,width=weights_T_Delaunay_sp_centrality)

        if cmap_edges is None and not fast_plot:
            custom_draw_edges(G, pos=p, width=widths, edge_color=edge_colors, edgelist=edgelist,
                              alpha=alpha_edge)  # %0.2f'%(5*max_width))
            # custom_draw_nodes(G, pos=p, node_size=node_size,ax=ax,node_color=node_colors,cmap=cmap_nodes
            #                    ,node_shape=node_shape,alpha=alpha_node,nodelist=nodelist)
        else:
            nx.draw_networkx_edges(G, pos=p, width=widths, edge_color=edge_colors, edge_cmap=cmap_edges,
                               edge_vmin=vmin_edge, edge_vmax=vmax_edge, edgelist=edgelist,
                               connectionstyle='arc3,rad=1000000',alpha=alpha_edge)#%0.2f'%(5*max_width))

        if isinstance(node_shape,list):
            for s in np.unique(node_shape):
                node_list=np.where(np.array(node_shape)==s)[0]
                if isinstance(node_size,list) or isinstance(node_size,np.ndarray):
                    node_size_filt=np.array(node_size)[node_list]
                else:
                    node_size_filt=node_size
                if isinstance(node_colors,list) or isinstance(node_colors,np.ndarray):
                    node_colors_filt=np.array(node_colors)[node_list]
                else:
                    node_colors_filt=node_colors

                nx.draw_networkx_nodes(G,p,nodelist=node_list, node_size=node_size_filt,ax=ax,node_color=node_colors_filt,cmap=cmap_nodes
                                   ,node_shape=s,alpha=alpha_node)
        else:

            nx.draw_networkx_nodes(G,pos=p, node_size=node_size,ax=ax,node_color=node_colors,cmap=cmap_nodes
                               ,node_shape=node_shape,alpha=alpha_node,nodelist=nodelist)

        # nx.draw_networkx(G_Delaunay,pos=p, node_size=10,ax=ax,with_labels=True)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set(xlim=x_lim, ylim=y_lim)
    ax.set_aspect('equal')
    if k1==1 and k2==1:
        plt.tight_layout()
    plt.draw()
    if axis==False:
        plt.axis('off')
    #plt.pause(0.00001)

    if return_ax and return_fig and counter_plot==0 and new:
        return fig, ax
    elif return_ax:
        return ax


def custom_draw_edges(G, pos, width=None, edge_color=None,edgelist=None,alpha=0.5):
    if isinstance(width, numbers.Number) and width==0:
         return
    if edgelist is None:
        edgelist = G.edges()

    for edge_counter,e in enumerate(edgelist):
        u,v=e
        if width is None:
            linewidth=None
        elif isinstance(width, numbers.Number):
            linewidth=width
        elif hasattr(width, '__iter__'):
            linewidth=width[edge_counter]

        if edge_color is None:
            c='k'
        elif isinstance(edge_color, numbers.Number) or isinstance(edge_color,str) or (isinstance(edge_color,np.ndarray) and edge_color.ndim==1):
            c=edge_color
        elif hasattr(edge_color, '__iter__'):
            c=edge_color[edge_counter]

        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=c, linewidth=linewidth,
                           alpha=alpha, solid_capstyle='round',zorder=-1)#,cmap=edge_cmap)


def custom_draw_nodes(G, pos, node_size,ax,node_color,cmap,node_shape,alpha,nodelist=None):
    if nodelist is None:
        nodelist=np.arange(len(pos))
    P=np.array(list(pos.values()))
    ax.scatter(P[nodelist,0],P[nodelist,1],alpha=alpha,s=node_size,cmap=cmap,marker=node_shape,c=node_color)