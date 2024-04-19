# Central Spanning Tree

This repository includes the code of the experiments conducted in the paper "The Central Spanning Tree" by Enrique 
Fita Sanmartin, Christoph Schnörr and Fred A. Hamprecht. The paper is available [here](https://arxiv.org/pdf/2404.06447.pdf). 

![Figure 1](./Figures/figures_paper/CST_table.png)

[11]: Gilbert et. al.: Steiner Minimal Trees (1968)

[18]: Kruskal: On the Shortest Spanning Subtree of a Graph and the Traveling Salesman Problem (1956)

[21]: Masone et. al.: The minimum routing cost tree problem: State of the art and a core-node based heuristic algorithm (2019)


## INSTALLATION
The code should work for Python>=3.8. To install the required packages, run the following command:
#### Install the required packages
```bash
pip install -r requirements.txt
```

## Usage Example
```python
import numpy as np
try:
    # if installed
    from CST.T_datacls.T_datacls import T_data
except ModuleNotFoundError:
    #if Not installed
    from lib.CST.T_datacls.T_datacls import T_data

# Generate random data
n = 100
np.random.seed(0)
P = np.random.rand(n, 2)

# Create the T_data object
tdata = T_data(P,verbose=False)

# Compute the CST and BCST
alpha=0.5 # alpha parameter
maxiter_mSTreg = 10 # maximum number of iterations for the mSTreg algorithm
return_topo_CST = True # If True, the CST topology is also computed and stored in the trees dictionary of tdata
tdata.compute_BCST(alpha=alpha, maxiter_mSTreg=maxiter_mSTreg, return_topo_CST=return_topo_CST)

# Keys of the dictionary with the trees
print(tdata.trees.keys())

```
Àfter running `tdata.compute_BCST`, the object `tdata` stores in an internal dictionary `tdata.trees` the CST and 
BCST trees. The keys of the dictionary are the names of the trees, which are composed by the tree type (CST or BCST) and 
the alpha parameter used to compute the tree, e.g. 'CST_0.50' & 'BCST_0.50'.
The minimum spanning tree is accessed by the key 'mST'.


The values of the dictionary are Tree_cls objects, which store the tree topology and geometry.

#### CST trees
```python
print(tdata.trees['CST_%0.2f' % alpha])
'''
Output:
Tree:
    T: 100x100
    widths
    cost: 1.518992
'''
```
CST trees contain the following attributes:

- `T`: Sparse adjacency matrix of the tree
- `widths`: List containing the centralities of each edge. They are ordered in the same way as the edges in the 
  adjacency matrix, row-wise.
- `cost`: Total cost of the tree.
#### BCST trees
```python
print(tdata.trees['BCST_%0.2f' % alpha])
'''
Output:
Tree_SP:
    T: 198x198
    num_terminals: 100
    adj: 98x3
    adj_flows: 98x3
    coords: 198x2
    widths
    cost: 1.457789
    T_bp_filtered: 163x163
    coords_filtered: 163x2
    idxs_filtered: len=163
    widths_filtered
'''
```
BCST trees contain the following attributes:

-`Tree_SP`: Sparse adjacency matrix of the tree. Terminals correspond to the first indices while Steiner points are 
the last ones.
- `num_terminals`: Number of terminal nodes in the tree.
- `adj`: Different representation of the tree topology. It is a 2D array where each row corresponds to a Steiner 
  point, and the columns are indices of the adjacent nodes (3 per Steiner point).
- `coords`: Coordinates of the nodes in the tree. Terminals correspond to the first indices while Steiner points are 
the last ones.
- `widths`: List containing the centralities of each edge, ordered in the same way as the edges in the adjacency matrix.
- `cost`: Total cost of the tree.
- `T_bp_filtered`: Sparse adjacency matrix of the tree after removing collapsed Steiner points, i.e. Steiner points 
  which shared the same position once they were optimized.
- `coords_filtered`: Coordinates of the nodes in the filtered tree.
- `idxs_filtered`: List containing the indices of the coordinates included in the filtered tree.
- `widths_filtered`: Centralities of the edges in the filtered tree.



## Definition
The Central Spanning Tree (CST) is a family of robust spanning trees embedded in Euclidean space, whose geometrical 
structure is resilient against perturbations such as noise on the coordinates of the nodes. Two variants of the 
problem are explored: one permitting the inclusion of Steiner points (referred to as branched central spanning tree 
or BCST), and another that does not. The family of trees is defined through a parameterized NP-hard minimization 
problem over the edge lengths, with specific instances including the minimum spanning tree or the Euclidean Steiner 
tree. The minimization problem weighs the length of the edges by their tree edge-centralities, which are regulated 
by a parameter $\alpha$. Formally, the CST is defined as the solution to the following optimization problem:

$$\underset{T}{\arg \min}\sum_{(i,j)\in E_{T}}\big(m_{ij}(1-m_{ij})\big)^{\alpha}||x_i-x_j||$$

where $m_{ij}$ and $(1-m_{ij})$ are the normalized cardinalities of the components resulting from the removal of the edge 
$e=(i,j)$ from $T$. the product $m_e(1-m_e)$ is proportional to the "edge betweeness centrality" of $e$ in $T$.


## Evolution (Branched) Central Spanning Tree with respect to α

| CST                                                                   | BCST                                                                   |
|-----------------------------------------------------------------------|------------------------------------------------------------------------|
| ![Figure 1](./Figures/GIFS/alpha_evolution/uniform/CST/animation.gif) | ![Figure 2](./Figures/GIFS/alpha_evolution/uniform/BCST/animation.gif) |


## Stability
As $\alpha$ increases the CST becomes more stable against perturbations in the coordinates of the nodes at the 
expense of having larger edges. The parameter $\alpha$ can be interpreted as a trade-off between the stability of 
the tree and the data fidelity, where the latter is intuitively related to connecting the nodes with short edges. 

![Figure 1](./Figures/figures_paper/Stability_figure.png) 

|                               CST                               |                               BCST                                |
|:---------------------------------------------------------------:|:-----------------------------------------------------------------:|
|                             α=0.00                              |                              α=0.00                               |
| ![Figure 1](./Figures/GIFS/BOUNCING/uniform/CST_alpha_0.00.gif) | ![Figure 2](./Figures/GIFS/BOUNCING/uniform/BCST_alpha_0.00.gif)  |
|                             α=0.25                              |                              α=0.25                               |
| ![Figure 3](./Figures/GIFS/BOUNCING/uniform/CST_alpha_0.25.gif) | ![Figure 4](./Figures/GIFS/BOUNCING/uniform/BCST_alpha_0.25.gif)  |
|                             α=0.50                              |                              α=0.50                               |
| ![Figure 5](./Figures/GIFS/BOUNCING/uniform/CST_alpha_0.50.gif) | ![Figure 6](./Figures/GIFS/BOUNCING/uniform/BCST_alpha_0.50.gif)  |
|                             α=0.75                              |                              α=0.75                               |
| ![Figure 7](./Figures/GIFS/BOUNCING/uniform/CST_alpha_0.75.gif) | ![Figure 8](./Figures/GIFS/BOUNCING/uniform/BCST_alpha_0.75.gif)  |
|                             α=1.00                              |                              α=1.00                               |
| ![Figure 9](./Figures/GIFS/BOUNCING/uniform/CST_alpha_1.00.gif) | ![Figure 10](./Figures/GIFS/BOUNCING/uniform/BCST_alpha_1.00.gif) |

## mSTreg heuristic algorithm

The central spanning tree is a NP-hard problem. It is known that BCST topology can be represented by a so called 
full topology, which is a tree that contains all input nodes (terminals) as leaves and all Steiner points have 
degree 3. 
To approximate the solution, we propose a heuristic algorithm called mSTreg, which alternates between two steps:
- **Geometry update**: It updates the coordinates of the Steiner points. Given a topology, the optimal position of the 
  coordinates can be computed efficiently. Code for the geometry update is available has been based on the 
  optimization of the Steiner points for the Branched Optimal Transport problem, available [here](https://github.com/sciai-lab/BranchedOT).
- **Topology update**: It updates the topology of the tree. Given the coordinates of the nodes and the optimal Steiner 
  point coordinates of the previous step, it updates the topology of the tree, by computing the minimum spanning 
  tree (mST) over all nodes (Steiner and terminal nodes). The mST is then transformed into a full topology since we 
  know that the optimal topology is a full topology.

![Figure 1](./Figures/figures_paper/mSTreg_summary_figure_triangle.png)


### Iterations of the mSTreg heuristic algorithm
| α = 0.25 | α = 0.50 |
|--------------|--------------|
| ![Figure 1](./Figures/GIFS/mSTREG_iterations/uniform/alpha=0.25/animation.gif) | ![Figure 2](./Figures/GIFS/mSTREG_iterations/uniform/alpha=0.50/animation.gif) |

| α = 0.75 | α = 1.00 |
|--------------|--------------|
| ![Figure 3](./Figures/GIFS/mSTREG_iterations/uniform/alpha=0.75/animation.gif) | ![Figure 4](./Figures/GIFS/mSTREG_iterations/uniform/alpha=1.00/animation.gif) |






## 3D Plant skeletonization 
### Tomato plant 2, Day 5
| BCST  α = 0.00                                                                                     | BCST  α = 0.50                                                                                           | BCST  α = 0.70                                                                                           | BCST  α = 1.00                                                                                           |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| ![Figure 1](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day5_n=5000_BCST_0.00_prior.gif) | ![Figure 2](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day5_n=5000_BCST_0.50_prior.gif) | ![Figure 3](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day5_n=5000_BCST_0.70_prior.gif) | ![Figure 4](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day5_n=5000_BCST_1.00_prior.gif) |       


### Tomato plant 2, Day 8
| BCST  α = 0.00                                                                                     | BCST  α = 0.50                                                                                           | BCST  α = 0.70                                                                                           | BCST  α = 1.00                                                                                           |
|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| ![Figure 1](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day8_n=5000_BCST_0.00_prior.gif) | ![Figure 2](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day8_n=5000_BCST_0.50_prior.gif) | ![Figure 3](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day8_n=5000_BCST_0.70_prior.gif) | ![Figure 4](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day8_n=5000_BCST_1.00_prior.gif) |       


### Tomato plant 2, Day 13
| BCST  α = 0.00                                                                                                 | BCST  α = 0.50                                                                                            | BCST  α = 0.70                                                                                                 | BCST  α = 1.00                                                                                            |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| ![Figure 1](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day13_n=5000_BCST_0.00_prior.gif) | ![Figure 2](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day13_n=5000_BCST_0.50_prior.gif) | ![Figure 3](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day13_n=5000_BCST_0.70_prior.gif) | ![Figure 4](./Figures/GIFS/plant_skeleton/tomato_plant2/n=5000/tomato_plant2_day13_n=5000_BCST_1.00_prior.gif) |       
