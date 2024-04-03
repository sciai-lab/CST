
import os
print('cwd',os.getcwd())

if '/Benchmark_mSTreg' in os.getcwd():
    os.chdir('../../../')
    print('cwd updated',os.getcwd())

    import sys
    sys.path.insert(os.getcwd(),0)
    print(sys.path)

import os
import numpy as np
from lib.CST.T_datacls import T_data,load_object
from lib.CST.T_datacls.utilities.graphtools import Wiener_index
import itertools
from lib.CST.methods.mSTreg.topology.topology import adj_to_adj_sparse

#%%
# Set folder to save results
folder_ratios = 'Experiments/Benchmark_mSTreg/benchmark_smalltoydata/ratios/'
folder_quantiles = 'Experiments/Benchmark_mSTregBenchmark_smalltoydata_CST/quantiles/'
os.makedirs(folder_ratios,True)
os.makedirs(folder_quantiles,True)

# Define a function to calculate the number of branched topologies based on the number of terminals
def num_branched_topos(num_terminals):
    num_topos = 1
    for i in range(2 * num_terminals):
        k = 2 * num_terminals - 5 - 2 * i
        if k > 1:
            num_topos *= k
        else:
            break
    return num_topos

#%%
# Set parameters
n=8
save=True
maxiter_mSTreg=10
maxfreq_mSTreg=5
verbose=False

# Calculate the number of branched topologies for the given number of terminals
num_topos=num_branched_topos(num_terminals=n)

# Define a range of alpha values from 0 to 1 with a step of 0.1
alpha_range=np.array(list(range(11)))/10

# Define lists of criteria for branching and merging in the algorithm
order_criterium_ls = ['closestterminals']
merging_criterium_ls = ['closest', 'tryall', ]
criterium_BP_position_update_ls = ['median', 'no_update']
total_options = len(criterium_BP_position_update_ls) * len(merging_criterium_ls) * len(order_criterium_ls)
ls_criteriums = list(itertools.product(order_criterium_ls, criterium_BP_position_update_ls, merging_criterium_ls))

# Initialize dictionaries to store ratios and quantiles for CST and BCST
ratios_CST_dict={}
quantiles_CST_dict={}
for order_criterium, criterium_BP_position_update, merging_criterium in ls_criteriums:
    txt_CST = 'NOBPcost_' + '-'.join((order_criterium, criterium_BP_position_update, merging_criterium))
    ratios_CST_dict[txt_CST]=[]
    quantiles_CST_dict[txt_CST]=[]
ratios_BCST=[]
quantiles_BCST=[]

# Define folders for loading solved tree data
folder='Experiments/Benchmark_mSTreg/benchmark_smalltoydata/Data/n=%i/'%n
files=os.listdir(folder)

# Iterate through files in the specified folder
for file in files:
    print('n=%i, file'%n,file)
    # Load data from the file
    tdata_load=load_object(folder+file)
    tdata=T_data(tdata_load.X,verbose=verbose)

    # Iterate through alpha values
    for alpha in alpha_range:
        best_cost_BCST = tdata_load.costs['BCST_%0.2f' % alpha]
        best_cost_CST = Wiener_index(tdata_load.trees['best_CST_%0.2f'%alpha], alpha) / (n ** (2 * alpha))
        first_iter = True

        # Iterate through different criteria for branching and merging
        for order_criterium, criterium_BP_position_update, merging_criterium in ls_criteriums:
            txt_CST = 'NOBPcost_' + '-'.join((order_criterium, criterium_BP_position_update, merging_criterium))

            # Compute BCST
            tdata.compute_BCST(alpha=alpha,maxiter_mSTreg=maxiter_mSTreg,
                               maxfreq_mSTreg=maxfreq_mSTreg,order_criterium = order_criterium, merging_criterium = merging_criterium,
                         criterium_BP_position_update = criterium_BP_position_update,compute_CST_each_iter=True)
            if first_iter:# BCST is computed every time for the CST, but we only need the results once
                # Get BCST outputs
                adj = tdata.BCST_outputs['BCST_%0.2f' % alpha]['adj']
                flows = tdata.BCST_outputs['BCST_%0.2f' % alpha]['flows']
                TB_FLOWS = adj_to_adj_sparse(adj, flows=flows)
                coords = tdata.BCST_outputs['BCST_%0.2f' % alpha]['P']

                BCST_cost=tdata.costs['BCST_%0.2f' % alpha]
                cost_ratio_BCST=BCST_cost/best_cost_BCST
                first_iter=False

            # Get CST outputs
            T_CST = tdata.trees['CST_%0.2f'%alpha]
            CST_cost= tdata.costs['CST_%0.2f'%alpha]

            cost_ratio_CST = CST_cost / best_cost_CST
            ratios_CST_dict[txt_CST].append([alpha, cost_ratio_CST])
            for i, cost in enumerate(sorted(tdata_load.CST_costs_dict[alpha])):
                if CST_cost <= cost:
                    quantile_CST = 100 * i / max(len(tdata_load.CST_costs_dict[alpha]), n ** (n - 2))
                    break
            quantiles_CST_dict[txt_CST].append([alpha, quantile_CST])

        ratios_BCST.append([alpha,cost_ratio_BCST])

        for i,cost in enumerate(sorted(tdata_load.BCST_costs_dict[alpha])):
            if BCST_cost<=cost:
                quantile_BCST=100*i/max(len(tdata_load.BCST_costs_dict[alpha]),num_topos)
                break

        quantiles_BCST.append([alpha,quantile_BCST])

# Convert lists to numpy arrays
ratios_BCST=np.array(ratios_BCST)
quantiles_BCST=np.array(quantiles_BCST)
for order_criterium, criterium_BP_position_update, merging_criterium in ls_criteriums:
    txt_CST = 'NOBPcost_' + '-'.join((order_criterium, criterium_BP_position_update, merging_criterium))
    ratios_CST_dict[txt_CST] = np.array(ratios_CST_dict[txt_CST])
    quantiles_CST_dict[txt_CST] = np.array(quantiles_CST_dict[txt_CST])

# Save data if the 'save' flag is set to True
if save:
    os.makedirs(folder_ratios,True)
    np.save(folder_ratios+'ratios_BCST_n=%i'%n,ratios_BCST)

    os.makedirs(folder_quantiles,True)
    np.save(folder_quantiles+'quantiles_BCST_n=%i'%n,quantiles_BCST)


    for order_criterium, criterium_BP_position_update, merging_criterium in ls_criteriums:
        txt_CST = 'NOBPcost_' + '-'.join((order_criterium, criterium_BP_position_update, merging_criterium))
        np.save(folder_ratios + 'ratios_CST_%s_n=%i' % (txt_CST, n), ratios_CST_dict[txt_CST])

        np.save(folder_quantiles +'quantiles_CST_%s_n=%i' % (txt_CST, n), quantiles_CST_dict[txt_CST])

