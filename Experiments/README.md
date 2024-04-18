This folder includes the code for the experiments in the paper. 
- **Benchmark_OR folder**: It contains the code for the experiments in the paper of section 4.3 that rely on the OR 
  library instances, available [here](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/esteininfo.html). In the Data 
  folder of the repository, you can find the instances used in the experiments.
- **benchmark_smalltoydata folder**: It contains the code for the experiments in the paper of section 4 that rely on the 
  small toy data instances in section 4.3.
- **Plant_skeletonizations folder**: It contains the code for the experiments in the paper of section 5.1 that rely on 
  the plant skeletonization examples.
- **Single_cell folder**: It contains the code for the experiments in the paper of section 5.2 that rely on the single 
  cell examples.
- **Empirical_stability file**: It contains the code for the experiments in the paper of section 5.3 that rely on the 
  empirical stability analysis.
- **Time_complexity file**: It contains the code for the experiments in the paper of section 5.4 that rely on the time 
  complexity analysis.


### Extra requirements
To run some experiments you may need to install in addition to the requirements:


- pandas
- h5py

They can be installed using pip or conda
#### pip
```bash
pip install pandas h5py
```
#### Conda
```bash 

conda install pandas h5py
```