import numpy as np
import os

current_file_directory=os.path.dirname(os.path.abspath(__file__))
def load_ORdata_instance(num_terminals=100,instance=0):
    '''
    Load synthetic data from the OR library dataset
    http://people.brunel.ac.uk/~mastjjb/jeb/orlib/esteininfo.html
    :param num_terminals: numer of terminals
    :param instance: instance number of the data
    :return: P: point cloud of terminals    (np.array)
    '''
    data_folder=os.path.join(current_file_directory,'../../../','Data','OR_library_data')
    filename='estein%i.%i.npy' % (num_terminals,instance)
    #assert file exists
    assert os.path.exists(os.path.join(data_folder,filename)), 'file %s does not exist in %s'%(filename,data_folder)
    return np.load(os.path.join(data_folder,filename))