
import os
import warnings
import numpy as np
import scipy.sparse as sp
import struct
import requests
import zipfile
#%%
file_path = os.path.abspath(__file__)
# Extract the directory path
directory_current_file = os.path.dirname(file_path)

#%%
#####################################################################################################
#####################################################################################################
#####################################################################################################

# TREE POINT CLOUD https://github.com/liujiboy/TreePointCloud

#####################################################################################################
#####################################################################################################
#####################################################################################################
def load_treepointcloud_data(data_modality='clean', resolution=100, tree_instance=0, tree_size='medium',
                             extra_text='',data_folder=None):
    '''
    Load synthetic tree from the TreePointCloud dataset
    https://github.com/liujiboy/TreePointCloud

    :param data_modality:
    :param resolution:
    :param tree_instance:
    :param tree_size:
    :return:
    '''
    
    if data_folder is None:
        data_folder=os.path.join(directory_current_file,'../..','Data','Meaningful_Spanning_Tree','TreePointCloud-master')
    
        if data_modality == 'clean':
            data_folder += '/scan_ply'
        
        elif data_modality == 'noise':
            data_folder += '/scan_ply_noise'
        elif data_modality == 'missing':
            data_folder += '/scan_ply_missing'
        elif data_modality == 'uneven':
            data_folder += '/scan_ply_uneven'
        else:
            raise ValueError('data_modality should be clean, noise, missing or uneven')
    
    
    if data_modality == 'missing':
        assert extra_text!='', "for %s extra_text should be '_2_50' or '_3_90'"%data_modality
        
    elif data_modality == 'noise':
        assert extra_text!='',"for %s extra_text should be '_30_20', '_30_30' or '_30_50'"%data_modality
        assert resolution==100,"for %s resolution should be 100"%data_modality
    
    if resolution != 100 and (data_modality == 'noise' or data_modality == 'uneven' or data_modality == 'missing'):
        warnings.warn("for %s resolution should be 100, setting resolution to 100" % data_modality)
        resolution = 100
    
    if extra_text!='' and (data_modality=='clean' or data_modality=='uneven'):
        warnings.warn("for %s 'extra_text' should be '' (empty string), setting 'extra_text' to empty string ''"%data_modality)
        extra_text=''
    
    if data_modality == 'clean':
        filename = 'tree_%s_%i_res_%i.ply' % (tree_size, tree_instance, resolution)
    else:
        filename = 'tree_%s_%i_res_%i_%s%s.ply' % (tree_size, tree_instance, resolution, data_modality, extra_text)

    file_path = os.path.join(data_folder, filename)
    return load_ply_vertices(file_path)

def load_treepointcloud_GTskeleton(tree_instance=0, tree_size='medium',compute_distance=False,data_folder=None):
    '''
    Returns the ground truth skeleton of a tree from the TreePointCloud dataset
    https://github.com/liujiboy/TreePointCloud
    
    It returns the vertices and the adjacency matrix of the skeleton.
    
    :param tree_instance:
    :param tree_size:
    :param data_folder:
    :param compute_distance: If compute_distance is True, the adjacency matrix
    stores the Euclidean distances between connected vertices. Otherwise, it stores a value of 1 for each connection.
    :return: np.array with vertices, sp.sparse matrix with adjacency matrix
    '''
    filename_gt = 'tree_%s_%i.obj' % (tree_size, tree_instance)
    if data_folder is None:
        data_folder = os.path.join(directory_current_file,'../..','Data','Meaningful_Spanning_Tree','TreePointCloud-master','scan_skel')
    
    vertices_array, adjacency = _load_GTskeleton(os.path.join(data_folder, filename_gt),
                                                 compute_distance=compute_distance)
    return  vertices_array, adjacency

def read_ply_header(file_path):
    """Read the header of a PLY file."""
    header = []
    with open(file_path, 'rb') as f:
        for line in f:
            line = line.strip().decode('ascii')
            header.append(line)
            if line == "end_header":
                break
    return header

def read_ply_vertices(file_path, header):
    """Read the vertices of a binary PLY file."""
    num_vertices = 0
    data_start = 0
    for line in header:
        if line.startswith("element vertex"):
            num_vertices = int(line.split()[2])
        if line == "end_header":
            data_start = header.index(line) + 1
            break
    
    vertices = []
    with open(file_path, 'rb') as f:
        for _ in range(data_start):
            f.readline()
        for _ in range(num_vertices):
            vertex_data = f.read(12)
            vertex = struct.unpack('<fff', vertex_data)
            vertices.append(vertex)
    return vertices

def load_ply_vertices(file_path):
    """Load the vertices of a binary PLY file into a numpy array."""
    header = read_ply_header(file_path)
    vertices = read_ply_vertices(file_path, header)
    vertices = np.array(vertices)
    vertices[:, [1, 2]] = vertices[:, [2, 1]]
    return vertices

def _load_GTskeleton(file_path, compute_distance=False):
    """Load the vertices of an OBJ file representing a GT skeleton into a numpy array
    and return the adjacency matrix in sparse format.

    If compute_distance is True, the adjacency matrix stores the Euclidean distances
    between connected vertices. Otherwise, it stores a value of 1 for each connection.
    """
    
    vertices = []
    edges = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex positions
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('l '):  # Lines indicating edges
                _, v1, v2 = line.split()
                edges.append((int(v1) - 1, int(v2) - 1))  # OBJ uses 1-indexed vertices, converting to 0-indexed
    
    vertices_array = np.array(vertices)
    vertices_array[:, [1, 2]] = vertices_array[:, [2, 1]]  # Swap Y and Z coordinates
    n_vertices = len(vertices)
    
    # Create the adjacency matrix
    adjacency = sp.lil_matrix((n_vertices, n_vertices))
    for v1, v2 in edges:
        if compute_distance:
            dist = np.linalg.norm(vertices_array[v1] - vertices_array[v2])
            adjacency[v1, v2] = dist
            adjacency[v2, v1] = dist  # Ensure symmetry
        else:
            adjacency[v1, v2] = 1
            adjacency[v2, v1] = 1  # Ensure symmetry
    
    return vertices_array, adjacency

#%%
#####################################################################################################
#####################################################################################################
#####################################################################################################

# CHERRY TREE

#####################################################################################################
#####################################################################################################
#####################################################################################################
def load_cherrytree():
    filename = os.path.join(directory_current_file,'..','Data/cherry_tree.npy')
    
    # Load the saved point cloud from the file
    return np.load(filename)

#%%
#####################################################################################################
#####################################################################################################
#####################################################################################################

# TOMATO PLANT

#####################################################################################################
#####################################################################################################
#####################################################################################################
def load_plantdata(plant_type, plant_num, day):
    data_folder = os.path.join(directory_current_file, '..', 'Data')
    os.makedirs(data_folder, exist_ok=True)
    plant_folder = '4d_plant_registration_data/%s/plant%i/' % (plant_type, plant_num)
    folder_plant = os.path.join(data_folder, plant_folder)
    filename = '03-%s.txt' % str(day).zfill(2)

    # Check if data folder exists
    if not os.path.exists(os.path.join(data_folder, '4d_plant_registration_data/')):
        # Print a warning instead of raising an error
        print('Data folder does not exist. Downloading data...')

        url = "https://www.ipb.uni-bonn.de/html/projects/4d_plant_registration/4d_plant_registration_data.zip"

        # Download the file
        print('Downloading 4d_plant_registration_data.zip')
        zip_filename = os.path.join(data_folder, "4d_plant_registration_data.zip")
        response = requests.get(url)
        with open(zip_filename, "wb") as f:
            f.write(response.content)

        # Extract the ZIP file
        print('Extracting 4d_plant_registration_data.zip')
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

        # Remove the ZIP file
        os.remove(zip_filename)

    assert os.path.isfile(os.path.join(folder_plant, filename)), "File %s does not exist in %s." % (
    filename, plant_folder)
    return np.loadtxt(os.path.join(folder_plant, filename))
