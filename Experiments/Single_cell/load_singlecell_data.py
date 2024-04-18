import numpy as np
import requests

import os
import h5py

file_path = os.path.abspath(__file__)
# Extract the directory path
directory_current_file = os.path.dirname(file_path)
data_folder = os.path.join(directory_current_file, '../..', 'Data','single_cell_data')
os.makedirs(data_folder, exist_ok=True)

try :
    #use scanpy to download and preprocess the data
    import scanpy as sc
    
    sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
    sc.logging.print_versions()
    sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')
    sc.settings.datasetdir=data_folder
    
except ImportError:
    print("scanpy not installed. If data is already downloaded and preprocessed, you can ignore this message")
    pass




def load_paul15():
    filename_preprocessed = 'paul15_preprocessed.h5'
    filepath_preprocessed=os.path.join(data_folder,filename_preprocessed)
    #check if file exists
    if not os.path.exists(filepath_preprocessed):
        adata = sc.datasets.paul15()
        print('preprocessing data')
        preprocess_paul15(adata)


        print('Saving preprocessed data')
        X_PCA = adata.obsm['X_pca']
        X_PAGA = adata.obsm['X_draw_graph_fa']
        labels = np.array(adata.obs['paul15_clusters'])
        with h5py.File(filepath_preprocessed, "w") as f:
            f.create_dataset('X_PCA50', data=X_PCA)
            f.create_dataset('X_PAGA', data=X_PAGA)
            f['labels'] = labels.astype('S')
        print('Saved %s in %s' % (filename_preprocessed, data_folder))
    with h5py.File(filepath_preprocessed, 'r') as f:
        print(f.keys())
        X_PCA = np.array(f['X_PCA50'])
        X_PAGA = np.array(f['X_PAGA'])
        labels = np.array(f['labels']).astype(str)
    return X_PCA,X_PAGA,labels



def preprocess_paul15(adata):
    adata.X = adata.X.astype('float64')

    print('Applying zheng recipe')
    sc.pp.recipe_zheng17(adata)

    print ('Computing PAGA embedding')
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
    sc.tl.draw_graph(adata)

    # Denoising graph
    sc.tl.diffmap(adata)
    np.random.seed(12)
    # adata.obsm['X_diffmap']+=np.random.normal(0,0.000001,adata.obsm['X_diffmap'].shape)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
    sc.tl.draw_graph(adata)

    # Compute PAGA
    sc.tl.louvain(adata, resolution=1.0)

    sc.tl.paga(adata, groups='louvain')

    sc.pl.paga(adata, threshold=0.03, show=False)

    sc.tl.paga(adata, groups='louvain')


    sc.tl.draw_graph(adata, init_pos='paga')

    # sc.pl.draw_graph(adata, color=['louvain_anno'], legend_loc='on data')

    return adata


def load_setty():

    filename_preprocessed = 'setty_preprocessed.h5'

    filepath_preprocessed = os.path.join(data_folder, filename_preprocessed)
    # check if file exists
    if not os.path.exists(filepath_preprocessed):
        filename = 'setty.h5ad'
        filepath = os.path.join(data_folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            print("Downloading Setty data")
            download_Settydata(filepath)
        
        load_ad=sc.read(filepath)

        adata = sc.AnnData(load_ad.raw.X.copy())

        #dictionary to map numbers to cell labels INFO from scvelo (adata = scv.datasets.bonemarrow())
        num2label={'0': 'HSC_1',
                     '1': 'HSC_2',
                     '2': 'Ery_1',
                     '3': 'Mono_1',
                     '4': 'Precursors',
                     '5': 'CLP',
                     '6': 'Mono_2',
                     '7': 'DCs',
                     '8': 'Ery_2',
                     '9': 'Mega'}

        X_RAW = load_ad.raw.X.A
        print('preprocessing data')
        adata.X = adata.X.astype('float64')
        print('Applying zheng recipe')
        sc.pp.recipe_zheng17(adata)

        sc.tl.pca(adata, svd_solver='arpack',n_comps=79)

        print('Saving preprocessed data')
        X_PCA = adata.obsm['X_pca']
        X_TSNE = load_ad.obsm['tsne']
        labels = np.array([num2label[l] for l in load_ad.obs['clusters']])
        colors=np.array([load_ad.uns['cluster_colors'][l] for l in load_ad.obs['clusters']])
        with h5py.File(filepath_preprocessed, "w") as f:
            f.create_dataset('X_PCA50', data=X_PCA)
            f.create_dataset('X_TSNE', data=X_TSNE)
            f.create_dataset('X_RAW', data=X_RAW)
            f['labels'] = labels.astype('S')
            f['colors'] =colors.astype('S')


    with h5py.File(filepath_preprocessed, 'r') as f:
        print(f.keys())
        X_PCA = np.array(f['X_PCA50'])
        X_RAW = np.array(f['X_RAW'])
        X_TSNE = np.array(f['X_TSNE'])
        labels = np.array(f['labels']).astype(str)
        colors=np.array(f['colors']).astype(str)
    return X_RAW, X_PCA, X_TSNE, labels,colors

def download_Settydata(filepath='setty.h5ad'):
    '''
        Download human_cd34_bm_rep1.h5ad from
        https://data.humancellatlas.org/explore/projects/091cf39b-01bc-42e5-9437-f419a66c8a45/project-matrices
    '''

    url='https://storage.googleapis.com/datarepo-4ef3f5a3-bucket/0e5e329e-2709-4ceb-bfe3-97d23a652ac0/3428f967-3376-4051-b6f7-8dd84580ca5b/human_cd34_bm_rep1.h5ad?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=datarepo-jade-api%40terra-datarepo-production.iam.gserviceaccount.com%2F20240418%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240418T090949Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&requestedBy=azul-ucsc-0-public-prod%40platform-hca-prod.iam.gserviceaccount.com&userProject=datarepo-1dbff5cd&X-Goog-Signature=ab645a8f0055d5b002bf17e363951e19201b8b06a6ecb840ca52e26a227e586cb6355f06c8ed2cf0a43930ff47560c832221db5234eff689f495a21cf090960aa51fa795740f5c64bf205ce677b02ab2775c21964cc025353f34cf43f9c93fd52f2c5196bca29c83913b3857e4773541539bbabb66cf90932a665571fe14d325f225b155ba67f417f2809a0da6ad19f4c71f3c587af470451e243694b30755d4f93d184e1e6615f741d6239050f85d505732cc3f35ca0908de42daee1baadf566ad6a045319fc1a1063a46f58714211db6613725ee8f3990c3d678002900cd356c156465b5bebadb278bed89bf704406e65d36ad07359a875a7b9cd4ffb6513d'

    # Create the data folder if it doesn't exist
    directory_path, filename = os.path.split(filepath)
    os.makedirs(directory_path, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File '{filename}' downloaded and saved to '{directory_path}'.")
