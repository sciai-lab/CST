import numpy as np
import requests
import scanpy as sc
import os
import h5py

import scvelo as scv

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), facecolor='white')
file_path = os.path.abspath(__file__)
# Extract the directory path
directory_current_file = os.path.dirname(file_path)
data_folder = os.path.join(directory_current_file, '../..', 'Data','single_cell_data')
os.makedirs(data_folder, exist_ok=True)

sc.settings.datasetdir=data_folder

def load_paul15():
    filename_preprocessed = 'paul15_preprocessed.h5'
    filepath_preprocessed=os.path.join(data_folder,filename_preprocessed)
    #check if file exists
    if not os.path.exists(filepath_preprocessed):
        adata = sc.datasets.paul15()
        X_RAW=adata.X.copy()
        print('preprocessing data')
        preprocess_paul15(adata)


        print('Saving preprocessed data')
        X_PCA = adata.obsm['X_pca']
        X_PAGA = adata.obsm['X_draw_graph_fa']
        labels = np.array(adata.obs['paul15_clusters'])
        with h5py.File(filepath_preprocessed, "w") as f:
            f.create_dataset('X_PCA50', data=X_PCA)
            f.create_dataset('X_PAGA', data=X_PAGA)
            f.create_dataset('X_RAW', data=X_RAW)
            f['labels'] = labels.astype('S')
        print('Saved %s in %s' % (filename_preprocessed, data_folder))
    with h5py.File(filepath_preprocessed, 'r') as f:
        print(f.keys())
        X_PCA = np.array(f['X_PCA50'])
        X_RAW = np.array(f['X_RAW'])
        X_PAGA = np.array(f['X_PAGA'])
        labels = np.array(f['labels']).astype(str)
    return X_RAW,X_PCA,X_PAGA,labels

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
        https://data.humancellatlas.org/explore/projects/091cf39b-01bc-42e5-9437-f419a66c8a45/project-matrices
    '''

    url='https://storage.googleapis.com/datarepo-4ef3f5a3-bucket/0e5e329e-2709-4ceb-bfe3-97d23a652ac0/3428f967-3376-4051-b6f7-8dd84580ca5b/human_cd34_bm_rep1.h5ad?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=datarepo-jade-api%40terra-datarepo-production.iam.gserviceaccount.com%2F20231101%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231101T122535Z&X-Goog-Expires=900&X-Goog-SignedHeaders=host&requestedBy=azul-ucsc-0-public-prod%40platform-hca-prod.iam.gserviceaccount.com&userProject=datarepo-1dbff5cd&X-Goog-Signature=239d77c93c65ae88471ea1e7a7023a6af8afdd0bc42fc88251ef6001627d18d836dd064b80d9b9e65bf762db1aeabec96c50c5d030cd8ea47d93acf9f45a29075d783b49f2e99f3c6c89e4ad4f42e3257d6aadb01ca328292fe8238556b2f0d44b48b1fb977716e5a8ea7de53379b9990ef82f5d6e60cba67d67ba61452774a2edc2898e7aca968fe75792e4f38913cdc1076f01428953c5e983ea588c9040edd273140f571db2b5110443f2a13ea475e591a0d46d213dcbe0fcdf2da327d620355687d05331b1bdee10cff55652bcadee31ba20318069b512a727c0878a081b07811b47beae7a11fb9e034835bfe04c81ad223ffe8d54e0f19bc1c5f9aa5920'

    # Create the data folder if it doesn't exist
    directory_path, filename = os.path.split(filepath)
    os.makedirs(directory_path, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True)
    with open(filepath, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"File '{filename}' downloaded and saved to '{directory_path}'.")
