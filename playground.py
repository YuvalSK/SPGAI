# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 18:44:49 2025

@author: YSK
"""

%aimport typing_extensions
from ai4bmr_datasets import Danenberg2022, Jackson2020, Keren2018
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
from openTSNE import TSNE #faster
import umap.umap_ as umap

import matplotlib.pyplot as plt
import seaborn as sns

srcs = [Danenberg2022, Jackson2020, Keren2018] 
src = srcs[0]
src_name = str(src).split(".")[-1][:-2]
print(f"Dataset: {src_name}")
base_dir = Path(f"c://Users//User/Projects/Epigenetics/{src_name}")  # can be None, resolves to ~/.cache/ai4bmr_datasets by default

dataset = src(base_dir)

# issues:
    ## Danenbergh2022 - complete (manually run an R script to convert file types) 
    ## Jackson2020 - this runs an error with pandas formating (series and Int32 indexing)
    # FileNotFoundError: [Errno 2] No such file or directory: 'c:\\Users\\User\\Projects\\Epigenetics\\Jackson2020\\02_processed\\metadata\\published'
    ## Keren2019
    # DeflateError: libdeflate_zlib_decompress returned LIBDEFLATE_INSUFFICIENT_SPACE
dataset.prepare_data()  # only needs to be run once

dataset.setup(image_version="published", mask_version="published")

dataset.setup(
    image_version="published",
    mask_version="published",
    feature_version="published", load_intensity=True,
    metadata_version="published", load_metadata=True
)

#### example properties:
#print(len(dataset.sample_ids))    # List of sample IDs
#print(dataset.images.keys()) # Dictionary of images
#print(dataset.masks.keys())  # Dictionary of masks
#print(dataset.intensity.shape)  # Cell x marker matrix
#print(dataset.metadata.shape)   # Cell x annotation matrix
#sample_id = dataset.sample_ids[0]  # get the first sample ID
#img = dataset.images[sample_id].data
#print("Image shape:", img.shape)
#dataset.intensity.to_csv(f"{src_name}.csv")

#### dimensionality reduction analysis
# --- Corr ---
spearman_corr = dataset.intensity.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap='coolwarm',
            annot_kws={"size": 3},
            vmax = 1, vmin=-1)
plt.title('Spearman Correlation Heatmap')
plt.savefig(f"Figures/{src_name}-spearman.png", dpi=600)
## the raw markers are mostly anticorrelated, not sure why
 
#change to arcsinh + Ztransform (as in Harpaz 2022)     
scaler = StandardScaler()
scaled_data = scaler.fit_transform(dataset.intensity)

pca = PCA()
pca_result = pca.fit(scaled_data)
explained_variance = pca.explained_variance_ratio_

# Scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
plt.xlabel('PCs')
plt.ylabel('Explained Variation')
plt.grid(True)
elbow1 = 3
plt.axvline(x=elbow1, c='r', linestyle="dashed", label=f"{explained_variance[:elbow1].sum()*100:.1f}% variation by {elbow1} PCs")
elbow2 = 10
plt.axvline(x=elbow2, c='r', linestyle="dotted", label=f"{explained_variance[:elbow2].sum()*100:.1f}% variation by {elbow2} PCs")
plt.legend()
plt.savefig(f"Figures/{src_name}-pca-scree.png", dpi=600)
## with elbow methods, seems 3 components are not bad => ~30% of variance


# reducing dimensions of data
pca = PCA(n_components=10)
pca_data = pca.fit_transform(scaled_data)

# 2-tSNE
tsne = TSNE(n_jobs=4, #if openTSNE
            n_components=2,
            perplexity=30, 
            learning_rate=200, 
            n_iter=300, 
            metric="euclidean",
            random_state=42, 
            verbose=True)
tsne_result = tsne.fit(pca_data)

#tsne_result = tsne.fit_transform(pca_data)

# 3-UMAP
umap_model = umap.UMAP(n_jobs=4,
                       n_components=2, 
                       n_neighbors=10, 
                       min_dist=0.05, 
                       n_epochs=200,
                       metric='euclidean',
                       random_state=42,
                       verbose=True)

umap_result = umap_model.fit_transform(pca_data)

fig, axs = plt.subplots(2, 3, figsize=(18, 5))

# PCA plots
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled_data)

axs[0,0].scatter(pca_result[:, 0], pca_result[:, 1], 
               facecolors='none',
               edgecolors='gray',
               s=10,
               alpha=0.7)
axs[0,0].set_xlabel(f'PC1 [A.U.] ({pca.explained_variance_ratio_[0]:.1%} variance)')
axs[0,0].set_ylabel(f'PC2 [A.U.] ({pca.explained_variance_ratio_[1]:.1%} variance)')

axs[0,1].scatter(pca_result[:, 0], pca_result[:, 2], 
               facecolors='none',
               edgecolors='gray',
               s=10,
               alpha=0.7)
axs[0,1].set_xlabel(f'PC1 [A.U.] ({pca.explained_variance_ratio_[0]:.1%} variance)')
axs[0,1].set_ylabel(f'PC3 [A.U.] ({pca.explained_variance_ratio_[2]:.1%} variance)')

axs[0,2].scatter(pca_result[:, 1], pca_result[:, 2], 
               facecolors='none',
               edgecolors='gray',
               s=10,
               alpha=0.7)
axs[0,2].set_xlabel(f'PC2 [A.U.] ({pca.explained_variance_ratio_[1]:.1%} variance)')
axs[0,2].set_ylabel(f'PC3 [A.U.] ({pca.explained_variance_ratio_[2]:.1%} variance)')


# t-SNE plot
axs[1,0].scatter(tsne_result[:, 0], tsne_result[:, 1],     
               facecolors='none',
               edgecolors='gray', 
               s=10,
               alpha=0.7)
axs[1,0].set_xlabel('t-SNE 1 [A.U.]')
axs[1,0].set_ylabel('t-SNE 2 [A.U.]')

# UMAP plot
axs[1,2].scatter(umap_result[:, 0], umap_result[:, 1], 
               facecolors='none',
               edgecolors='gray',
               s=10,
               alpha=0.7)
axs[1,2].set_xlabel('UMAP 1 [A.U.]')
axs[1,2].set_ylabel('UMAP 2 [A.U.]')

plt.tight_layout()
#plt.show()
plt.savefig(f'Figures/{src_name}-exploration.png', dpi=600)


