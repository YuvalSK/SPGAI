# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 18:44:49 2025

@author: YSK
"""
%aimport typing_extensions
from ai4bmr_datasets import Danenberg2022, Jackson2020, Keren2018
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

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

dataset.intensity.head()
dataset.metadata.head()

dataset.intensity.to_parquet(f"{base_dir}/{src_name}-intensity.parquet", engine="pyarrow", compression="snappy")
dataset.metadata.to_parquet(f"{base_dir}/{src_name}-metadata.parquet", engine="pyarrow", compression="snappy")

#### example properties of the dataset:
#print(len(dataset.sample_ids))    # List of sample IDs
#print(dataset.images.keys()) # Dictionary of images
#print(dataset.masks.keys())  # Dictionary of masks
#print(dataset.intensity.shape)  # Cell x marker matrix
#print(dataset.metadata.shape)   # Cell x annotation matrix
#sample_id = dataset.sample_ids[0]  # get the first sample ID
#img = dataset.images[sample_id].data
#print("Image shape:", img.shape)

#to load the data
df = pd.read_parquet(f"{base_dir}/{src_name}.parquet", engine="pyarrow")
df_meta = pd.read_parquet(f"{base_dir}/{src_name}-metadata.parquet", engine="pyarrow")
#print(set(df_meta.label))

# --- Corr ---
spearman_corr = df.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap='coolwarm',
            annot_kws={"size": 3},
            vmax = 1, vmin=-1)
plt.title('Correlation Heatmap (p)')
plt.savefig(f"Figures/{src_name}-spearman.png", dpi=600)

#normalization
## 1) Z transform
scaler = StandardScaler()
z_data = scaler.fit_transform(df)

## 2) arcsinh     
scFac=5
scaled_data=np.arcsinh(z_data/scFac)

## To do:
### 3) residual to accont for systematic effects 

# Compute total intensity per cell
scaled_data = pd.DataFrame(scaled_data)
scaled_data = scaled_data.sum(axis=1).values.reshape(-1, 1)

# Initialize empty DataFrame for residuals
df_residuals = pd.DataFrame(index=scaled_data.index, columns=scaled_data.columns)

# For each marker, regress on total intensity and store residuals
for marker in df.columns:
    y = df[marker].values.reshape(-1, 1)
    model = LinearRegression().fit(total_intensity, y)
    y_pred = model.predict(total_intensity)
    residuals = (y - y_pred).ravel()
    df_residuals[marker] = residuals
    
spearman_corr_res = df_residuals.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr_res, cmap='coolwarm',
            annot_kws={"size": 3},
            vmax = 1, vmin=-1)
plt.title('Residuals Correlation Heatmap (p)')
plt.savefig(f"Figures/{src_name}-spearman-res.png", dpi=600)


#### dimensionality reduction analysis
#normalization (as in Harpaz 2022):



pca = PCA()
pca_result = pca.fit(scaled_data)
explained_variance = pca.explained_variance_ratio_

def pca_plots(data, file_name):  
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    ########## bottom plot ########## 
    axs[1].plot(range(1, len(explained_variance) + 1), explained_variance, 
                color='gray')
    
    cumulative_variance = explained_variance.cumsum()
    axs[1].plot(range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        color='k',
        linestyle='--',
        marker='o',
    )
    
    axs[1].set_xlabel('PCs')
    axs[1].set_ylabel('Explained Variation')
    elbow1 = 2
    axs[1].axvline(x=elbow1, c='r', linestyle="dashed", label=f"{explained_variance[:elbow1].sum()*100:.1f}% variation by {elbow1} PCs")
    axs[1].legend(loc='center right')
    elbow2 = 10
    axs[1].axvline(x=elbow2, c='k', linestyle="dashed", label=f"{explained_variance[:elbow2].sum()*100:.1f}% variation by {elbow1} PCs")
    axs[1].legend(loc='center right')
    
    ########## top plot ########## 
    palette = {'epithelial': '#1f77b4', 'non-epithelial': '#ff7f0e'}  # blue & orange
    df_meta['super_category'] = df_meta['is_epithelial'].map({1: 'epithelial', 0: 'non-epithelial'})
    
    for category in ['epithelial', 'non-epithelial']:
        mask = df_meta['super_category'].eq(category).to_numpy(dtype=bool)
        axs[0].scatter(pca_result[mask, 0], pca_result[mask, 1],
                       alpha=0.5,
                       label=category,
                       color=palette[category])
    
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axs[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=600)

pca_plots(scaled_data, f"Figures/{src_name}-pca-norm-res-main.png")

# reducing dimensions of data
pca = PCA(n_components=10)
pca_data = pca.fit_transform(scaled_data)


# 2-tSNE
tsne1 = TSNE(n_jobs=2,
            n_components=2,
            perplexity=30, 
            learning_rate=200, 
            n_iter=250, 
            metric="euclidean",
            random_state=1111, 
            verbose=True)
tsne1_result = tsne1.fit(pca_data)

df = pd.DataFrame(tsne1_result)
df.to_csv(f'Results/{src_name}-dm-res-tsne1.csv')

tsne2 = TSNE(n_jobs=2,
            n_components=2,
            perplexity=50, 
            learning_rate=200, 
            n_iter=300, 
            metric="euclidean",
            random_state=1111, 
            verbose=True)
tsne2_result = tsne2.fit(pca_data)

df = pd.DataFrame(tsne2_result)
df.to_csv(f'Results/{src_name}-dm-res-tsne2.csv')

# 3-UMAP
umap1_model = umap.UMAP(n_jobs=2,
                       n_components=2, 
                       n_neighbors=15, 
                       min_dist=0.05, 
                       n_epochs=100,
                       metric='euclidean',
                       verbose=True)

umap1_result = umap1_model.fit_transform(pca_data)

df = pd.DataFrame(umap1_result)
df.to_csv(f'Results/{src_name}-dm-res-umap1.csv')

umap2_model = umap.UMAP(n_jobs=2,
                       n_components=2, 
                       n_neighbors=45, 
                       min_dist=0.05, 
                       n_epochs=100,
                       metric='euclidean',
                       verbose=True)

umap2_result = umap2_model.fit_transform(pca_data)

df = pd.DataFrame(umap2_result)
df.to_csv(f'Results/{src_name}-dm-res-umap2.csv')

def plot_dm(file_name):
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))
    
    # t-SNE plot
    axs[0,0].scatter(tsne1_result[:, 0], tsne1_result[:, 1],     
                   facecolors='none',
                   edgecolors='gray', 
                   alpha=0.2)
    axs[0,0].set_xlabel('t-SNE 1 [A.U.]')
    axs[0,0].set_ylabel('t-SNE 2 [A.U.]')
    axs[0,0].set_title('t-SNE with perplexity = 30')
    
    axs[0,1].scatter(tsne2_result[:, 0], tsne2_result[:, 1],     
                   facecolors='none',
                   edgecolors='gray', 
                   alpha=0.2)
    axs[0,1].set_xlabel('t-SNE 1 [A.U.]')
    axs[0,1].set_ylabel('t-SNE 2 [A.U.]')
    axs[0,1].set_title('t-SNE with perplexity = 50')
    
    
    # UMAP plot
    axs[1,0].scatter(umap1_result[:, 0], umap1_result[:, 1], 
                   facecolors='none',
                   edgecolors='gray',
                   alpha=0.2)
    axs[1,0].set_xlabel('UMAP 1 [A.U.]')
    axs[1,0].set_ylabel('UMAP 2 [A.U.]')
    axs[1,0].set_title('UMAP with n_neighbors = 15')
    
    axs[1,1].scatter(umap2_result[:, 0], umap2_result[:, 1], 
                   facecolors='none',
                   edgecolors='gray',
                   alpha=0.2)
    axs[1,1].set_xlabel('UMAP 1 [A.U.]')
    axs[1,1].set_ylabel('UMAP 2 [A.U.]')
    axs[1,1].set_title('UMAP with n_neighbors = 45')
    
    labels = df_meta['label'].unique()
    palette = {label: color for label, color in zip(labels, sns.color_palette(n_colors=len(labels)))}
    
    for label in labels:
        mask = df_meta['label'].eq(label).to_numpy(dtype=bool)
        axs[0,0].scatter(tsne1_result[mask, 0], tsne1_result[mask, 1],
                       alpha=0.2,
                       label=label,
                       color=palette[label])  # Fallback color
        axs[0,1].scatter(tsne2_result[mask, 0], tsne2_result[mask, 1],
                       alpha=0.2,
                       label=label,
                       color=palette[label])
        axs[1,0].scatter(umap1_result[mask, 0], umap1_result[mask, 1],
                       alpha=0.2,
                       label=label,
                       color=palette[label])
        axs[1,1].scatter(umap2_result[mask, 0], umap2_result[mask, 1],
                       alpha=0.2,
                       label=label,
                       color=palette[label])
        
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=6, fontsize='x-small', frameon=True)    
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Leave space for the legend

    plt.savefig(file_name, dpi=600)

plot_dm(f'Figures/{src_name}-dr-norm-res-all.png')

