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

params = {'axes.titlesize': 30,
          'legend.fontsize': 16,
          'figure.figsize': (16, 10),
          'axes.labelsize': 8,
          'axes.titlesize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'figure.titlesize': 30}

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

srcs = [Danenberg2022, Jackson2020, Keren2018] 
src = srcs[0]
src_name = str(src).split(".")[-1][:-2]
print(f"Dataset: {src_name}")
base_dir = Path(f"c://Users//User/Projects/Epigenetics/{src_name}")  # can be None, resolves to ~/.cache/ai4bmr_datasets by default

dataset = src(base_dir)

# issues:
    ## Danenbergh2022 - manually run an R script to convert file types
    ## Jackson2020 - this runs an error with pandas formating (series and Int32 indexing)
    # FileNotFoundError: [Errno 2] No such file or directory: 'c:\\Users\\User\\Projects\\Epigenetics\\Jackson2020\\02_processed\\metadata\\published'
    ## Keren2019 - DeflateError: libdeflate_zlib_decompress returned LIBDEFLATE_INSUFFICIENT_SPACE

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


#to load the data
df = pd.read_parquet(f"{base_dir}/{src_name}.parquet", engine="pyarrow")
df_meta = pd.read_parquet(f"{base_dir}/{src_name}-metadata.parquet", engine="pyarrow")
print(set(df_meta.label)) #cell types

# --- Raw corr ---
spearman_corr = df.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap='coolwarm',
            vmax = 1, vmin=-1)
plt.tight_layout()
plt.savefig(f"Figures/{src_name}-spearman-raw.png", dpi=600)


# --- Normalization ---
# exploration to determine how to reduce systematic technical variation?
## archsinh and then regress out noise or the opposite? 
plt.hist(df["histone_h3"], bins=10, facecolor='none', edgecolor='k', 
         label='H3 raw', log=True)
plt.hist(np.arcsinh(df["histone_h3"]), bins=20, 
         edgecolor='blue', 
         label='H3 arcsinh', log=True)
plt.legend()
plt.show()
## The common approch is arcsinh → regress. This reduces the influence of high-intensity outliers (by compressing extreme values).
## for this analysis, we will do the opposite (regress out → arcsinh). This increases the influence of high-intensity outliers (by using the raw cytof values) 

## regress out by total signal or core histones? 
tot = df.sum(axis=1)
plt.scatter(df["histone_h3"], tot, facecolor='none', edgecolor='k')
plt.plot(df["histone_h3"],df["histone_h3"],c='k',linestyle='--')
plt.xlabel('Histone H3')
plt.ylabel('Total signal')
plt.show()
## for this analysis, we assume the assume core histones (e.g., H3) should be expressed at a similar level in all cells

def norm(data, idv):
        
  #step 1: regressing out systematic noise by idv
  df_residuals = pd.DataFrame(index=data.index, columns=data.columns)
  for marker in data.columns:
    y = data[marker].values.reshape(-1, 1)
    model = LinearRegression().fit(idv, y)
    y_pred = model.predict(idv)
    residuals = (y - y_pred).ravel()
    df_residuals[marker] = residuals
  # Step 2: arcsinh normalization
  scFac=5
  scaled_data=np.arcsinh(df_residuals/scFac)
  # Step 3: Z-transform
  scaler = StandardScaler()
  z_data = scaler.fit_transform(scaled_data)
  #back to df
  z_data = pd.DataFrame(z_data, columns=data.columns, index=data.index)
  return z_data

h3_intensity = df["histone_h3"].values.reshape(-1, 1) # Reshape to 2D array
df_norm = norm(df, h3_intensity)

# after normalization
spearman_corr = df_norm.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap='coolwarm',
            vmax = 1, vmin=-1)
plt.tight_layout()
plt.savefig(f"Figures/{src_name}-spearman-norm.png", dpi=600)

# --- Dimensionality reduction ---
pca = PCA()
pca_result = pca.fit(df_norm)
explained_variance = pca.explained_variance_ratio_

def wrap_label(label, width=10):
    return '\n'.join([label[i:i+width] for i in range(0, len(label), width)])

    
def pca_plots(data, file_name):
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    palette = {'epithelial': 'red', 'non-epithelial': 'blue'}
    df_meta['super_category'] = df_meta['is_epithelial'].map({1: 'epithelial', 0: 'non-epithelial'})
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])
    ax2 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ########## left plot ########## 
    ax2.bar(range(1, len(explained_variance) + 1), explained_variance, 
                facecolor='none',
                edgecolor='k')
    
    cumulative_variance = explained_variance.cumsum()
    ax2.plot(range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        color='k')
    ax2.set_xlabel('PCs')
    ax2.set_ylabel('Variation')
    elbow1 = 2
    ax2.axvline(x=elbow1, c='purple', linestyle="--", label=f"{explained_variance[:elbow1].sum()*100:.1f}% by {elbow1} PCs")
    elbow2 = 10
    #ax2.axvline(x=elbow2, c='gray', linestyle="dotted", label=f"{explained_variance[:elbow2].sum()*100:.1f}% by {elbow2} PCs")
    print(f"{explained_variance[:elbow2].sum()*100:.1f}% by {elbow2} PCs")
    ax2.set_ylim(0,1.01)
    ax2.legend(loc='center right', frameon=True)
    ########## top plot ########## 
    for category in ['epithelial', 'non-epithelial']:
        mask = df_meta['super_category'].eq(category).to_numpy(dtype=bool)
        ax1.scatter(pca_result[mask, 1], pca_result[mask, 0],
                       alpha=0.2,
                       label=category,
                       facecolors='none',
                       edgecolor=palette[category])
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_ylabel(f'PC1 ({explained_variance[0]:.1%})')
    ax1.set_xlabel(f'PC2 ({explained_variance[1]:.1%})')
    
    ########## middle plot ########## 
    pc2_weights = pca.components_[1]
    features = data.columns
    sorted_indices = np.argsort(np.abs(pc2_weights))
    sorted_features = features[sorted_indices]
    sorted_weights = pc2_weights[sorted_indices]
    ax3.barh(sorted_features, sorted_weights, color='k')
    ax3.axvline(x=0, c='k')
    ax3.set_xlabel('PC2 weights')
    ax3.set_yticks(range(len(sorted_features)))
    #wrapped_labels = [wrap_label(label, width=5) for label in sorted_features]
    ax3.set_yticklabels(sorted_features, 
                        rotation=0, 
                        fontsize=6)
    
    plt.tight_layout()
    plt.savefig(file_name, dpi=600)

pca_plots(df_norm, f"Figures/{src_name}-pca-norm.png")

# for faster computation
n_d = 10
pca = PCA(n_components=n_d)
pca_data = pca.fit_transform(df_norm)

# 2-tSNE
tsne1 = TSNE(n_jobs=2,
            n_components=2,
            perplexity=10, 
            learning_rate=200, 
            n_iter=200, 
            metric="euclidean",
            random_state=1111, 
            verbose=True)
tsne1_result = tsne1.fit(pca_data)

df = pd.DataFrame(tsne1_result)
df.to_csv(f'Results/{src_name}-dm-tsne1.csv')

tsne2 = TSNE(n_jobs=2,
            n_components=2,
            perplexity=30, 
            learning_rate=200, 
            n_iter=200, 
            metric="euclidean",
            random_state=1111, 
            verbose=True)
tsne2_result = tsne2.fit(pca_data)

df = pd.DataFrame(tsne2_result)
df.to_csv(f'Results/{src_name}-dm-tsne2.csv')

# 3-UMAP
umap1_model = umap.UMAP(n_jobs=2,
                       n_components=2, 
                       n_neighbors=15, 
                       min_dist=0.05, 
                       n_epochs=200,
                       metric='euclidean',
                       verbose=True)

umap1_result = umap1_model.fit_transform(pca_data)

df = pd.DataFrame(umap1_result)
df.to_csv(f'Results/{src_name}-dm-umap1.csv')

umap2_model = umap.UMAP(n_jobs=2,
                       n_components=2, 
                       n_neighbors=50, 
                       min_dist=0.05, 
                       n_epochs=200,
                       metric='euclidean',
                       verbose=True)

umap2_result = umap2_model.fit_transform(pca_data)

df = pd.DataFrame(umap2_result)
df.to_csv(f'Results/{src_name}-dm-umap2.csv')

def plot_dm(file_name):
    fig, axs = plt.subplots(2, 2, figsize=(18, 5))
    
    # t-SNE plot
    axs[0,0].scatter(tsne1_result[:, 0], tsne1_result[:, 1],     
                   facecolors='none',
                   edgecolors='gray', 
                   alpha=0.2)
    axs[0,0].set_xlabel('t-SNE 1 [A.U.]')
    axs[0,0].set_ylabel('t-SNE 2 [A.U.]')
    axs[0,0].set_title('t-SNE with perplexity = 10')
    
    axs[0,1].scatter(tsne2_result[:, 0], tsne2_result[:, 1],     
                   facecolors='none',
                   edgecolors='gray', 
                   alpha=0.2)
    axs[0,1].set_xlabel('t-SNE 1 [A.U.]')
    axs[0,1].set_ylabel('t-SNE 2 [A.U.]')
    axs[0,1].set_title('t-SNE with perplexity = 30')
    
    
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
    axs[1,1].set_title('UMAP with n_neighbors = 50')
    
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
    fig.legend(handles, labels, loc='outside right upper',
               ncol=6, fontsize='x-small', frameon=True)    
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Leave space for the legend

    plt.savefig(file_name, dpi=600)

plot_dm(f'Figures/{src_name}-dr-norm-all.png')

################# draft code ###################
#### example properties of the dataset:
#print(len(dataset.sample_ids))    # List of sample IDs
#print(dataset.images.keys()) # Dictionary of images
#print(dataset.masks.keys())  # Dictionary of masks
#print(dataset.intensity.shape)  # Cell x marker matrix
#print(dataset.metadata.shape)   # Cell x annotation matrix
#sample_id = dataset.sample_ids[0]  # get the first sample ID
#img = dataset.images[sample_id].data
#print("Image shape:", img.shape)