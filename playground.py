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
          'axes.labelsize': 16,
          'axes.titlesize': 12,
          'xtick.labelsize': 16,
          'ytick.labelsize': 16,
          'figure.titlesize': 30}

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')

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


#load the data from parquet
df = pd.read_parquet(f"{base_dir}/{src_name}.parquet", engine="pyarrow")

#load labels
df_meta = pd.read_parquet(f"{base_dir}/{src_name}-metadata.parquet", engine="pyarrow")
print(set(df_meta.is_epithelial)) #cell types
print(set(df_meta.label)) #cell types

#labels for visualization
df_meta['epithelial_label'] = df_meta.apply(
    lambda row: 'epithelial' if row['is_epithelial'] == 1 else row['label'],
    axis=1
)

marker_colors = {
    # Epithelial (Tumor)
    form('pan_cytokeratin'): 'orange',
    form('cytokeratin_8_18'): 'orange',
    form('cytokeratin_5'): 'orange',
    form('estrogen_receptor_alpha'): 'orange',
    form('c_erb_b_2_her2_3b5'): 'orange',
    form('c_erb_b_2_her2_d8f12'): 'orange',

    # TME - Stromal + Immune
    form('cxcl12_sdf_1'): 'blue',
    form('cd3'): 'blue',
    form('cd4'): 'blue',
    form('cd8a'): 'blue',
    form('cd20'): 'blue',
    form('cd45'): 'blue',
    form('cd45ra'): 'blue',
    form('cd68'): 'blue',
    form('cd163'): 'blue',
    form('foxp3'): 'blue',
}

# --- Raw corr ---
spearman_corr = df.corr(method='spearman')
plt.figure(figsize=(16, 12))
sns.heatmap(spearman_corr, cmap='coolwarm',
            vmax = 1, vmin=-1)
plt.tight_layout()
plt.savefig(f"Figures/{src_name}-spearman-raw.png", dpi=600)
'''
markers are mainly positivly correlated without normaization, suggesting systematic/technical noise... 
'''

# --- Normalization ---

def plot_core_h3(df, file_name):
    
    df['tot'] = df.sum(axis=1)  
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax2 = fig.add_subplot(gs[0, 0]) # avg
    ax1 = fig.add_subplot(gs[1, 1]) # h3
    ax3 = fig.add_subplot(gs[0, 1]) # avg vs h3
    ax1.hist(df["histone_h3"], bins=20, 
             facecolor='none', edgecolor='k', 
             label='Core H3', log=True)
    ax1.set_ylabel('Freq. [#]')
    ax1.set_xlabel('Intensity')
    ax1.legend()
    ax2.hist(df["tot"], bins=20, 
             facecolor='none', edgecolor='k', 
             label='Total signal', log=True)
    ax2.set_ylabel('Freq. [#]')
    ax2.set_xlabel('Intensity')
    ax2.legend()
    ax3.scatter(np.arcsinh(df["histone_h3"]), 
                np.arcsinh(df["tot"]), 
                facecolor='none', edgecolor='k',
                alpha=0.5)
    ax3.plot(np.arcsinh(df["histone_h3"]),np.arcsinh(df["histone_h3"]),c='gray',linestyle='--')
    ax3.set_xlabel('Arcsinh(H3)')
    ax3.set_ylabel('Archsinh(Total signal)')
    plt.tight_layout()
    plt.savefig(file_name, dpi=600)
    
plot_core_h3(df, f"Figures/{src_name}-h3.png")

def norm(data, idv):
    """
    how to reduce systematic technical variation?
    The common approch arcsinh → regress. This reduces the influence of high-intensity outliers (by compressing extreme values).
    Here:              regress out → arcsinh
    This increases the influence of high-intensity outliers by using the measured intensity
    """
    
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

spearman_corr = df_norm.corr(method='spearman')
plt.figure()
sns.heatmap(spearman_corr, cmap='coolwarm',
            vmax = 1, vmin=-1)
plt.tight_layout()
plt.savefig(f"Figures/{src_name}-spearman-norm.png", dpi=600)

# --- Dimensionality reduction ---

# --- PCA ---

pca = PCA()
pca_result = pca.fit(df_norm)
explained_variance = pca.explained_variance_ratio_


def form(x):
    return x.strip().lower().replace('-', '_')
    
def pca_plot(data, tags, marker_colors, file_name):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax_pc1 = fig.add_subplot(gs[0, 1])
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_pc2 = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis('off')  # No axis needed for table
    # Plot PC1 weights (Top Left)
    pc1_weights = pca.components_[0]
    features = data.columns
    sorted_indices_pc1 = np.argsort(np.abs(pc1_weights))
    sorted_features_pc1 = features[sorted_indices_pc1]
    sorted_weights_pc1 = pc1_weights[sorted_indices_pc1]
    ax_pc1.barh(sorted_features_pc1, sorted_weights_pc1, color='gray')
    ax_pc1.axvline(x=0, c='k')
    ax_pc1.set_xlabel('PC1 weights')
    ax_pc1.set_yticks(range(len(sorted_features_pc1)))
    ax_pc1.set_yticklabels(sorted_features_pc1, fontsize=8)
    # Plot PC2 weights (Bottom Right)
    pc2_weights = pca.components_[1]
    features = data.columns
    sorted_indices_pc2 = np.argsort(np.abs(pc2_weights))
    sorted_features_pc2 = features[sorted_indices_pc2]
    sorted_weights_pc2 = pc2_weights[sorted_indices_pc2]
    y_pos = np.arange(len(sorted_features_pc2))
    ax_pc2.barh(y_pos, sorted_weights_pc2, color='gray')
    ax_pc2.axvline(x=0, c='k')
    ax_pc2.set_xlabel('PC2 weights')
    ax_pc2.set_yticks(y_pos)  # Set y positions to the correct range
    ax_pc2.set_yticklabels(sorted_features_pc2, fontsize=8)
    for label in ax_pc2.get_yticklabels():
        feature = form(label.get_text())
        color = marker_colors.get(feature, 'black')
        label.set_color(color)
        
    for label in ax_pc1.get_yticklabels():
        feature = form(label.get_text())
        color = marker_colors.get(feature, 'black')
        label.set_color(color)
    # Plot PC1 x PC2 space (Top Right) - PC1 on y-axis, PC2 on x-axis
    labels = {1:"Epithelial", 0:"TME"}
    for g in range(len(labels)):
        mask = df_meta[tags].eq(g).to_numpy(dtype=bool)
        ax_scatter.scatter(
            pca_result[mask, 1],  # PC2 on x-axis
            pca_result[mask, 0],  # PC1 on y-axis
            s=3,
            alpha=0.2,
            label=labels[g]
        )
    ax_scatter.legend(loc='upper right', frameon=True)
    ax_scatter.set_xlabel(f'PC2 ({explained_variance[1]:.1%})')
    ax_scatter.set_ylabel(f'PC1 ({explained_variance[0]:.1%})')
    
    # Data for table
    table_data = [
        ['cxcl12_sdf_1', 'TME', 'Stromal: chemokine'],
        ['cd68', 'TME', 'Immune: macrophage'],
        ['cd163', 'TME', 'Immune: M2 Macrophage'],
        ['foxp3', 'TME', 'Immune: regulatory T-cell'],
        ['cd45', 'TME', 'Immune: pan-leukocyte'],
        ['cd45ra', 'TME', 'Immune: naive T-cell'],
        ['cd3', 'TME', 'Immune: mature T-cell'],
        ['cd4', 'TME', 'Immune: helper T-cell'],
        ['cd8a', 'TME', 'Immune: cytotoxic T-cell'],
        ['cd20', 'TME', 'Immune: B-cell'],
        ['pan_cytokeratin', 'Epithelial', 'General epithelial'],
        ['cytokeratin_8_18', 'Epithelial', 'Luminal epithelial'],
        ['c_erb_b_2_her2_3b5', 'Epithelial', 'HER2 oncogene'],
        ['estrogen_receptor_alpha', 'Epithelial', 'Hormone receptor (ERα)'],
        ['c_erb_b_2_her2_d8f12', 'Epithelial', 'HER2 oncogene'],
        ['cytokeratin_5', 'Epithelial', 'Basal epithelial']
    ]
        
    # Column labels
    columns = ['Marker name', 'Tag', 'Function']
    
    # Create the table
    table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='left',
        colLoc='left',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            continue
        if col == 1:  # Tag column
            tag_val = cell.get_text().get_text()
            if "Epithelial" in tag_val:
                cell.set_facecolor("#ffa500")   # Light orange
            elif "TME" in tag_val:
                cell.set_facecolor("#cce0ff")   # Light blue
    # Header formatting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(12)
            cell.set_facecolor('#e0e0e0')    
    sns.despine(ax=ax_pc1, top = True, right = True)
    ax_pc1.text(-0.15, 1.05, '(b)', 
            transform=ax_pc1.transAxes,
            fontsize=14, 
            fontweight='bold', 
            va='top', 
            ha='right')
    sns.despine(ax=ax_scatter, top = True, right = True)
    ax_scatter.text(-0.15, 1.05, '(a)', 
            transform=ax_scatter.transAxes,
            fontsize=14, 
            fontweight='bold', 
            va='top', 
            ha='right')
    sns.despine(ax=ax_pc2, top = True, right = True)
    ax_pc2.text(-0.15, 1.05, '(c)', 
            transform=ax_pc2.transAxes,
            fontsize=14, 
            fontweight='bold', 
            va='top', 
            ha='right')
    ax_scatter.grid(False)
    ax_pc2.grid(False)
    ax_pc1.grid(False)
    ax_table.text(-0.15, 1.05, '(d)', 
            transform=ax_table.transAxes,
            fontsize=14, 
            fontweight='bold', 
            va='top', 
            ha='right')
    

    plt.tight_layout()
    plt.savefig(file_name, dpi=600)
    plt.close(fig)
pca_plot(df_norm, "is_epithelial", marker_colors, f"Figures/{src_name}-pca-norm.png")

# to reduce computational time and memory use
# Note: we loose some information (~50% variance)
n_d = 15
pca = PCA(n_components=n_d)
pca_data = pca.fit_transform(df_norm)

#df_norm = df_norm.reset_index(drop=True)  # Reset to default index
#X = df_norm.values  # Convert to NumPy array

# --- tSNE & UMAP ---
ps = [5, 10, 15, 30, 50]
us = [5, 10, 15, 30, 50]
m = 'euclidean' # or cosine

for p in ps: 
    print(p)
    tsne = TSNE(n_jobs=2,
                n_components=2,
                perplexity=p, 
                learning_rate=200, 
                n_iter=200, 
                metric=m,
                random_state=1111, 
                verbose=False)
    tsne_result = tsne.fit(pca_data)
    temp = pd.DataFrame(tsne_result)
    temp.to_csv(f'Results/{src_name}-tsne-{p}-{m}.csv')
    del temp
    del tsne_result 
    
# UMAP
for u in us:
    print(u)
    umap_model = umap.UMAP(n_jobs=2,
                           n_components=2, 
                           n_neighbors=u, 
                           min_dist=0.1, 
                           n_epochs=200,
                           metric=m,
                           verbose=False)
    
    umap_result = umap_model.fit_transform(pca_data)
    
    temp = pd.DataFrame(umap_result)
    temp.to_csv(f'Results/{src_name}-umap-{u}-{m}.csv')
    del temp
    del umap_result 

def plot_dr(tsne_perplexities, umap_neighbors):
    
    color_map = {1: 'orange', 0: 'blue'}
    label_map = {1: 'Epithelial', 0: 'TME'}
    
    # Load results
    tsne_results = [
        pd.read_csv(f'Results/{src_name}-dm-tsne-{p}-m.csv').to_numpy() 
        for p in tsne_perplexities
    ]
    umap_results = [
        pd.read_csv(f'Results/{src_name}-dm-umap-{n}-m.csv').to_numpy() 
        for n in umap_neighbors
    ]
    
    # Set up figure: 2 rows x 5 cols
    fig, axs = plt.subplots(2, len(umap_results), figsize=(25, 10))
    
    # --- Top row: t-SNE ---
    for i, (p, result) in enumerate(zip(tsne_perplexities, tsne_results)):
        ax = axs[0, i]
        for label in [0, 1]:
            mask = df_meta['is_epithelial'].eq(label).to_numpy(dtype=bool)
            ax.scatter(result[mask, 1], result[mask, 2],
                       alpha=0.2, color=color_map[label], label=label_map[label], s=2)
        ax.set_xlabel('t-SNE 1 [A.U.]')
        ax.set_ylabel('t-SNE 2 [A.U.]')
        ax.set_title(f't-SNE, perplexity={p}')
    
    # --- Bottom row: UMAP ---
    for i, (n, result) in enumerate(zip(umap_neighbors, umap_results)):
        ax = axs[1, i]
        for label in [0, 1]:
            mask = df_meta['is_epithelial'].eq(label).to_numpy(dtype=bool)
            ax.scatter(result[mask, 1], result[mask, 2],
                       alpha=0.2, color=color_map[label], label=label_map[label], s=2)
        ax.set_xlabel('UMAP 1 [A.U.]')
        ax.set_ylabel('UMAP 2 [A.U.]')
        ax.set_title(f'UMAP, n_neighbors={n}')
    
    # Remove unused bottom-right axes
    #for j in range(3, 5):
    #    fig.delaxes(axs[1, j])
    
    # Shared legend
    handles, labels_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper right', bbox_to_anchor=(1.12, 1), borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(f'Figures/{src_name}-dr-summary.png', dpi=500)
    #plt.close(fig)

plot_dr(ps, us)

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