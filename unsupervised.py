import os
import torch
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from matplotlib.ticker import FixedLocator
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch import nn
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from Autoencoder import Encoder
import pickle
import matplotlib as mpl
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 20

DEFAULT_SPINE_W = 2.2   
DEFAULT_TICK_W  = 2.2   
DEFAULT_TICK_LEN = 7    

def apply_heavy_axes(ax, spine_width=DEFAULT_SPINE_W, tick_width=DEFAULT_TICK_W, tick_length=DEFAULT_TICK_LEN):
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.tick_params(width=tick_width, length=tick_length)

LATENT_DIM = 14
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import re

RENAME_MAP = {2: 0, 1: 1, 3: 2, 0: 3, 4: 4}

def remap_labels(labels, mapping):
    labels = np.asarray(labels)
    new_labels = labels.copy()
    for old, new in mapping.items():
        mask = (labels == old)
        new_labels[mask] = new
    return new_labels

def _normalize_col(col: str) -> str:
    col = str(col).strip().lower()
    col = col.replace('μl', 'ul')  
    col = col.replace('μ', 'u')
    col = col.replace('．', '.').replace('。', '.')
    col = re.sub(r'[^a-z0-9]+', '', col)
    return col

def _find_column(df, candidates_regex):
    norm_map = {_normalize_col(c): c for c in df.columns}
    for norm, raw in norm_map.items():
        if re.search(candidates_regex, norm):
            return raw
    return None

class ImageWithPhysicalDataset(Dataset):
    def __init__(self, excel_file, image_folder="blueOnly", sheet_name=0, exp_col=None):
        df = pd.read_excel(excel_file, sheet_name='data', header=1, engine='openpyxl')
        df.columns = (df.columns.astype(str)
                    .str.replace(r'\s+', ' ', regex=True)
                    .str.strip())

        df = df.dropna(how='all', axis=1)  
        df = df.dropna(how='all')          

        if exp_col is None:
            candidates = r'(?:^|.*)(exp|experiment)(no|id|index)?\d*.*'
            exp_col_found = _find_column(df, candidates)
            if exp_col_found is None:
                exp_col_found = _find_column(df, r'^(expno|expid|experimentno|experimentid|no|id)$')
            exp_col = exp_col_found


        df['exp_id'] = df['Exp.No.'].astype(str).str.extract(r'(\d+)').astype(int)
        df['image_filename'] = df['exp_id'].astype(int).apply(lambda x: f"Exp{x:03d}_blueOnly.bmp")
        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.transform = transforms.ToTensor()

        wanted_phys = {
            'Nozzle inner diameter (mm)': r'(nozzle|inner).*diameter.*mm',
            'Flow rate (μl/min)':         r'(flow).*rate.*(ul|min|μl|min)'
        }

        col_map = {}
        for canonical, regex in wanted_phys.items():
            found = _find_column(self.df, regex)
            col_map[canonical] = found

        self.phys_cols = list(col_map.values())
        X_phys = self.df[self.phys_cols].apply(pd.to_numeric, errors='coerce')

        self.scaler = StandardScaler()
        self.scaled_phys = torch.tensor(self.scaler.fit_transform(X_phys.values), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, row["image_filename"])
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return self.scaled_phys[idx], image, row['image_filename']

def encode_all_images_with_phys(encoder, dataset, device='cuda'):
    encoder.eval()
    all_latents, all_phys, all_labels = [], [], []

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for phys, img, name in tqdm(loader, desc="Encoding images"):
            img = img.to(device)
            z = encoder(img).cpu().numpy()
            all_latents.append(z)
            all_phys.append(phys.numpy())
            all_labels.extend(name)

    Z = np.concatenate(all_latents, axis=0)
    P = np.concatenate(all_phys, axis=0)
    features = np.concatenate([Z, P], axis=1)
    return features, all_labels

def plot_bar_each_variable(cluster_stats, save_dir, method):
    means = cluster_stats.xs("mean", axis=1, level=1)
    stds = cluster_stats.xs("std", axis=1, level=1)

    for col_idx, col_name in enumerate(means.columns):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, cluster_id in enumerate(means.index):
            mean_val = means.loc[cluster_id, col_name]
            std_val = stds.loc[cluster_id, col_name]

            lower_err = min(std_val, mean_val) if mean_val >= 0 else std_val
            lower_err = max(lower_err, 0) 
            upper_err = std_val 

            ax.bar(i, mean_val, yerr=[[lower_err], [upper_err]], capsize=4, label=f"Cluster {cluster_id}")

        ax.set_xticks(range(len(means.index)))
        ax.set_xticklabels([f"Cluster {i}" for i in means.index], fontsize=14)
        if col_name == "Ratio (L2/L1)":
            col_name = r"$L_2/L_1$"
        ax.set_ylabel(col_name)
        apply_heavy_axes(ax)  
        if col_name == r"$L_2/L_1$":
            col_name = "Ratio (L2/L1)"
        safe_name = col_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")
        fname = os.path.join(save_dir, f"bar_{safe_name}_{method}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close(fig)

def run_clustering_and_plot(features, df, dataset, method='kmeans', k=3, save_dir='cluster_result'):
    os.makedirs(save_dir, exist_ok=True)
    if method == 'gmm':
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(features)
    elif method == 'hdbscan':
        model = hdbscan.HDBSCAN(min_cluster_size=k)
        labels = model.fit_predict(features)
    else:
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(features)

    labels = remap_labels(labels, RENAME_MAP)

    df_out = df.copy()
    df_out["Cluster"] = labels
    df_out.to_csv(os.path.join(save_dir, f"{method}_result.csv"), index=False)
    liquid_cluster_table = df_out.groupby(['Liquid type', 'Cluster']).size().unstack(fill_value=0)
    liquid_cluster_table.to_csv(os.path.join(save_dir, f"{method}_liquid_cluster_table.csv"))

    cluster_stats = df_out.groupby("Cluster")[
        [
            "Nozzle inner diameter (mm)",
            "Flow rate (μl/min)",
            "Viscosity (mPa.s)",
            "Density (kg/m^3)",
            "Surface tension (mN/m)",
            "Ratio (L2/L1)",
            "Ratio (L2/D)",
            "Bo"
        ]
    ].agg(["mean", "std"])
    cluster_stats.to_csv(os.path.join(save_dir, f"{method}_cluster_stats.csv"))
    plot_bar_each_variable(cluster_stats, save_dir, method)

    print(f"[{method}] Generating PCA, Re-Oh, and Average Shape plots...")

    #img_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #all_images = []
    #for _, (phys, img, fname) in enumerate(img_loader):
    #    all_images.append(img.squeeze(0))  # [1, 1, H, W] -> [1, H, W]
    #all_images = torch.stack(all_images)  # [N, 1, H, W]

    #for cluster_id in np.unique(labels):
    #    indices = np.where(labels == cluster_id)[0]
    #    if len(indices) == 0:
    #        continue
    #    avg_img = all_images[indices].mean(dim=0).squeeze(0)  # [H, W]
    #    fig, ax = plt.subplots(figsize=(6, 5))
    #    ax.imshow(avg_img.numpy(), cmap='gray')
    #    ax.axis('off')  
    #    plt.tight_layout()
    #    plt.savefig(os.path.join(save_dir, f"mean_image_cluster{cluster_id}_{method}.png"), dpi=300)
    #    plt.close(fig)

    re = df_out["Reynolds Number"]
    oh = df_out["Oh"]
    cluster = df_out["Cluster"]
    color_map = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster_id in np.unique(cluster):
        mask = (cluster == cluster_id)
        ax.scatter(
            re[mask], oh[mask],
            c=[color_map(cluster_id % 10)],
            label=f"Cluster {cluster_id}",
            s=40, edgecolors='k', linewidths=0.3
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"$Oh$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc='upper right', fontsize=16)
    apply_heavy_axes(ax)  
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(save_dir, f"re_oh_{method}_cluster_only.png"), dpi=300)
    plt.close(fig)

    from matplotlib.ticker import FixedLocator, NullFormatter, LogFormatterMathtext


    ticks = [1000, 4000]
    re = df_out["Bo"]
    oh = df_out["Oh"]
    cluster = df_out["Cluster"]
    color_map = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.tick_params(width=2.0, length=6)

    for cluster_id in np.unique(cluster):
        mask = (cluster == cluster_id)
        ax.scatter(
            re[mask], oh[mask],
            c=[color_map(cluster_id % 10)],
            label=f"Cluster {cluster_id}",
            s=40, edgecolors='k', linewidths=0.3
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())  
    ax.xaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel(r"$Bo$")
    ax.set_ylabel(r"$Oh$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(loc='upper left', fontsize=16)
    apply_heavy_axes(ax)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(save_dir, f"bo_oh_{method}_cluster_only.png"), dpi=300)
    plt.close(fig)
    return df_out

def analyze_hdbscan_min_cluster_size(features, save_dir="./"):
    os.makedirs(save_dir, exist_ok=True)

    min_cluster_sizes = [15, 25, 35, 50, 75, 100]
    n_clusters_list = []
    silhouette_scores = []
    dbi_scores = []

    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs)
        labels = clusterer.fit_predict(features)

        n_clusters = len(set(labels))
        n_clusters_list.append(n_clusters)

        if n_clusters > 1:
            try:
                sil_score = silhouette_score(features, labels)
            except:
                sil_score = np.nan
            try:
                dbi_score = davies_bouldin_score(features, labels)
            except:
                dbi_score = np.nan
        else:
            sil_score = np.nan
            dbi_score = np.nan

        silhouette_scores.append(sil_score)
        dbi_scores.append(dbi_score)

        print(f"min_cluster_size={mcs}: n_clusters={n_clusters}, Silhouette={sil_score:.4f}, DBI={dbi_score:.4f}")

    # Best min_cluster_size based on Silhouette
    best_idx = np.nanargmax(silhouette_scores)
    best_mcs = min_cluster_sizes[best_idx]
    print(f"Best min_cluster_size for HDBSCAN: {best_mcs}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("Number of clusters", color="tab:blue")
    ax1.plot(min_cluster_sizes, n_clusters_list, marker="o", color="tab:blue", label="Num Clusters")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    apply_heavy_axes(ax1)  

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:red")
    ax2.plot(min_cluster_sizes, silhouette_scores, marker="s", color="tab:red", label="Silhouette Score")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    apply_heavy_axes(ax2) 

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel("Davies-Bouldin Index", color="tab:green")
    ax3.plot(min_cluster_sizes, dbi_scores, marker="^", color="tab:green", label="DBI")
    ax3.tick_params(axis='y', labelcolor="tab:green")
    apply_heavy_axes(ax3)  

    plt.title("HDBSCAN: Num Clusters, Silhouette & DBI vs. min_cluster_size")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "hdbscan_min_cluster_size_analysis.png"), dpi=300)
    plt.close(fig)

    return best_mcs

def plot_silhouette_scores(features, save_dir="./"):
    k_range = range(2, 8)
    kmeans_scores, gmm_scores = [], []
    kmeans_dbi, gmm_dbi = [], []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        kmeans_labels = kmeans.labels_
        kmeans_scores.append(silhouette_score(features, kmeans_labels))
        kmeans_dbi.append(davies_bouldin_score(features, kmeans_labels))

        gmm = GaussianMixture(n_components=k, random_state=0).fit(features)
        gmm_labels = gmm.predict(features)
        gmm_scores.append(silhouette_score(features, gmm_labels))
        gmm_dbi.append(davies_bouldin_score(features, gmm_labels))

        print(f"K={k}: KMeans Silhouette={kmeans_scores[-1]:.4f}, DBI={kmeans_dbi[-1]:.4f}; "
              f"GMM Silhouette={gmm_scores[-1]:.4f}, DBI={gmm_dbi[-1]:.4f}")

    best_k_kmeans = k_range[np.argmax(kmeans_scores)]
    best_k_gmm = k_range[np.argmax(gmm_scores)]
    print(f"Best K for KMeans: {best_k_kmeans}, Best K for GMM: {best_k_gmm}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k_range, kmeans_scores, marker='o', label='KMeans')
    ax.plot(k_range, gmm_scores, marker='s', label='GMM')
    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Silhouette Score')
    ax.legend()
    ax.grid(True)
    apply_heavy_axes(ax)  
    plt.tight_layout()
    plt.savefig(f"{save_dir}/silhouette_scores.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(k_range, kmeans_dbi, marker='o', label='KMeans DBI')
    ax.plot(k_range, gmm_dbi, marker='s', label='GMM DBI')
    ax.set_xlabel(r'$K$')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.grid(True)
    apply_heavy_axes(ax) 
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dbi_scores.png", dpi=300)
    plt.close(fig)

    return best_k_kmeans, best_k_gmm

if __name__ == "__main__":
    excel_path = "data.xlsx"
    encoder_path = "es_best_encoder.pth"
    features_file = "features.npy"
    df_info_file = "df_info.pkl"

    if os.path.exists(features_file) and os.path.exists(df_info_file):
        print("Loading existing features and df_info...")
        features = np.load(features_file)
        with open(df_info_file, "rb") as f:
            df_info = pickle.load(f)
        dataset = ImageWithPhysicalDataset(excel_path)
    else:
        dataset = ImageWithPhysicalDataset(excel_path)
        encoder = Encoder().to(DEVICE)
        encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
        encoder.eval()
        features, filenames = encode_all_images_with_phys(encoder, dataset, device=DEVICE)
        df_info = dataset.df.copy()

        np.save(features_file, features)
        with open(df_info_file, "wb") as f:
            pickle.dump(df_info, f)
        print(f"Saved features to {features_file} and df_info to {df_info_file}")

    best_k_kmeans, best_k_gmm = plot_silhouette_scores(features, save_dir="./")
    best_mcs_hdbscan = analyze_hdbscan_min_cluster_size(features, save_dir="./")
    best_k_kmeans, best_k_gmm = 5, 5
    best_mcs_hdbscan = 50

    run_clustering_and_plot(features, df_info, dataset, method="kmeans", k=best_k_kmeans, save_dir=f"cluster_result_kmeans_k{best_k_kmeans}")
    run_clustering_and_plot(features, df_info, dataset, method="gmm", k=best_k_gmm, save_dir=f"cluster_result_gmm_k{best_k_gmm}")
    run_clustering_and_plot(features, df_info, dataset, method="hdbscan", k=best_mcs_hdbscan, save_dir=f"cluster_result_hdbscan_mcs{best_mcs_hdbscan}")
