import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
import argparse
import os

from trainer import WAE_GAN
from data_module import CelebADataModule

def visualize(ckpt_path, num_samples=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {ckpt_path}...")
    
    model = WAE_GAN.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    config = dict(model.hparams)
    
    latent_dim = config.get('LATENT_DIM', 64) 
    
    dm = CelebADataModule(config)
    dm.setup()
    dataloader = dm.val_dataloader()

    print(f"Collecting {num_samples} samples...")
    encoded_samples = []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch.to(device)
            z = model.encoder(x)
            encoded_samples.append(z)
            count += x.size(0)
            if count >= num_samples:
                break
    
    z_enc = torch.cat(encoded_samples, dim=0)[:num_samples].cpu().numpy()
    z_prior = model.prior.sample(num_samples, device).cpu().numpy()

    print(">>> Generating Projections (Image 1)...")
    
    all_data = np.concatenate([z_enc, z_prior], axis=0)
    
    # (1) PCA
    pca = PCA(n_components=2)
    res_pca = pca.fit_transform(all_data)
    
    # (2) t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
    res_tsne = tsne.fit_transform(all_data)
    
    # (3) UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    res_umap = reducer.fit_transform(all_data)

    # Plotting
    fig1, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    def plot_scatter(ax, data, title):
        ax.scatter(data[num_samples:, 0], data[num_samples:, 1], c='red', alpha=0.15, s=10, label='Prior p(z)')
        ax.scatter(data[:num_samples, 0], data[:num_samples, 1], c='blue', alpha=0.15, s=10, label='Encoded q(z|x)')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    plot_scatter(axes[0], res_pca, f"PCA (Explained Var: {pca.explained_variance_ratio_.sum():.2f})")
    plot_scatter(axes[1], res_tsne, "t-SNE")
    plot_scatter(axes[2], res_umap, "UMAP")
    
    plt.tight_layout()
    plt.savefig("latent_projections.png", dpi=300)
    plt.close()
    print("Saved 'latent_projections.png'")

    print(f">>> Generating Dimension-wise Distributions for {latent_dim} dims...")

    cols = 4
    rows = 4
    dims_per_page = cols * rows
    
    num_pages = int(np.ceil(latent_dim / dims_per_page))
    
    bins = 50

    global_min = min(z_enc.min(), z_prior.min())
    global_max = max(z_enc.max(), z_prior.max())
    range_lim = (global_min - 0.5, global_max + 0.5)

    for page in range(num_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16)) 
        axes = axes.flatten()
        
        start_idx = page * dims_per_page
        end_idx = min(start_idx + dims_per_page, latent_dim)
        
        print(f"  - Processing Page {page+1}/{num_pages} (Dims {start_idx} ~ {end_idx-1})")
        
        for i in range(dims_per_page):
            dim_idx = start_idx + i
            ax = axes[i]
            
            if dim_idx < latent_dim:
                enc_data = z_enc[:, dim_idx]
                prior_data = z_prior[:, dim_idx]

                ax.hist(enc_data, bins=bins, range=range_lim, density=True, 
                        color='blue', alpha=0.4, label='Posterior q(z|x)')
                
                ax.hist(prior_data, bins=bins, range=range_lim, density=True, 
                        color='red', alpha=0.4, label='Prior p(z)')
                
                enc_mu, enc_std = enc_data.mean(), enc_data.std()
                title_text = f"Dim {dim_idx}\n" \
                             f"μ={enc_mu:.2f}, σ={enc_std:.2f}"
                ax.set_title(title_text, fontsize=12, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=10)
            else:
                ax.axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=16)

        plt.tight_layout()
        
        save_name = f"latent_dist_page_{page+1:02d}.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Saved {num_pages} images (latent_dist_page_XX.png)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    visualize(args.ckpt, args.samples)