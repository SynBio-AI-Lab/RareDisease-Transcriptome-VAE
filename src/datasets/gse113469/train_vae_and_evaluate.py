"""
GSE113469 VAE Training and Evaluation
=====================================
GSE113469 데이터로 VAE 모델을 학습하고,
합성 데이터를 생성하여 품질을 평가합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# =============================================================================
# 설정
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(message)s')

# 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "processed", "GSE113469_scaled.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# 하이퍼파라미터 (GSE134900과 동일)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 500
HIDDEN_DIM = 256
LATENT_DIM = 64
KL_WEIGHT = 0.0005
TEMPERATURE = 2.0  # Sampling Temperature
SYNTHETIC_SAMPLES = 1000
EARLY_STOP_PATIENCE = 30
SEED = 42

# 재현성
torch.manual_seed(SEED)
np.random.seed(SEED)

# 디바이스
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

# 시각화 설정
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =============================================================================
# VAE 모델 정의 (GSE134900과 동일 구조)
# =============================================================================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, kl_weight):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss


# =============================================================================
# 메인 실행
# =============================================================================
def main():
    print("=" * 70)
    print("GSE113469 VAE Training and Synthetic Data Generation")
    print("=" * 70)
    
    # =========================================================================
    # [1/5] 데이터 로드
    # =========================================================================
    print("\n[1/5] Loading Data...")
    
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"      Data shape: {df.shape} (samples x genes)")
    print(f"      Samples: {df.shape[0]}, Features: {df.shape[1]}")
    
    # 텐서 변환
    data = torch.FloatTensor(df.values).to(device)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    INPUT_DIM = df.shape[1]
    print(f"      Input dimension: {INPUT_DIM}")
    
    # =========================================================================
    # [2/5] VAE 모델 학습
    # =========================================================================
    print("\n[2/5] Training VAE Model...")
    print(f"      Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
    print(f"      Hidden: {HIDDEN_DIM}, Latent: {LATENT_DIM}")
    print(f"      Temperature: {TEMPERATURE}")
    
    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=20)
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon, x, mu, logvar, KL_WEIGHT)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_vae_model.pt"))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"      Early stopping at epoch {epoch+1}")
            break
    
    # 최적 모델 로드
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_vae_model.pt"), 
                                      weights_only=True))
    print(f"      Training complete! Best loss: {best_loss:.4f}")
    
    # =========================================================================
    # [3/5] 합성 데이터 생성
    # =========================================================================
    print(f"\n[3/5] Generating {SYNTHETIC_SAMPLES} Synthetic Samples...")
    print(f"      Sampling Temperature: {TEMPERATURE}")
    
    model.eval()
    with torch.no_grad():
        z = torch.randn(SYNTHETIC_SAMPLES, LATENT_DIM).to(device) * TEMPERATURE
        synthetic_data = model.decode(z).cpu().numpy()
    
    # DataFrame 저장
    df_synthetic = pd.DataFrame(synthetic_data, columns=df.columns)
    syn_path = os.path.join(RESULTS_DIR, "synthetic_GSE113469.csv")
    df_synthetic.to_csv(syn_path, index=False)
    print(f"      Saved: {syn_path}")
    
    # =========================================================================
    # [4/5] 통계적 평가
    # =========================================================================
    print("\n[4/5] Statistical Evaluation...")
    
    real_data = df.values
    
    # Mean & Std 비교
    real_mean = real_data.mean(axis=0)
    syn_mean = synthetic_data.mean(axis=0)
    real_std = real_data.std(axis=0)
    syn_std = synthetic_data.std(axis=0)
    
    # R² 계산
    slope_mean, intercept_mean, r_mean, _, _ = stats.linregress(real_mean, syn_mean)
    slope_std, intercept_std, r_std, _, _ = stats.linregress(real_std, syn_std)
    
    r2_mean = r_mean ** 2
    r2_std = r_std ** 2
    
    print(f"\n      === Statistical Fidelity ===")
    print(f"      Mean R²: {r2_mean:.4f}")
    print(f"      Std R²:  {r2_std:.4f}")
    
    # Diversity 계산 (Pairwise Distance)
    from sklearn.metrics import pairwise_distances
    sample_size = min(100, len(synthetic_data))
    sample_idx = np.random.choice(len(synthetic_data), sample_size, replace=False)
    syn_sample = synthetic_data[sample_idx]
    real_sample = real_data[np.random.choice(len(real_data), min(100, len(real_data)), replace=False)]
    
    syn_diversity = pairwise_distances(syn_sample).mean()
    real_diversity = pairwise_distances(real_sample).mean()
    diversity_ratio = syn_diversity / real_diversity
    
    print(f"      Diversity Ratio: {diversity_ratio:.4f} (Synthetic/Real)")
    
    # 결과 저장
    results_summary = {
        'Dataset': 'GSE113469',
        'Samples_Real': len(real_data),
        'Samples_Synthetic': SYNTHETIC_SAMPLES,
        'Features': INPUT_DIM,
        'Mean_R2': r2_mean,
        'Std_R2': r2_std,
        'Diversity_Real': real_diversity,
        'Diversity_Synthetic': syn_diversity,
        'Diversity_Ratio': diversity_ratio,
        'Temperature': TEMPERATURE
    }
    
    # =========================================================================
    # [5/5] 시각화
    # =========================================================================
    print("\n[5/5] Creating Visualizations...")
    
    # ----- 1. Mean/Std Scatter Plots -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean
    axes[0].scatter(real_mean, syn_mean, alpha=0.5, s=10, c='steelblue')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
    axes[0].set_xlabel('Real Data Mean')
    axes[0].set_ylabel('Synthetic Data Mean')
    axes[0].set_title(f'Mean Comparison (R² = {r2_mean:.4f})')
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Std
    axes[1].scatter(real_std, syn_std, alpha=0.5, s=10, c='darkorange')
    axes[1].plot([0, real_std.max()], [0, real_std.max()], 'r--', linewidth=2, label='y=x')
    axes[1].set_xlabel('Real Data Std')
    axes[1].set_ylabel('Synthetic Data Std')
    axes[1].set_title(f'Std Comparison (R² = {r2_std:.4f})')
    axes[1].legend()
    
    plt.suptitle(f'GSE113469: Statistical Fidelity (T={TEMPERATURE})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'statistical_fidelity_{TEMPERATURE}.png'))
    plt.close()
    print(f"      Saved: statistical_fidelity_{TEMPERATURE}.png")
    
    # ----- 2. PCA Visualization -----
    combined = np.vstack([real_data, synthetic_data])
    labels = ['Real'] * len(real_data) + ['Synthetic'] * len(synthetic_data)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'Real': '#2E86AB', 'Synthetic': '#E94F37'}
    
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels) == label
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                   c=colors[label], label=label, alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'GSE113469: PCA - Real vs Synthetic (T={TEMPERATURE})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'pca_comparison_{TEMPERATURE}.png'))
    plt.close()
    print(f"      Saved: pca_comparison_{TEMPERATURE}.png")
    
    # ----- 3. t-SNE Visualization -----
    print("      Computing t-SNE (this may take a moment)...")
    
    # 샘플 수 제한 (t-SNE 속도)
    n_real = min(len(real_data), 100)
    n_syn = min(len(synthetic_data), 300)
    
    real_sample_tsne = real_data[np.random.choice(len(real_data), n_real, replace=False)]
    syn_sample_tsne = synthetic_data[np.random.choice(len(synthetic_data), n_syn, replace=False)]
    combined_tsne = np.vstack([real_sample_tsne, syn_sample_tsne])
    labels_tsne = ['Real'] * n_real + ['Synthetic'] * n_syn
    
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
    tsne_result = tsne.fit_transform(combined_tsne)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label in ['Real', 'Synthetic']:
        mask = np.array(labels_tsne) == label
        ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                   c=colors[label], label=label, alpha=0.6, s=50)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f'GSE113469: t-SNE - Real vs Synthetic (T={TEMPERATURE})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'tsne_comparison_{TEMPERATURE}.png'))
    plt.close()
    print(f"      Saved: tsne_comparison_{TEMPERATURE}.png")
    
    # ----- 4. Training Loss Curve -----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'VAE Training Loss (T={TEMPERATURE})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'training_loss_{TEMPERATURE}.png'))
    plt.close()
    print(f"      Saved: training_loss_{TEMPERATURE}.png")
    
    # =========================================================================
    # 최종 요약
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"""
    Dataset: GSE113469 (Celiac Disease PBMC)
    
    VAE Configuration:
      - Hidden Dim: {HIDDEN_DIM}
      - Latent Dim: {LATENT_DIM}
      - Temperature: {TEMPERATURE}
      - Training Epochs: {len(train_losses)}
    
    Data:
      - Real Samples: {len(real_data)}
      - Synthetic Samples: {SYNTHETIC_SAMPLES}
      - Features: {INPUT_DIM}
    
    Statistical Fidelity:
      - Mean R²: {r2_mean:.4f}
      - Std R²:  {r2_std:.4f}
    
    Diversity:
      - Real: {real_diversity:.4f}
      - Synthetic: {syn_diversity:.4f}
      - Ratio: {diversity_ratio:.4f}
    
    Output Files:
      - Model: {RESULTS_DIR}/best_vae_model.pt
      - Synthetic Data: {syn_path}
      - Figures: {FIGURES_DIR}/
    """)
    
    # GSE134900 비교 참조값 (이전 실행 결과 기준)
    print("=" * 70)
    print("COMPARISON WITH GSE134900")
    print("=" * 70)
    print("""
    Metric          | GSE134900    | GSE113469    | Difference
    ----------------|--------------|--------------|------------
    Mean R²         | ~0.99        | {:.4f}       | {}
    Std R²          | ~0.95        | {:.4f}       | {}
    Diversity Ratio | ~1.5-2.0     | {:.4f}       | {}
    
    Note: GSE134900 values are approximate from previous runs.
    """.format(
        r2_mean, "Similar" if r2_mean > 0.95 else "Lower",
        r2_std, "Similar" if r2_std > 0.90 else "Lower",
        diversity_ratio, "Similar" if 1.0 < diversity_ratio < 2.5 else "Different"
    ))
    print("=" * 70)
    
    return results_summary


if __name__ == "__main__":
    results = main()
