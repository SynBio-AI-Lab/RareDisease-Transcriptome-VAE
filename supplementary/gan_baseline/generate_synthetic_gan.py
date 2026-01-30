"""
GAN-based Synthetic Data Generation for Celiac Disease
=======================================================
Uses the same preprocessing pipeline as the VAE model for fair comparison.
Generates synthetic gene expression data using Generative Adversarial Network.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging
import os
import sys

# =============================================================================
# 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # 상위 폴더 (2026 AI Challenge)
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "synthetic")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# =============================================================================
# 하이퍼파라미터 (VAE와 유사하게 설정)
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE_G = 2e-4      # Generator learning rate
LEARNING_RATE_D = 2e-4      # Discriminator learning rate
EPOCHS = 2000               # GAN은 더 많은 epoch 필요
HIDDEN_DIM = 512
LATENT_DIM = 64             # VAE와 동일
SEED = 42
SYNTHETIC_SAMPLES = 1000
TEMPERATURE = 2.0           # VAE와 동일한 sampling 범위

# GAN 특화 하이퍼파라미터
D_STEPS = 1                 # Discriminator 업데이트 횟수 per G step
LABEL_SMOOTHING = 0.1       # Label smoothing for stability
GRADIENT_PENALTY = 10.0     # WGAN-GP gradient penalty (optional)

torch.manual_seed(SEED)
np.random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# 데이터 로드 (VAE와 동일한 전처리)
# =============================================================================
if not os.path.exists(INPUT_FILE):
    logging.error(f"File not found: {INPUT_FILE}")
    sys.exit(1)

logging.info("Loading data (same preprocessing as VAE)...")
df = pd.read_csv(INPUT_FILE, index_col=0, compression='gzip')

# VAE와 동일: 행이 더 많고 1000개 초과면 전치
if df.shape[0] > df.shape[1] and df.shape[0] > 1000:
    df = df.T

# 중복 컬럼 제거
df = df.loc[:, ~df.columns.duplicated()]

# MinMaxScaler (VAE와 동일)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
input_dim = df.shape[1]

logging.info(f"Data shape: {df.shape[0]} samples x {df.shape[1]} genes")

# DataLoader 생성
train_tensor = torch.FloatTensor(data_scaled).to(device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# Generator 모델
# =============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM // 2),
            nn.BatchNorm1d(HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.BatchNorm1d(HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            
            nn.Linear(HIDDEN_DIM, input_dim),
            nn.Sigmoid()  # [0, 1] 범위 (MinMaxScaler 사용)
        )
    
    def forward(self, z):
        return self.model(z)

# =============================================================================
# Discriminator 모델
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 4),
            nn.LeakyReLU(0.2),
            
            nn.Linear(HIDDEN_DIM // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# =============================================================================
# 모델 초기화
# =============================================================================
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Weight initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

generator.apply(weights_init)
discriminator.apply(weights_init)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# =============================================================================
# 학습
# =============================================================================
logging.info(f"Starting GAN training on {device}...")
logging.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
logging.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

# Loss history
g_losses = []
d_losses = []
d_real_acc = []
d_fake_acc = []

for epoch in range(EPOCHS):
    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_d_real = 0
    epoch_d_fake = 0
    n_batches = 0
    
    for real_data, in train_loader:
        batch_size = real_data.size(0)
        
        # Labels with smoothing
        real_labels = torch.ones(batch_size, 1).to(device) * (1 - LABEL_SMOOTHING)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # =========================
        # Train Discriminator
        # =========================
        for _ in range(D_STEPS):
            optimizer_D.zero_grad()
            
            # Real data
            d_real_output = discriminator(real_data)
            d_real_loss = criterion(d_real_output, real_labels)
            
            # Fake data
            z = torch.randn(batch_size, LATENT_DIM).to(device) * TEMPERATURE
            fake_data = generator(z)
            d_fake_output = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake_output, fake_labels)
            
            # Total D loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()
        
        # =========================
        # Train Generator
        # =========================
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, LATENT_DIM).to(device) * TEMPERATURE
        fake_data = generator(z)
        g_output = discriminator(fake_data)
        
        # Generator wants D to output 1 (real)
        g_loss = criterion(g_output, torch.ones(batch_size, 1).to(device))
        g_loss.backward()
        optimizer_G.step()
        
        # Accumulate metrics
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_d_real += d_real_output.mean().item()
        epoch_d_fake += d_fake_output.mean().item()
        n_batches += 1
    
    # Average losses
    avg_g_loss = epoch_g_loss / n_batches
    avg_d_loss = epoch_d_loss / n_batches
    avg_d_real = epoch_d_real / n_batches
    avg_d_fake = epoch_d_fake / n_batches
    
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    d_real_acc.append(avg_d_real)
    d_fake_acc.append(avg_d_fake)
    
    # Logging
    if (epoch + 1) % 100 == 0:
        logging.info(
            f'Epoch {epoch+1:4d}/{EPOCHS} | '
            f'G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f} | '
            f'D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f}'
        )

logging.info("Training complete!")

# =============================================================================
# Loss Curve 저장
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1 = axes[0]
ax1.plot(g_losses, label='Generator Loss', alpha=0.8)
ax1.plot(d_losses, label='Discriminator Loss', alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('GAN Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# D accuracy
ax2 = axes[1]
ax2.plot(d_real_acc, label='D(real)', alpha=0.8)
ax2.plot(d_fake_acc, label='D(fake)', alpha=0.8)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Discriminator Output')
ax2.set_title('Discriminator Predictions')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
loss_path = os.path.join(FIG_DIR, "gan_training_loss.png")
plt.savefig(loss_path, dpi=300)
plt.close()
logging.info(f"Loss curve saved: {loss_path}")

# =============================================================================
# 모델 저장
# =============================================================================
model_path = os.path.join(MODEL_DIR, "gan_celiac.pt")
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'input_dim': input_dim,
    'latent_dim': LATENT_DIM,
    'scaler': scaler,
    'columns': df.columns.tolist(),
    'temperature': TEMPERATURE
}, model_path)
logging.info(f"Model saved: {model_path}")

# =============================================================================
# 합성 데이터 생성
# =============================================================================
logging.info(f"Generating {SYNTHETIC_SAMPLES} synthetic samples (Temperature={TEMPERATURE})...")

generator.eval()
with torch.no_grad():
    z = torch.randn(SYNTHETIC_SAMPLES, LATENT_DIM).to(device) * TEMPERATURE
    syn_scaled = generator(z).cpu().numpy()
    syn_data = scaler.inverse_transform(syn_scaled)

output_path = os.path.join(OUTPUT_DIR, "synthetic_celiac_gan.csv")
pd.DataFrame(syn_data, columns=df.columns).to_csv(output_path, index=False)
logging.info(f"Synthetic data saved: {output_path}")

# =============================================================================
# PCA 시각화 (Real vs GAN Synthetic)
# =============================================================================
logging.info("Creating PCA visualization...")

# Standardize for PCA
scaler_pca = StandardScaler()
X_real = scaler_pca.fit_transform(df.values)
X_syn = scaler_pca.transform(syn_data)

# PCA
X_combined = np.vstack([X_real, X_syn])
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_combined)

# Split back
X_real_pca = X_pca[:len(df)]
X_syn_pca = X_pca[len(df):]

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c='#3498db', label='Real', alpha=0.7, s=50, edgecolors='white')
plt.scatter(X_syn_pca[:, 0], X_syn_pca[:, 1], c='#e74c3c', label='GAN Synthetic', alpha=0.5, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title(f'PCA: Real vs GAN Synthetic Data\n(Temperature={TEMPERATURE}, {SYNTHETIC_SAMPLES} samples)')
plt.legend()
plt.grid(True, alpha=0.3)

pca_path = os.path.join(FIG_DIR, "pca_gan_validation.png")
plt.savefig(pca_path, dpi=300, bbox_inches='tight')
plt.close()
logging.info(f"PCA plot saved: {pca_path}")

# =============================================================================
# 요약 통계
# =============================================================================
print("\n" + "=" * 70)
print("GAN SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 70)
print(f"""
Model Configuration:
  - Latent Dimension: {LATENT_DIM}
  - Hidden Dimension: {HIDDEN_DIM}
  - Temperature: {TEMPERATURE}
  - Training Epochs: {EPOCHS}

Generated Data:
  - Samples: {SYNTHETIC_SAMPLES}
  - Features (Genes): {input_dim}

Output Files:
  - Synthetic Data: {output_path}
  - Model: {model_path}
  - Loss Curve: {loss_path}
  - PCA Plot: {pca_path}
""")
print("=" * 70)

logging.info("Done!")
