"""
Vanilla GAN for Celiac Disease Gene Expression Data
====================================================
Mode Collapse 현상 탐지 및 시각화 포함
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import sys

# =============================================================================
# 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "vanilla_gan_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 하이퍼파라미터
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
EPOCHS = 3000
LATENT_DIM = 64
HIDDEN_DIM = 256
SEED = 42
SYNTHETIC_SAMPLES = 1000

# Mode Collapse 탐지용 체크포인트
CHECKPOINT_EPOCHS = [100, 500, 1000, 1500, 2000, 2500, 3000]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =============================================================================
# 데이터 로드 (VAE와 동일한 전처리)
# =============================================================================
print("\n[1/5] Loading data...")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE, index_col=0, compression='gzip')
if df.shape[0] > df.shape[1] and df.shape[0] > 1000:
    df = df.T
df = df.loc[:, ~df.columns.duplicated()]

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
input_dim = df.shape[1]

print(f"   Data: {df.shape[0]} samples x {df.shape[1]} genes")

train_tensor = torch.FloatTensor(data_scaled).to(device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)

# =============================================================================
# Vanilla GAN - Generator
# =============================================================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM * 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        return self.model(z)

# =============================================================================
# Vanilla GAN - Discriminator
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# =============================================================================
# Mode Collapse 탐지 함수
# =============================================================================
def calculate_diversity_metrics(generated_samples):
    """생성 샘플의 다양성 측정"""
    # 1. 평균 쌍별 거리 (Average Pairwise Distance)
    if len(generated_samples) > 500:
        idx = np.random.choice(len(generated_samples), 500, replace=False)
        samples = generated_samples[idx]
    else:
        samples = generated_samples
    
    distances = pairwise_distances(samples, metric='euclidean')
    avg_distance = np.mean(distances[np.triu_indices(len(samples), k=1)])
    
    # 2. 표준편차 (각 feature의 평균 std)
    avg_std = np.mean(np.std(generated_samples, axis=0))
    
    # 3. 고유 샘플 비율 (중복 체크)
    rounded = np.round(generated_samples, decimals=3)
    unique_ratio = len(np.unique(rounded, axis=0)) / len(generated_samples)
    
    return {
        'avg_pairwise_distance': avg_distance,
        'avg_std': avg_std,
        'unique_ratio': unique_ratio
    }

def generate_samples(generator, n_samples):
    """Generator로 샘플 생성"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, LATENT_DIM).to(device)
        samples = generator(z).cpu().numpy()
    generator.train()
    return samples

# =============================================================================
# 모델 초기화
# =============================================================================
print("\n[2/5] Initializing Vanilla GAN...")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
criterion = nn.BCELoss()

print(f"   Generator params: {sum(p.numel() for p in generator.parameters()):,}")
print(f"   Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

# =============================================================================
# 학습
# =============================================================================
print(f"\n[3/5] Training for {EPOCHS} epochs...")

# 기록용 리스트
g_losses = []
d_losses = []
d_real_outputs = []
d_fake_outputs = []
diversity_history = []
checkpoint_samples = {}

# Real data의 다양성 (기준)
real_diversity = calculate_diversity_metrics(data_scaled)
print(f"\n   Real data diversity:")
print(f"     - Avg Pairwise Distance: {real_diversity['avg_pairwise_distance']:.4f}")
print(f"     - Avg Std: {real_diversity['avg_std']:.4f}")
print(f"     - Unique Ratio: {real_diversity['unique_ratio']:.2%}")
print()

for epoch in range(EPOCHS):
    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_d_real = 0
    epoch_d_fake = 0
    n_batches = 0
    
    for real_data, in train_loader:
        batch_size = real_data.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        
        # Real
        d_real = discriminator(real_data)
        d_loss_real = criterion(d_real, real_labels)
        
        # Fake
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_data = generator(z)
        d_fake = discriminator(fake_data.detach())
        d_loss_fake = criterion(d_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # --- Train Generator ---
        optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_data = generator(z)
        g_output = discriminator(fake_data)
        g_loss = criterion(g_output, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # 기록
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_d_real += d_real.mean().item()
        epoch_d_fake += d_fake.mean().item()
        n_batches += 1
    
    # 평균
    avg_g_loss = epoch_g_loss / n_batches
    avg_d_loss = epoch_d_loss / n_batches
    avg_d_real = epoch_d_real / n_batches
    avg_d_fake = epoch_d_fake / n_batches
    
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)
    d_real_outputs.append(avg_d_real)
    d_fake_outputs.append(avg_d_fake)
    
    # 체크포인트에서 다양성 측정
    if (epoch + 1) in CHECKPOINT_EPOCHS:
        samples = generate_samples(generator, 500)
        diversity = calculate_diversity_metrics(samples)
        diversity['epoch'] = epoch + 1
        diversity_history.append(diversity)
        checkpoint_samples[epoch + 1] = samples
        
        print(f"   Epoch {epoch+1:4d} | G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | "
              f"D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f} | "
              f"Diversity: {diversity['avg_pairwise_distance']:.4f}")
    elif (epoch + 1) % 500 == 0:
        print(f"   Epoch {epoch+1:4d} | G: {avg_g_loss:.4f} | D: {avg_d_loss:.4f} | "
              f"D(real): {avg_d_real:.3f} | D(fake): {avg_d_fake:.3f}")

print("\n   Training complete!")

# =============================================================================
# [4/5] 시각화
# =============================================================================
print("\n[4/5] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- 1. Loss Curves ---
ax1 = axes[0, 0]
ax1.plot(g_losses, label='Generator Loss', alpha=0.8, color='#e74c3c')
ax1.plot(d_losses, label='Discriminator Loss', alpha=0.8, color='#3498db')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- 2. Discriminator Output ---
ax2 = axes[0, 1]
ax2.plot(d_real_outputs, label='D(real) - should stay ~1', alpha=0.8, color='#27ae60')
ax2.plot(d_fake_outputs, label='D(fake) - should approach 0.5', alpha=0.8, color='#9b59b6')
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Ideal D(fake)')
ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Discriminator Output')
ax2.set_title('Discriminator Predictions Over Training')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.1, 1.1)

# --- 3. Diversity Over Time (Mode Collapse 탐지) ---
ax3 = axes[1, 0]
epochs_check = [d['epoch'] for d in diversity_history]
distances = [d['avg_pairwise_distance'] for d in diversity_history]
ax3.plot(epochs_check, distances, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Generated')
ax3.axhline(y=real_diversity['avg_pairwise_distance'], color='#3498db', linestyle='--', 
            linewidth=2, label=f"Real Data ({real_diversity['avg_pairwise_distance']:.4f})")
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Average Pairwise Distance')
ax3.set_title('Sample Diversity Over Training\n(Mode Collapse Detection)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Mode Collapse 경고
final_diversity = diversity_history[-1]['avg_pairwise_distance']
collapse_ratio = final_diversity / real_diversity['avg_pairwise_distance']
if collapse_ratio < 0.5:
    ax3.text(0.5, 0.1, 'WARNING: Severe Mode Collapse Detected!', 
             transform=ax3.transAxes, fontsize=12, color='red', 
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
elif collapse_ratio < 0.7:
    ax3.text(0.5, 0.1, 'CAUTION: Moderate Mode Collapse', 
             transform=ax3.transAxes, fontsize=11, color='orange', 
             ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- 4. PCA Comparison ---
ax4 = axes[1, 1]
final_samples = generate_samples(generator, SYNTHETIC_SAMPLES)

scaler_pca = StandardScaler()
X_real = scaler_pca.fit_transform(data_scaled)
X_fake = scaler_pca.transform(final_samples)

pca = PCA(n_components=2, random_state=SEED)
X_combined = np.vstack([X_real, X_fake])
X_pca = pca.fit_transform(X_combined)

X_real_pca = X_pca[:len(data_scaled)]
X_fake_pca = X_pca[len(data_scaled):]

ax4.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c='#3498db', label='Real', alpha=0.7, s=60, edgecolors='white')
ax4.scatter(X_fake_pca[:, 0], X_fake_pca[:, 1], c='#e74c3c', label='GAN Generated', alpha=0.4, s=30)
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax4.set_title('PCA: Real vs GAN Generated')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "vanilla_gan_analysis.png")
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Main figure saved: {fig_path}")

# --- 5. PCA Evolution (Mode Collapse 시각화) ---
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
axes2 = axes2.flatten()

for idx, epoch_num in enumerate(CHECKPOINT_EPOCHS[:6]):
    ax = axes2[idx]
    if epoch_num in checkpoint_samples:
        samples = checkpoint_samples[epoch_num]
        X_fake_check = scaler_pca.transform(samples)
        X_fake_pca_check = pca.transform(X_fake_check)
        
        ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c='#3498db', label='Real', alpha=0.5, s=40)
        ax.scatter(X_fake_pca_check[:, 0], X_fake_pca_check[:, 1], c='#e74c3c', label='Generated', alpha=0.4, s=20)
        
        div = diversity_history[idx]
        ax.set_title(f'Epoch {epoch_num}\nDiversity: {div["avg_pairwise_distance"]:.4f}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        if idx == 0:
            ax.legend()

plt.suptitle('Mode Collapse Detection: PCA Evolution Over Training', fontsize=14, y=1.02)
plt.tight_layout()
evolution_path = os.path.join(OUTPUT_DIR, "pca_evolution_mode_collapse.png")
plt.savefig(evolution_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Evolution figure saved: {evolution_path}")

# =============================================================================
# [5/5] 결과 저장 및 요약
# =============================================================================
print("\n[5/5] Saving results...")

# 합성 데이터 저장
syn_data = scaler.inverse_transform(final_samples)
output_path = os.path.join(OUTPUT_DIR, "synthetic_vanilla_gan.csv")
pd.DataFrame(syn_data, columns=df.columns).to_csv(output_path, index=False)
print(f"   Synthetic data saved: {output_path}")

# 모델 저장
model_path = os.path.join(OUTPUT_DIR, "vanilla_gan_model.pt")
torch.save({
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'scaler': scaler,
    'columns': df.columns.tolist()
}, model_path)
print(f"   Model saved: {model_path}")

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("VANILLA GAN TRAINING SUMMARY")
print("=" * 70)

print(f"""
Configuration:
  - Epochs: {EPOCHS}
  - Latent Dim: {LATENT_DIM}
  - Hidden Dim: {HIDDEN_DIM}
  - Learning Rate: {LEARNING_RATE}

Final Metrics:
  - Generator Loss: {g_losses[-1]:.4f}
  - Discriminator Loss: {d_losses[-1]:.4f}
  - D(real): {d_real_outputs[-1]:.4f}
  - D(fake): {d_fake_outputs[-1]:.4f}

Mode Collapse Analysis:
  - Real Data Diversity (Pairwise Dist): {real_diversity['avg_pairwise_distance']:.4f}
  - Generated Diversity (Pairwise Dist): {final_diversity:.4f}
  - Diversity Ratio: {collapse_ratio:.2%}
  - Status: {"SEVERE MODE COLLAPSE" if collapse_ratio < 0.5 else "MODERATE MODE COLLAPSE" if collapse_ratio < 0.7 else "ACCEPTABLE DIVERSITY" if collapse_ratio < 0.9 else "GOOD DIVERSITY"}

Output Files:
  - {fig_path}
  - {evolution_path}
  - {output_path}
  - {model_path}
""")
print("=" * 70)
