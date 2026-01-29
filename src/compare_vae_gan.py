"""
Final Comparison: Real vs VAE vs GAN Synthetic Data
====================================================
논문 투고용 시각화 및 통계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(BASE_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
VAE_PATH = os.path.join(BASE_DIR, "data", "synthetic", "synthetic_celiac_data.csv")
GAN_PATH = os.path.join(BASE_DIR, "add GAN", "vanilla_gan_output", "synthetic_vanilla_gan.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 논문용 스타일 설정 (Publication Quality)
# =============================================================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
    'axes.grid': False,
})

SEED = 42
np.random.seed(SEED)

# 색상 팔레트 (논문용)
COLORS = {
    'Real': '#2E86AB',       # 파란색
    'VAE': '#A23B72',        # 자주색
    'GAN': '#F18F01',        # 주황색
    'GAN_early': '#C73E1D'   # 빨간색 (초기 Mode Collapse)
}

# =============================================================================
# 데이터 로드
# =============================================================================
print("=" * 70)
print("FINAL COMPARISON: Real vs VAE vs GAN")
print("=" * 70)
print("\n[1/5] Loading data...")

# Real Data
df_real = pd.read_csv(REAL_PATH, index_col=0, compression='gzip')
if df_real.shape[0] > df_real.shape[1] and df_real.shape[0] > 1000:
    df_real = df_real.T
df_real = df_real.loc[:, ~df_real.columns.duplicated()]

# VAE Synthetic Data
df_vae = pd.read_csv(VAE_PATH)
if 'Unnamed: 0' in df_vae.columns:
    df_vae = df_vae.drop('Unnamed: 0', axis=1)

# GAN Synthetic Data
df_gan = pd.read_csv(GAN_PATH)
if 'Unnamed: 0' in df_gan.columns:
    df_gan = df_gan.drop('Unnamed: 0', axis=1)

# 공통 유전자 정렬
common_genes = df_real.columns.intersection(df_vae.columns).intersection(df_gan.columns)
df_real = df_real[common_genes]
df_vae = df_vae[common_genes]
df_gan = df_gan[common_genes]

print(f"   Real Data: {df_real.shape[0]} samples x {df_real.shape[1]} genes")
print(f"   VAE Synthetic: {df_vae.shape[0]} samples")
print(f"   GAN Synthetic: {df_gan.shape[0]} samples")
print(f"   Common Genes: {len(common_genes)}")

# =============================================================================
# 데이터 스케일 보정 (GAN 데이터 보정)
# =============================================================================
print("\n[2/5] Scaling data...")

# 표준화 (시각화용)
scaler = StandardScaler()
X_real = scaler.fit_transform(df_real.values)
X_vae = scaler.transform(df_vae.values)
X_gan = scaler.transform(df_gan.values)

print("   All datasets scaled to common scale")

# =============================================================================
# 통계적 지표 계산
# =============================================================================
print("\n[3/5] Computing statistical metrics...")

def calculate_metrics(real_data, syn_data, name):
    """원본 대비 합성 데이터의 통계적 지표 계산"""
    
    # 1. 평균/표준편차 상관계수 및 R²
    real_mean = np.mean(real_data, axis=0)
    syn_mean = np.mean(syn_data, axis=0)
    real_std = np.std(real_data, axis=0)
    syn_std = np.std(syn_data, axis=0)
    
    corr_mean, _ = stats.pearsonr(real_mean, syn_mean)
    corr_std, _ = stats.pearsonr(real_std, syn_std)
    r2_mean = corr_mean ** 2
    r2_std = corr_std ** 2
    
    # 2. 데이터 다양성 (Pairwise Distance)
    if len(syn_data) > 500:
        idx = np.random.choice(len(syn_data), 500, replace=False)
        samples = syn_data[idx]
    else:
        samples = syn_data
    
    distances = pairwise_distances(samples, metric='euclidean')
    diversity = np.mean(distances[np.triu_indices(len(samples), k=1)])
    
    # Real data diversity (기준)
    if len(real_data) > 100:
        idx_r = np.random.choice(len(real_data), min(100, len(real_data)), replace=False)
        real_samples = real_data[idx_r]
    else:
        real_samples = real_data
    real_distances = pairwise_distances(real_samples, metric='euclidean')
    real_diversity = np.mean(real_distances[np.triu_indices(len(real_samples), k=1)])
    
    diversity_ratio = diversity / real_diversity if real_diversity > 0 else 0
    
    return {
        'Model': name,
        'Mean_Corr': corr_mean,
        'Mean_R2': r2_mean,
        'Std_Corr': corr_std,
        'Std_R2': r2_std,
        'Diversity': diversity,
        'Real_Diversity': real_diversity,
        'Diversity_Ratio': diversity_ratio
    }

# 각 모델별 메트릭 계산
metrics_vae = calculate_metrics(df_real.values, df_vae.values, 'VAE')
metrics_gan = calculate_metrics(df_real.values, df_gan.values, 'GAN')

# Real 데이터 자체의 다양성 (기준)
real_self_diversity = metrics_vae['Real_Diversity']

print(f"\n   {'Model':<8} {'Mean R²':<12} {'Std R²':<12} {'Diversity':<12} {'Div. Ratio':<12}")
print("   " + "-" * 56)
print(f"   {'VAE':<8} {metrics_vae['Mean_R2']:<12.4f} {metrics_vae['Std_R2']:<12.4f} "
      f"{metrics_vae['Diversity']:<12.2f} {metrics_vae['Diversity_Ratio']:<12.2%}")
print(f"   {'GAN':<8} {metrics_gan['Mean_R2']:<12.4f} {metrics_gan['Std_R2']:<12.4f} "
      f"{metrics_gan['Diversity']:<12.2f} {metrics_gan['Diversity_Ratio']:<12.2%}")
print(f"   {'Real':<8} {'(baseline)':<12} {'(baseline)':<12} {real_self_diversity:<12.2f} {'100.00%':<12}")

# =============================================================================
# PCA 시각화 (3 서브플롯)
# =============================================================================
print("\n[4/5] Creating PCA visualization...")

# PCA 계산
pca = PCA(n_components=2, random_state=SEED)
X_all = np.vstack([X_real, X_vae, X_gan])
X_pca_all = pca.fit_transform(X_all)

n_real = len(X_real)
n_vae = len(X_vae)
n_gan = len(X_gan)

X_real_pca = X_pca_all[:n_real]
X_vae_pca = X_pca_all[n_real:n_real+n_vae]
X_gan_pca = X_pca_all[n_real+n_vae:]

# Figure 1: PCA 비교 (3 서브플롯)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 공통 축 범위 계산
x_min, x_max = X_pca_all[:, 0].min() - 10, X_pca_all[:, 0].max() + 10
y_min, y_max = X_pca_all[:, 1].min() - 10, X_pca_all[:, 1].max() + 10

# Plot 1: Real Data
ax1 = axes[0]
ax1.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c=COLORS['Real'], 
            s=80, alpha=0.8, edgecolors='white', linewidth=0.5, label='Real')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax1.set_title('(A) Real Data', fontweight='bold')
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.text(0.05, 0.95, f'n = {n_real}', transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: VAE Synthetic
ax2 = axes[1]
ax2.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c=COLORS['Real'], 
            s=60, alpha=0.3, edgecolors='none', label='Real')
ax2.scatter(X_vae_pca[:, 0], X_vae_pca[:, 1], c=COLORS['VAE'], 
            s=30, alpha=0.6, edgecolors='none', label='VAE')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax2.set_title('(B) VAE Synthetic', fontweight='bold')
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.legend(loc='upper right')
ax2.text(0.05, 0.95, f'R² Mean = {metrics_vae["Mean_R2"]:.3f}\nDiversity = {metrics_vae["Diversity_Ratio"]:.1%}', 
         transform=ax2.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: GAN Synthetic
ax3 = axes[2]
ax3.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c=COLORS['Real'], 
            s=60, alpha=0.3, edgecolors='none', label='Real')
ax3.scatter(X_gan_pca[:, 0], X_gan_pca[:, 1], c=COLORS['GAN'], 
            s=30, alpha=0.6, edgecolors='none', label='GAN')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax3.set_title('(C) GAN Synthetic', fontweight='bold')
ax3.set_xlim(x_min, x_max)
ax3.set_ylim(y_min, y_max)
ax3.legend(loc='upper right')
ax3.text(0.05, 0.95, f'R² Mean = {metrics_gan["Mean_R2"]:.3f}\nDiversity = {metrics_gan["Diversity_Ratio"]:.1%}', 
         transform=ax3.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('PCA Comparison: Real vs VAE vs GAN Synthetic Data', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
pca_path = os.path.join(OUTPUT_DIR, "Figure3_PCA_Comparison.png")
plt.savefig(pca_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   PCA comparison saved: {pca_path}")

# =============================================================================
# Figure 2: 통합 오버레이 비교
# =============================================================================
fig2, ax = plt.subplots(figsize=(12, 10))

ax.scatter(X_real_pca[:, 0], X_real_pca[:, 1], c=COLORS['Real'], 
           s=100, alpha=0.9, edgecolors='white', linewidth=1, label=f'Real (n={n_real})', zorder=3)
ax.scatter(X_vae_pca[:, 0], X_vae_pca[:, 1], c=COLORS['VAE'], 
           s=40, alpha=0.5, edgecolors='none', label=f'VAE (n={n_vae})', zorder=2)
ax.scatter(X_gan_pca[:, 0], X_gan_pca[:, 1], c=COLORS['GAN'], 
           s=40, alpha=0.5, edgecolors='none', label=f'GAN (n={n_gan})', zorder=1)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=16)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=16)
ax.set_title('PCA: Real vs VAE vs GAN Synthetic Data\n(Celiac Disease Gene Expression)', 
             fontsize=18, fontweight='bold')
ax.legend(loc='upper right', fontsize=14, framealpha=0.9)

plt.tight_layout()
overlay_path = os.path.join(OUTPUT_DIR, "Figure4_PCA_Overlay.png")
plt.savefig(overlay_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   PCA overlay saved: {overlay_path}")

# =============================================================================
# Figure 3: 통계 지표 비교 Bar Chart
# =============================================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart data
models = ['VAE', 'GAN']
colors_bar = [COLORS['VAE'], COLORS['GAN']]

# Mean R²
ax_r2 = axes3[0]
r2_values = [metrics_vae['Mean_R2'], metrics_gan['Mean_R2']]
bars1 = ax_r2.bar(models, r2_values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax_r2.set_ylabel('R² Score', fontsize=14)
ax_r2.set_title('(A) Mean Correlation (R²)', fontweight='bold')
ax_r2.set_ylim(0, 1.1)
ax_r2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars1, r2_values):
    ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Std R²
ax_std = axes3[1]
std_values = [metrics_vae['Std_R2'], metrics_gan['Std_R2']]
bars2 = ax_std.bar(models, std_values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax_std.set_ylabel('R² Score', fontsize=14)
ax_std.set_title('(B) Std Correlation (R²)', fontweight='bold')
ax_std.set_ylim(0, 1.1)
ax_std.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars2, std_values):
    ax_std.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
               f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Diversity Ratio
ax_div = axes3[2]
div_values = [metrics_vae['Diversity_Ratio'] * 100, metrics_gan['Diversity_Ratio'] * 100]
bars3 = ax_div.bar(models, div_values, color=colors_bar, edgecolor='black', linewidth=1.5)
ax_div.set_ylabel('Diversity Ratio (%)', fontsize=14)
ax_div.set_title('(C) Sample Diversity', fontweight='bold')
ax_div.set_ylim(0, 120)
ax_div.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Real Data (100%)')
for bar, val in zip(bars3, div_values):
    ax_div.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax_div.legend(loc='upper right')

plt.suptitle('Statistical Comparison: VAE vs GAN', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
stats_path = os.path.join(OUTPUT_DIR, "Figure5_Statistics_Comparison.png")
plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Statistics comparison saved: {stats_path}")

# =============================================================================
# Figure 4: Mean/Std Scatter Comparison
# =============================================================================
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))

real_mean = np.mean(df_real.values, axis=0)
real_std = np.std(df_real.values, axis=0)
vae_mean = np.mean(df_vae.values, axis=0)
vae_std = np.std(df_vae.values, axis=0)
gan_mean = np.mean(df_gan.values, axis=0)
gan_std = np.std(df_gan.values, axis=0)

# VAE Mean
ax_vm = axes4[0, 0]
ax_vm.scatter(real_mean, vae_mean, alpha=0.3, s=5, c=COLORS['VAE'])
lims = [min(real_mean.min(), vae_mean.min()), max(real_mean.max(), vae_mean.max())]
ax_vm.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax_vm.set_xlabel('Real Mean')
ax_vm.set_ylabel('VAE Mean')
ax_vm.set_title(f'(A) VAE Mean (R² = {metrics_vae["Mean_R2"]:.4f})', fontweight='bold')

# VAE Std
ax_vs = axes4[0, 1]
ax_vs.scatter(real_std, vae_std, alpha=0.3, s=5, c=COLORS['VAE'])
lims = [min(real_std.min(), vae_std.min()), max(real_std.max(), vae_std.max())]
ax_vs.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax_vs.set_xlabel('Real Std')
ax_vs.set_ylabel('VAE Std')
ax_vs.set_title(f'(B) VAE Std (R² = {metrics_vae["Std_R2"]:.4f})', fontweight='bold')

# GAN Mean
ax_gm = axes4[1, 0]
ax_gm.scatter(real_mean, gan_mean, alpha=0.3, s=5, c=COLORS['GAN'])
lims = [min(real_mean.min(), gan_mean.min()), max(real_mean.max(), gan_mean.max())]
ax_gm.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax_gm.set_xlabel('Real Mean')
ax_gm.set_ylabel('GAN Mean')
ax_gm.set_title(f'(C) GAN Mean (R² = {metrics_gan["Mean_R2"]:.4f})', fontweight='bold')

# GAN Std
ax_gs = axes4[1, 1]
ax_gs.scatter(real_std, gan_std, alpha=0.3, s=5, c=COLORS['GAN'])
lims = [min(real_std.min(), gan_std.min()), max(real_std.max(), gan_std.max())]
ax_gs.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
ax_gs.set_xlabel('Real Std')
ax_gs.set_ylabel('GAN Std')
ax_gs.set_title(f'(D) GAN Std (R² = {metrics_gan["Std_R2"]:.4f})', fontweight='bold')

plt.suptitle('Gene-level Statistics: Real vs Synthetic', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "Figure6_MeanStd_Scatter.png")
plt.savefig(scatter_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   Mean/Std scatter saved: {scatter_path}")

# =============================================================================
# 최종 비교 테이블 출력
# =============================================================================
print("\n[5/5] Final comparison table...")

print("\n" + "=" * 80)
print("TABLE: Comprehensive Comparison of Synthetic Data Generation Methods")
print("=" * 80)
print(f"""
{'Metric':<30} {'VAE':<20} {'GAN':<20} {'Winner':<10}
{'-'*80}
{'Mean Correlation (r)':<30} {metrics_vae['Mean_Corr']:<20.4f} {metrics_gan['Mean_Corr']:<20.4f} {'VAE' if metrics_vae['Mean_Corr'] > metrics_gan['Mean_Corr'] else 'GAN':<10}
{'Mean R²':<30} {metrics_vae['Mean_R2']:<20.4f} {metrics_gan['Mean_R2']:<20.4f} {'VAE' if metrics_vae['Mean_R2'] > metrics_gan['Mean_R2'] else 'GAN':<10}
{'Std Correlation (r)':<30} {metrics_vae['Std_Corr']:<20.4f} {metrics_gan['Std_Corr']:<20.4f} {'VAE' if metrics_vae['Std_Corr'] > metrics_gan['Std_Corr'] else 'GAN':<10}
{'Std R²':<30} {metrics_vae['Std_R2']:<20.4f} {metrics_gan['Std_R2']:<20.4f} {'VAE' if metrics_vae['Std_R2'] > metrics_gan['Std_R2'] else 'GAN':<10}
{'Diversity (Pairwise Dist)':<30} {metrics_vae['Diversity']:<20.2f} {metrics_gan['Diversity']:<20.2f} {'VAE' if abs(metrics_vae['Diversity_Ratio']-1) < abs(metrics_gan['Diversity_Ratio']-1) else 'GAN':<10}
{'Diversity Ratio vs Real':<30} {metrics_vae['Diversity_Ratio']*100:<19.2f}% {metrics_gan['Diversity_Ratio']*100:<19.2f}% {'VAE' if abs(metrics_vae['Diversity_Ratio']-1) < abs(metrics_gan['Diversity_Ratio']-1) else 'GAN':<10}
{'-'*80}
{'Real Data Diversity (Ref)':<30} {real_self_diversity:<20.2f} {'':<20} {'Baseline':<10}
""")
print("=" * 80)

# 승자 판정
vae_wins = sum([
    metrics_vae['Mean_R2'] > metrics_gan['Mean_R2'],
    metrics_vae['Std_R2'] > metrics_gan['Std_R2'],
    abs(metrics_vae['Diversity_Ratio'] - 1) < abs(metrics_gan['Diversity_Ratio'] - 1)
])

print(f"\nOVERALL WINNER: {'VAE' if vae_wins >= 2 else 'GAN'} ({vae_wins}/3 metrics)")
print("\n" + "=" * 80)

# =============================================================================
# 파일 목록 출력
# =============================================================================
print(f"""
OUTPUT FILES:
  1. {pca_path}
  2. {overlay_path}
  3. {stats_path}
  4. {scatter_path}

All figures are publication-ready (300 DPI, white background).
""")
print("=" * 80)
