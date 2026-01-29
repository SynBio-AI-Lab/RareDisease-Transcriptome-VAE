"""
Deep Validation for VAE-Generated Synthetic Data
=================================================
Publication-quality analysis for Nature Communications / Bioinformatics level submission.

Three core validation metrics:
1. Statistical Fidelity - Mean/Std correlation between Real vs Synthetic
2. Correlation Structure - Gene-gene interaction pattern preservation
3. ML Utility (TSTR) - Train on Synthetic, Test on Real performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# --- 경로 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(BASE_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
SYN_PATH = os.path.join(BASE_DIR, "data", "synthetic", "synthetic_celiac_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# --- Publication-quality 스타일 설정 ---
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

SEED = 42
np.random.seed(SEED)

# =============================================================================
# 데이터 로드
# =============================================================================
print("=" * 70)
print("DEEP VALIDATION: VAE Synthetic Data Quality Assessment")
print("=" * 70)
print("\n[1/4] Loading Data...")

df_real = pd.read_csv(REAL_PATH, index_col=0, compression='gzip')
if df_real.shape[0] > df_real.shape[1]:
    df_real = df_real.T
df_real = df_real.loc[:, ~df_real.columns.duplicated()]

df_syn = pd.read_csv(SYN_PATH)
if 'Unnamed: 0' in df_syn.columns:
    df_syn = df_syn.drop('Unnamed: 0', axis=1)
df_syn = df_syn.loc[:, ~df_syn.columns.duplicated()]

# 공통 유전자 정렬
common_genes = df_real.columns.intersection(df_syn.columns)
df_real = df_real[common_genes]
df_syn = df_syn[common_genes]

print(f"   - Real Data: {df_real.shape[0]} samples x {df_real.shape[1]} genes")
print(f"   - Synthetic Data: {df_syn.shape[0]} samples x {df_syn.shape[1]} genes")
print(f"   - Common Genes: {len(common_genes)}")

# =============================================================================
# 1. Statistical Fidelity (통계적 충실도)
# =============================================================================
print("\n[2/4] Statistical Fidelity Analysis...")

# 유전자별 평균, 표준편차 계산
real_mean = df_real.mean()
syn_mean = df_syn.mean()
real_std = df_real.std()
syn_std = df_syn.std()

# 상관계수 및 R-squared 계산
corr_mean, p_mean = stats.pearsonr(real_mean, syn_mean)
corr_std, p_std = stats.pearsonr(real_std, syn_std)
r2_mean = corr_mean ** 2
r2_std = corr_std ** 2

print(f"   - Mean Correlation: r = {corr_mean:.4f} (R² = {r2_mean:.4f}, p < {p_mean:.2e})")
print(f"   - Std Correlation:  r = {corr_std:.4f} (R² = {r2_std:.4f}, p < {p_std:.2e})")

# 시각화: Mean & Std Scatter Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean Scatter Plot
ax1 = axes[0]
ax1.scatter(real_mean, syn_mean, alpha=0.4, s=10, c='#3498db', edgecolors='none')
# Regression line
z = np.polyfit(real_mean, syn_mean, 1)
p = np.poly1d(z)
x_line = np.linspace(real_mean.min(), real_mean.max(), 100)
ax1.plot(x_line, p(x_line), 'r-', linewidth=2, label='Regression')
# Identity line
lims = [min(real_mean.min(), syn_mean.min()), max(real_mean.max(), syn_mean.max())]
ax1.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Identity (y=x)')
ax1.set_xlabel('Real Data (Mean)')
ax1.set_ylabel('Synthetic Data (Mean)')
ax1.set_title('Gene Expression Mean Comparison')
ax1.text(0.05, 0.95, f'r = {corr_mean:.4f}\nR² = {r2_mean:.4f}\np < {p_mean:.2e}',
         transform=ax1.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.legend(loc='lower right')

# Std Scatter Plot
ax2 = axes[1]
ax2.scatter(real_std, syn_std, alpha=0.4, s=10, c='#e74c3c', edgecolors='none')
z = np.polyfit(real_std, syn_std, 1)
p = np.poly1d(z)
x_line = np.linspace(real_std.min(), real_std.max(), 100)
ax2.plot(x_line, p(x_line), 'b-', linewidth=2, label='Regression')
lims = [min(real_std.min(), syn_std.min()), max(real_std.max(), syn_std.max())]
ax2.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Identity (y=x)')
ax2.set_xlabel('Real Data (Std)')
ax2.set_ylabel('Synthetic Data (Std)')
ax2.set_title('Gene Expression Std Comparison')
ax2.text(0.05, 0.95, f'r = {corr_std:.4f}\nR² = {r2_std:.4f}\np < {p_std:.2e}',
         transform=ax2.transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax2.legend(loc='lower right')

plt.tight_layout()
fidelity_path = os.path.join(FIG_DIR, "statistical_fidelity.png")
plt.savefig(fidelity_path)
plt.close()
print(f"   -> Saved: {fidelity_path}")

# =============================================================================
# 2. Correlation Structure (유전자 상관관계 보존)
# =============================================================================
print("\n[3/4] Correlation Structure Analysis...")

# 분산 기준 상위 50개 유전자 선택
gene_variance = df_real.var().sort_values(ascending=False)
top_50_genes = gene_variance.head(50).index.tolist()
print(f"   - Selected Top 50 High-Variance Genes")

# 상관행렬 계산
real_corr = df_real[top_50_genes].corr()
syn_corr = df_syn[top_50_genes].corr()

# 상관행렬 간의 유사도 측정
corr_matrices_corr, _ = stats.pearsonr(real_corr.values.flatten(), syn_corr.values.flatten())
print(f"   - Correlation Matrix Similarity: r = {corr_matrices_corr:.4f}")

# Heatmap 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Real Data Heatmap
sns.heatmap(real_corr, ax=axes[0], cmap='RdBu_r', center=0, 
            vmin=-1, vmax=1, square=True,
            xticklabels=False, yticklabels=False,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
axes[0].set_title('Real Data\nGene-Gene Correlation (Top 50 Genes)', fontsize=14)

# Synthetic Data Heatmap
sns.heatmap(syn_corr, ax=axes[1], cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True,
            xticklabels=False, yticklabels=False,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
axes[1].set_title('Synthetic Data\nGene-Gene Correlation (Top 50 Genes)', fontsize=14)

fig.suptitle(f'Correlation Structure Preservation (Matrix Similarity: r = {corr_matrices_corr:.4f})', 
             fontsize=16, y=1.02)
plt.tight_layout()
corr_path = os.path.join(FIG_DIR, "correlation_structure.png")
plt.savefig(corr_path)
plt.close()
print(f"   -> Saved: {corr_path}")

# =============================================================================
# 3. ML Utility - TSTR (Train on Synthetic, Test on Real)
# =============================================================================
print("\n[4/4] Machine Learning Utility (TSTR) Analysis...")

# PCA로 차원 축소 (50 components)
n_components = min(50, len(common_genes), df_real.shape[0] - 1)
scaler = StandardScaler()

X_real_scaled = scaler.fit_transform(df_real.values)
X_syn_scaled = scaler.transform(df_syn.values)

pca = PCA(n_components=n_components, random_state=SEED)
X_real_pca = pca.fit_transform(X_real_scaled)
X_syn_pca = pca.transform(X_syn_scaled)

print(f"   - PCA: Reduced to {n_components} components (Explained Var: {pca.explained_variance_ratio_.sum():.2%})")

# K-Means로 가상 레이블 생성 (실제 데이터 기준)
kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
y_real = kmeans.fit_predict(X_real_pca)
y_syn = kmeans.predict(X_syn_pca)

print(f"   - Generated Pseudo-labels via K-Means (k=2)")
print(f"     Real: Class 0 = {sum(y_real==0)}, Class 1 = {sum(y_real==1)}")
print(f"     Synthetic: Class 0 = {sum(y_syn==0)}, Class 1 = {sum(y_syn==1)}")

# Train/Test Split for Real Data
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real_pca, y_real, test_size=0.3, random_state=SEED, stratify=y_real
)

# Experiment A: Train on Real, Test on Real (Baseline)
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
clf_baseline.fit(X_real_train, y_real_train)
y_pred_baseline = clf_baseline.predict(X_real_test)
acc_baseline = accuracy_score(y_real_test, y_pred_baseline)
f1_baseline = f1_score(y_real_test, y_pred_baseline, average='weighted')

# Experiment B: Train on Synthetic, Test on Real (TSTR)
clf_tstr = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
clf_tstr.fit(X_syn_pca, y_syn)
y_pred_tstr = clf_tstr.predict(X_real_test)
acc_tstr = accuracy_score(y_real_test, y_pred_tstr)
f1_tstr = f1_score(y_real_test, y_pred_tstr, average='weighted')

# 결과 출력
print("\n" + "=" * 70)
print("TSTR RESULTS: Machine Learning Utility Evaluation")
print("=" * 70)
print(f"{'Experiment':<40} {'Accuracy':>12} {'F1-Score':>12}")
print("-" * 70)
print(f"{'A) Baseline (Train Real -> Test Real)':<40} {acc_baseline:>12.4f} {f1_baseline:>12.4f}")
print(f"{'B) TSTR (Train Synthetic -> Test Real)':<40} {acc_tstr:>12.4f} {f1_tstr:>12.4f}")
print("-" * 70)

# TSTR Ratio 계산
tstr_ratio_acc = acc_tstr / acc_baseline if acc_baseline > 0 else 0
tstr_ratio_f1 = f1_tstr / f1_baseline if f1_baseline > 0 else 0
print(f"{'TSTR Ratio (B/A)':<40} {tstr_ratio_acc:>12.2%} {tstr_ratio_f1:>12.2%}")
print("=" * 70)

# TSTR 결과 시각화
fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Accuracy', 'F1-Score']
baseline_scores = [acc_baseline, f1_baseline]
tstr_scores = [acc_tstr, f1_tstr]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_scores, width, label='Baseline (Real→Real)', 
               color='#3498db', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, tstr_scores, width, label='TSTR (Synthetic→Real)', 
               color='#e74c3c', edgecolor='black', linewidth=1.2)

# 값 표시
for bar, score in zip(bars1, baseline_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar, score in zip(bars2, tstr_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Score')
ax.set_title('TSTR Evaluation: Train on Synthetic, Test on Real\n(Random Forest Classifier)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.15)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

# TSTR Ratio 주석
ax.text(0.5, 0.02, f'TSTR Ratio: Acc={tstr_ratio_acc:.1%}, F1={tstr_ratio_f1:.1%}',
        transform=ax.transAxes, ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
tstr_path = os.path.join(FIG_DIR, "tstr_evaluation.png")
plt.savefig(tstr_path)
plt.close()
print(f"\n   -> Saved: {tstr_path}")

# =============================================================================
# 4. Feature Importance Analysis (Top 10 Genes)
# =============================================================================
print("\n[5/5] Feature Importance Analysis...")

# 고분산 유전자 500개로 학습 (원본 유전자 기반 중요도 추출)
n_top_genes = min(500, len(common_genes))
top_var_genes = gene_variance.head(n_top_genes).index.tolist()

X_real_top = df_real[top_var_genes].values
X_syn_top = df_syn[top_var_genes].values

# Standardize
scaler_fi = StandardScaler()
X_real_top_scaled = scaler_fi.fit_transform(X_real_top)
X_syn_top_scaled = scaler_fi.transform(X_syn_top)

# K-Means 레이블 (원본 데이터 기준)
kmeans_fi = KMeans(n_clusters=2, random_state=SEED, n_init=10)
y_real_fi = kmeans_fi.fit_predict(X_real_top_scaled)
y_syn_fi = kmeans_fi.predict(X_syn_top_scaled)

# TSTR용 Random Forest 학습 (합성 데이터로 학습)
clf_importance = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
clf_importance.fit(X_syn_top_scaled, y_syn_fi)

# Feature Importance 추출
feature_importance = clf_importance.feature_importances_
importance_df = pd.DataFrame({
    'Gene': top_var_genes,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

# Top 10 추출
top_10_genes = importance_df.head(10).copy()
top_10_genes['Rank'] = range(1, 11)
top_10_genes = top_10_genes[['Rank', 'Gene', 'Importance']]

print(f"   - Trained Random Forest on {n_top_genes} high-variance genes")
print(f"\n   Top 10 Most Important Genes:")
print("-" * 50)
for _, row in top_10_genes.iterrows():
    print(f"   {int(row['Rank']):2d}. {row['Gene']:<20} (Importance: {row['Importance']:.4f})")
print("-" * 50)

# Bar Plot 시각화
fig, ax = plt.subplots(figsize=(12, 7))

colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, 10))
bars = ax.barh(range(10), top_10_genes['Importance'].values[::-1], 
               color=colors[::-1], edgecolor='black', linewidth=1)

ax.set_yticks(range(10))
ax.set_yticklabels(top_10_genes['Gene'].values[::-1], fontsize=11)
ax.set_xlabel('Feature Importance', fontsize=13)
ax.set_title('Top 10 Most Important Genes\n(Random Forest trained on Synthetic Data)', fontsize=14)

# 값 표시
for i, (bar, val) in enumerate(zip(bars, top_10_genes['Importance'].values[::-1])):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')

ax.set_xlim(0, top_10_genes['Importance'].max() * 1.15)
ax.invert_yaxis()  # 1위가 맨 위로

plt.tight_layout()
importance_fig_path = os.path.join(FIG_DIR, "feature_importance_top10.png")
plt.savefig(importance_fig_path)
plt.close()
print(f"\n   -> Figure Saved: {importance_fig_path}")

# Excel 저장 (openpyxl 필요)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
excel_path = os.path.join(RESULTS_DIR, "feature_importance.xlsx")

# 전체 유전자 중요도도 함께 저장
importance_df_full = importance_df.reset_index(drop=True)
importance_df_full.insert(0, 'Rank', range(1, len(importance_df_full) + 1))

try:
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        top_10_genes.to_excel(writer, sheet_name='Top 10 Genes', index=False)
        importance_df_full.to_excel(writer, sheet_name='All Genes', index=False)
    print(f"   -> Excel Saved: {excel_path}")
except ImportError:
    # openpyxl이 없으면 CSV로 저장
    csv_path = os.path.join(RESULTS_DIR, "feature_importance_top10.csv")
    csv_path_all = os.path.join(RESULTS_DIR, "feature_importance_all.csv")
    top_10_genes.to_csv(csv_path, index=False)
    importance_df_full.to_csv(csv_path_all, index=False)
    print(f"   -> CSV Saved (openpyxl not installed): {csv_path}")

# =============================================================================
# 최종 요약
# =============================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)
print(f"""
1. STATISTICAL FIDELITY
   - Mean Correlation: r = {corr_mean:.4f} (R² = {r2_mean:.4f})
   - Std Correlation:  r = {corr_std:.4f} (R² = {r2_std:.4f})
   - Interpretation: {'Excellent' if r2_mean > 0.9 else 'Good' if r2_mean > 0.7 else 'Moderate'} preservation of gene-level statistics

2. CORRELATION STRUCTURE
   - Matrix Similarity: r = {corr_matrices_corr:.4f}
   - Interpretation: {'Excellent' if corr_matrices_corr > 0.9 else 'Good' if corr_matrices_corr > 0.7 else 'Moderate'} preservation of gene-gene interactions

3. ML UTILITY (TSTR)
   - Baseline Accuracy: {acc_baseline:.4f}
   - TSTR Accuracy: {acc_tstr:.4f} ({tstr_ratio_acc:.1%} of baseline)
   - Interpretation: {'Excellent' if tstr_ratio_acc > 0.9 else 'Good' if tstr_ratio_acc > 0.7 else 'Moderate'} downstream utility
""")
print("=" * 70)
print("All figures saved to: " + FIG_DIR)
print("=" * 70)
