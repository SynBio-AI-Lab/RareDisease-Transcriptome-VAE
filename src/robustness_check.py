"""
5-Run Robustness Check for VAE Synthetic Data Generation
=========================================================
Validates reproducibility of the VAE model across multiple random seeds.
Outputs Mean ± Std for TSTR metrics and identifies the representative run.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
ROBUSTNESS_DIR = os.path.join(BASE_DIR, "results", "robustness")
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

# Hyperparameters (PRESERVED from original)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 1000
HIDDEN_DIM = 512
LATENT_DIM = 64
KL_WEIGHT = 0.005
SYNTHETIC_SAMPLES = 1000
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 50
LR_SCHEDULER_PATIENCE = 30
TEMPERATURE = 2.0  # PRESERVED: Sampling Temperature

# 5 Independent Seeds
SEEDS = [42, 10, 2024, 777, 99]
N_RUNS = len(SEEDS)

# Publication-quality plot settings
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Load Real Data (Once)
# =============================================================================
print("=" * 70)
print("5-RUN ROBUSTNESS CHECK: VAE Synthetic Data Validation")
print("=" * 70)
print(f"\nDevice: {device}")
print(f"Seeds: {SEEDS}")
print(f"Temperature: {TEMPERATURE}")
print("=" * 70)

print("\n[0/6] Loading Real Data...")
if not os.path.exists(INPUT_FILE):
    print(f"ERROR: File not found: {INPUT_FILE}")
    sys.exit(1)

df_real = pd.read_csv(INPUT_FILE, index_col=0, compression='gzip')
if df_real.shape[0] > df_real.shape[1] and df_real.shape[0] > 1000:
    df_real = df_real.T
df_real = df_real.loc[:, ~df_real.columns.duplicated()]
print(f"   - Real Data: {df_real.shape[0]} samples x {df_real.shape[1]} genes")

# =============================================================================
# VAE Model Definition
# =============================================================================
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM // 2)
        self.fc21 = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc22 = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, HIDDEN_DIM // 2)
        self.bn3 = nn.BatchNorm1d(HIDDEN_DIM // 2)
        self.fc4 = nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM)
        self.bn4 = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc5 = nn.Linear(HIDDEN_DIM, input_dim)

    def encode(self, x):
        h = torch.relu(self.bn1(self.fc1(x)))
        h = torch.relu(self.bn2(self.fc2(h)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.bn3(self.fc3(z)))
        h = torch.relu(self.bn4(self.fc4(h)))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (KL_WEIGHT * KLD)

# =============================================================================
# Single Run Function
# =============================================================================
def run_single_experiment(seed, run_idx):
    """Execute full pipeline for a single seed."""
    
    run_dir = os.path.join(ROBUSTNESS_DIR, f"run_{run_idx + 1}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*50}")
    print(f"RUN {run_idx + 1}/{N_RUNS} (Seed: {seed})")
    print(f"{'='*50}")
    
    # --- Data Preparation ---
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_real.values)
    input_dim = df_real.shape[1]
    
    train_data, val_data = train_test_split(data_scaled, test_size=VALIDATION_SPLIT, random_state=seed)
    train_tensor = torch.FloatTensor(train_data).to(device)
    val_tensor = torch.FloatTensor(val_data).to(device)
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)
    
    # --- VAE Training ---
    model = VAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print("   Training VAE...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for data, in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, in val_loader:
                recon, mu, logvar = model(data)
                loss = loss_function(recon, data, mu, logvar)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"   Early stopping at epoch {epoch + 1}")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    
    # --- Synthetic Data Generation (with Temperature) ---
    model.eval()
    with torch.no_grad():
        z = torch.randn(SYNTHETIC_SAMPLES, LATENT_DIM).to(device) * TEMPERATURE
        syn_scaled = model.decode(z).cpu().numpy()
        syn_data = scaler.inverse_transform(syn_scaled)
    
    df_syn = pd.DataFrame(syn_data, columns=df_real.columns)
    
    # --- TSTR Evaluation ---
    scaler_eval = StandardScaler()
    X_real_scaled = scaler_eval.fit_transform(df_real.values)
    X_syn_scaled = scaler_eval.transform(df_syn.values)
    
    n_components = min(50, df_real.shape[1], df_real.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    X_real_pca = pca.fit_transform(X_real_scaled)
    X_syn_pca = pca.transform(X_syn_scaled)
    
    # K-Means pseudo-labels
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=10)
    y_real = kmeans.fit_predict(X_real_pca)
    y_syn = kmeans.predict(X_syn_pca)
    
    # Train/Test split
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
        X_real_pca, y_real, test_size=0.3, random_state=seed, stratify=y_real
    )
    
    # TSTR: Train on Synthetic, Test on Real
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_syn_pca, y_syn)
    y_pred = clf.predict(X_real_test)
    y_prob = clf.predict_proba(X_real_test)[:, 1]
    
    accuracy = accuracy_score(y_real_test, y_pred)
    f1 = f1_score(y_real_test, y_pred, average='weighted')
    try:
        auc = roc_auc_score(y_real_test, y_prob)
    except:
        auc = 0.5  # If only one class in test set
    
    print(f"   TSTR Results: Acc={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # --- PCA Visualization ---
    pca_2d = PCA(n_components=2, random_state=seed)
    X_combined = np.vstack([X_real_scaled, X_syn_scaled])
    X_pca_2d = pca_2d.fit_transform(X_combined)
    
    labels = ['Real'] * len(df_real) + ['Synthetic'] * len(df_syn)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in [('Real', '#3498db'), ('Synthetic', '#e74c3c')]:
        mask = np.array(labels) == label
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                   c=color, label=label, alpha=0.6, s=30, edgecolors='none')
    
    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA: Real vs Synthetic (Run {run_idx + 1}, Seed={seed})\n'
                 f'TSTR: Acc={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}')
    ax.legend(loc='upper right')
    
    pca_path = os.path.join(run_dir, f"pca_run_{run_idx + 1}.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"   PCA saved: {pca_path}")
    
    return {
        'run': run_idx + 1,
        'seed': seed,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'pca_path': pca_path,
        'val_loss': best_val_loss
    }

# =============================================================================
# Main Execution: 5-Run Loop
# =============================================================================
print("\n" + "=" * 70)
print("STARTING 5-RUN ROBUSTNESS CHECK")
print("=" * 70)

results = []
for i, seed in enumerate(SEEDS):
    result = run_single_experiment(seed, i)
    results.append(result)

# =============================================================================
# Metric Aggregation
# =============================================================================
results_df = pd.DataFrame(results)

acc_mean = results_df['accuracy'].mean()
acc_std = results_df['accuracy'].std()
f1_mean = results_df['f1'].mean()
f1_std = results_df['f1'].std()
auc_mean = results_df['auc'].mean()
auc_std = results_df['auc'].std()

# =============================================================================
# Identify Representative Run (closest to mean accuracy)
# =============================================================================
results_df['dist_to_mean'] = abs(results_df['accuracy'] - acc_mean)
representative_idx = results_df['dist_to_mean'].idxmin()
representative_run = results_df.loc[representative_idx]

# Copy representative PCA plot
import shutil
representative_pca_src = representative_run['pca_path']
representative_pca_dst = os.path.join(ROBUSTNESS_DIR, "Figure2_Representative_PCA.png")
shutil.copy(representative_pca_src, representative_pca_dst)

# =============================================================================
# Summary Output
# =============================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECK COMPLETE")
print("=" * 70)

print("\n### INDIVIDUAL RUN RESULTS ###")
print("-" * 70)
print(f"{'Run':<6} {'Seed':<8} {'Accuracy':<12} {'F1-Score':<12} {'AUC':<12}")
print("-" * 70)
for _, row in results_df.iterrows():
    marker = " *" if row['run'] == representative_run['run'] else ""
    print(f"{int(row['run']):<6} {int(row['seed']):<8} {row['accuracy']:<12.4f} {row['f1']:<12.4f} {row['auc']:<12.4f}{marker}")
print("-" * 70)
print("* = Representative Run (closest to mean)\n")

# Paper-ready summary
print("\n" + "=" * 70)
print("PAPER-READY SUMMARY (Copy-Paste)")
print("=" * 70)
summary_text = f"""
================================================================================
Table X: 5-Run Robustness Validation of VAE Synthetic Data (Temperature = {TEMPERATURE})
================================================================================

Metric              Mean +/- Std         Range
--------------------------------------------------------------------------------
Accuracy            {acc_mean:.4f} +/- {acc_std:.4f}      [{results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}]
F1-Score            {f1_mean:.4f} +/- {f1_std:.4f}      [{results_df['f1'].min():.4f} - {results_df['f1'].max():.4f}]
AUC                 {auc_mean:.4f} +/- {auc_std:.4f}      [{results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}]
--------------------------------------------------------------------------------

Representative Run: Run {int(representative_run['run'])} (Seed = {int(representative_run['seed'])})
  - Accuracy: {representative_run['accuracy']:.4f}
  - F1-Score: {representative_run['f1']:.4f}
  - AUC: {representative_run['auc']:.4f}

Conclusion: The VAE model demonstrates {
    'excellent' if acc_std < 0.02 else 'good' if acc_std < 0.05 else 'moderate'
} reproducibility across 5 independent runs with different random seeds.
================================================================================
"""
print(summary_text)

# Save summary to file
summary_path = os.path.join(ROBUSTNESS_DIR, "robustness_summary.txt")
with open(summary_path, 'w') as f:
    f.write(summary_text)

# Save detailed results to CSV
csv_path = os.path.join(ROBUSTNESS_DIR, "robustness_results.csv")
results_df.to_csv(csv_path, index=False)

print(f"\nFiles saved:")
print(f"  - Summary: {summary_path}")
print(f"  - Results CSV: {csv_path}")
print(f"  - Representative PCA: {representative_pca_dst}")
print(f"  - Individual PCAs: {ROBUSTNESS_DIR}/run_*/pca_run_*.png")
print("\n" + "=" * 70)
