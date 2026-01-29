import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REAL_PATH = os.path.join(BASE_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
SYN_PATH = os.path.join(BASE_DIR, "data", "synthetic", "synthetic_celiac_data.csv")
SAVE_PATH = os.path.join(BASE_DIR, "results", "figures", "pca_validation.png")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False 

print("데이터 로드 및 전처리...")
df_real = pd.read_csv(REAL_PATH, index_col=0, compression='gzip')
if df_real.shape[0] > df_real.shape[1]: df_real = df_real.T
df_real = df_real.loc[:, ~df_real.columns.duplicated()]

df_syn = pd.read_csv(SYN_PATH, index_col=0)
df_syn = df_syn.loc[:, ~df_syn.columns.duplicated()]

# 공통 유전자만 남기기
df_real, df_syn = df_real.align(df_syn, join='inner', axis=1)
print(f"최종 데이터 크기: {df_real.shape}")

# PCA 수행
X = np.vstack([df_real.values, df_syn.values])
y = ['Real'] * len(df_real) + ['Synthetic'] * len(df_syn)
X_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))

# 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, alpha=0.6, palette={'Real':'blue', 'Synthetic':'orange'})
plt.title("PCA: Real vs Synthetic")
plt.savefig(SAVE_PATH)
print(f"✅ 그래프 저장 완료: {SAVE_PATH}")