"""
GSE113469 Data Download and Preprocessing
=========================================
NCBI GEO에서 GSE113469 데이터를 다운로드하고,
기존 GSE134900 데이터와 호환되는지 확인합니다.

GSE113469: Celiac Disease PBMC gene expression study
"""

import GEOparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import re

# =============================================================================
# 경로 설정
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 기존 GSE134900 경로
GSE134900_PATH = os.path.join(PROJECT_DIR, "data", "raw", "NCBI GEO", 
                              "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")

GEO_ID = "GSE113469"

print("=" * 70)
print(f"{GEO_ID} Data Download and Analysis")
print("=" * 70)

# =============================================================================
# [1/5] GEOparse로 다운로드 (재시도 포함)
# =============================================================================
print(f"\n[1/5] Downloading {GEO_ID} from NCBI GEO...")
print("      (This may take several minutes...)")

# 기존 파일 삭제 (손상 방지)
soft_file = os.path.join(RAW_DIR, f"{GEO_ID}_family.soft.gz")
if os.path.exists(soft_file):
    os.remove(soft_file)

MAX_RETRIES = 5
gse = None

for attempt in range(MAX_RETRIES):
    try:
        print(f"      Attempt {attempt + 1}/{MAX_RETRIES}...")
        gse = GEOparse.get_GEO(geo=GEO_ID, destdir=RAW_DIR, silent=True)
        print("      Download and parsing complete!")
        break
    except Exception as e:
        print(f"      Attempt {attempt + 1} failed: {str(e)[:80]}")
        if os.path.exists(soft_file):
            os.remove(soft_file)
        if attempt == MAX_RETRIES - 1:
            print("\n      ERROR: All download attempts failed.")
            print("      Please try again later or download manually from:")
            print(f"      https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={GEO_ID}")
            sys.exit(1)

# =============================================================================
# 데이터셋 정보 출력
# =============================================================================
print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

metadata = gse.metadata
print(f"\n[Title]")
print(f"  {metadata.get('title', ['N/A'])[0]}")

print(f"\n[Summary]")
summary = metadata.get('summary', ['N/A'])[0]
words = summary.split()
line = "  "
for word in words:
    if len(line) + len(word) + 1 > 75:
        print(line)
        line = "  " + word
    else:
        line += " " + word if line != "  " else word
print(line)

# Organism 확인
print(f"\n[Organism]")
for gsm_name, gsm in list(gse.gsms.items())[:1]:
    org = gsm.metadata.get('organism_ch1', ['N/A'])[0]
    print(f"  {org}")

print(f"\n[Platform]")
platforms = list(gse.gpls.keys())
print(f"  {platforms}")

print(f"\n[Samples]")
print(f"  Total: {len(gse.gsms)} samples")

# =============================================================================
# [2/5] Expression Matrix 추출
# =============================================================================
print("\n" + "=" * 70)
print("[2/5] Extracting Expression Matrix...")

gsm_names = list(gse.gsms.keys())
first_gsm = gse.gsms[gsm_names[0]]
print(f"      Columns: {first_gsm.table.columns.tolist()}")

expression_data = {}
for gsm_name in gsm_names:
    gsm = gse.gsms[gsm_name]
    table = gsm.table
    if 'VALUE' in table.columns and 'ID_REF' in table.columns:
        expression_data[gsm_name] = table.set_index('ID_REF')['VALUE']

df_expr_probes = pd.DataFrame(expression_data)
print(f"      Shape (probes x samples): {df_expr_probes.shape}")

df_expr_probes = df_expr_probes.apply(pd.to_numeric, errors='coerce')
df_expr_probes = df_expr_probes.dropna(how='all')
print(f"      After NaN removal: {df_expr_probes.shape}")

# =============================================================================
# [3/5] 유전자 심볼 매핑
# =============================================================================
print("\n" + "=" * 70)
print("[3/5] Mapping Probe IDs to Gene Symbols...")

gpl_name = platforms[0]
gpl = gse.gpls[gpl_name]
gpl_table = gpl.table

print(f"      Platform: {gpl_name}")
print(f"      Columns: {gpl_table.columns.tolist()[:15]}")

# 유전자 심볼 컬럼 찾기
id_to_symbol = {}
if 'Symbol' in gpl_table.columns:
    print("      Using 'Symbol' column")
    id_to_symbol = gpl_table.set_index('ID')['Symbol'].to_dict()
elif 'ILMN_Gene' in gpl_table.columns:
    print("      Using 'ILMN_Gene' column")
    id_to_symbol = gpl_table.set_index('ID')['ILMN_Gene'].to_dict()
elif 'Gene Symbol' in gpl_table.columns:
    print("      Using 'Gene Symbol' column")
    id_to_symbol = gpl_table.set_index('ID')['Gene Symbol'].to_dict()

# 프로브 -> 유전자 심볼 매핑
df_expr_probes['Gene_Symbol'] = df_expr_probes.index.map(lambda x: id_to_symbol.get(str(x), None))

# 유효한 심볼만
df_valid = df_expr_probes[df_expr_probes['Gene_Symbol'].notna() & 
                          (df_expr_probes['Gene_Symbol'] != '')].copy()
print(f"      Probes with symbols: {len(df_valid)} / {len(df_expr_probes)}")

# 샘플 심볼 확인
sample_symbols = df_valid['Gene_Symbol'].dropna().unique()[:10].tolist()
print(f"      Sample symbols: {sample_symbols}")

# 중복 유전자 평균
sample_cols = gsm_names
df_genes = df_valid.groupby('Gene_Symbol')[sample_cols].mean()
print(f"      Unique genes: {len(df_genes)}")

# 전치: (samples x genes)
df_expr = df_genes.T

# =============================================================================
# [4/5] GSE134900과 비교
# =============================================================================
print("\n" + "=" * 70)
print("[4/5] Comparing with GSE134900 (Celiac Disease)...")

common_genes = []
compatibility = "UNKNOWN"

if os.path.exists(GSE134900_PATH) and len(df_genes) > 0:
    df_ref = pd.read_csv(GSE134900_PATH, index_col=0, compression='gzip')
    
    ref_genes = set(df_ref.index.tolist())
    new_genes = set(df_expr.columns.tolist())
    common_genes = sorted(list(ref_genes.intersection(new_genes)))
    
    print(f"\n      GSE134900: {len(ref_genes)} genes")
    print(f"      {GEO_ID}: {len(new_genes)} genes")
    print(f"      COMMON: {len(common_genes)} genes")
    
    if len(common_genes) > 5000:
        compatibility = "HIGH"
        print("      >> EXCELLENT overlap!")
    elif len(common_genes) > 1000:
        compatibility = "GOOD"
        print("      >> Good overlap!")
    elif len(common_genes) > 100:
        compatibility = "MODERATE"
        print("      >> Moderate overlap")
    else:
        compatibility = "LOW"
        print("      >> Low overlap")

# =============================================================================
# [5/5] 전처리 및 저장
# =============================================================================
print("\n" + "=" * 70)
print("[5/5] Preprocessing and Saving...")

if len(df_genes) > 0:
    if len(common_genes) >= 100:
        df_filtered = df_expr[common_genes]
        print(f"      Filtered to {len(common_genes)} common genes")
    else:
        df_filtered = df_expr
        common_genes = list(df_expr.columns)
        print(f"      Using all {len(common_genes)} genes")
    
    # 전처리
    df_filtered = df_filtered.clip(lower=0).replace(0, 1e-6)
    df_log2 = np.log2(df_filtered + 1)
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_log2.values)
    df_scaled = pd.DataFrame(data_scaled, index=df_log2.index, columns=df_log2.columns)
    
    # 저장
    df_log2.to_csv(os.path.join(PROCESSED_DIR, f"{GEO_ID}_log2.csv"))
    df_scaled.to_csv(os.path.join(PROCESSED_DIR, f"{GEO_ID}_scaled.csv"))
    with open(os.path.join(PROCESSED_DIR, "common_genes.txt"), 'w') as f:
        f.write('\n'.join(common_genes))
    
    import pickle
    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"      Saved to: {PROCESSED_DIR}")

# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Dataset: {GEO_ID}
  Title: Celiac Disease PBMC Gene Expression
  Organism: Homo sapiens
  Samples: {len(gse.gsms)}
  Genes: {len(df_genes)}
  
Compatibility with GSE134900: {compatibility}
  Common Genes: {len(common_genes)}
""")

if compatibility in ["HIGH", "GOOD"]:
    print(">> This dataset is HIGHLY SUITABLE for your VAE model!")
    print(">> Both datasets study Celiac Disease with significant gene overlap.")
elif compatibility == "MODERATE":
    print(">> This dataset can be used with some limitations.")
else:
    print(">> Limited overlap with reference dataset.")

print("=" * 70)
