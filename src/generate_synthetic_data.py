import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import os
import sys

# --- 경로 자동 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "data", "raw", "NCBI GEO", "GSE134900_normalized_expr.valerie_celiac.human.csv.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "synthetic")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 하이퍼파라미터 (튜닝 완료) ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 1000            # 최대 에폭 (Early Stopping으로 조기 종료 가능)
HIDDEN_DIM = 512
LATENT_DIM = 64
KL_WEIGHT = 0.005        # latent space 정규화 (0.001~0.01 권장)
SEED = 42
SYNTHETIC_SAMPLES = 1000
VALIDATION_SPLIT = 0.2   # 검증 세트 비율
EARLY_STOPPING_PATIENCE = 50  # 검증 손실이 개선되지 않으면 조기 종료
LR_SCHEDULER_PATIENCE = 30    # Learning Rate 감소 patience
TEMPERATURE = 2.0             # 생성 시 Latent Space 탐색 범위 확대 (1.0=기본, 2.0=2배 분산)

torch.manual_seed(SEED)
np.random.seed(SEED)
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 데이터 로드 ---
if not os.path.exists(INPUT_FILE):
    logging.error(f"❌ 파일 없음: {INPUT_FILE}")
    sys.exit(1)

logging.info("데이터 로드 중...")
df = pd.read_csv(INPUT_FILE, index_col=0, compression='gzip')
if df.shape[0] > df.shape[1] and df.shape[0] > 1000:
    df = df.T
df = df.loc[:, ~df.columns.duplicated()] # 중복 제거

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df.values)
input_dim = df.shape[1]

# Train/Validation 분리
train_data, val_data = train_test_split(data_scaled, test_size=VALIDATION_SPLIT, random_state=SEED)
logging.info(f"📊 데이터 분리: 학습 {len(train_data)}개, 검증 {len(val_data)}개")

train_tensor = torch.FloatTensor(train_data).to(device)
val_tensor = torch.FloatTensor(val_data).to(device)
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE, shuffle=False)

# --- VAE 모델 (Batch Normalization 적용) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
        self.bn1 = nn.BatchNorm1d(HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_DIM // 2)
        self.fc21 = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)  # mu
        self.fc22 = nn.Linear(HIDDEN_DIM // 2, LATENT_DIM)  # logvar
        
        # Decoder
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
        return torch.sigmoid(self.fc5(h))  # [0,1] 범위로 제한

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + (KL_WEIGHT * KLD)

# --- 학습 ---
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
)

# Early Stopping 변수
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

logging.info(f"🚀 학습 시작 (최대 {EPOCHS} Epochs, Early Stopping 적용)...")

for epoch in range(EPOCHS):
    # --- 학습 단계 ---
    model.train()
    train_loss = 0
    for data, in train_loader:
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # --- 검증 단계 ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, in val_loader:
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader.dataset)
    
    # Learning Rate Scheduler 업데이트
    scheduler.step(avg_val_loss)
    
    # Early Stopping 체크
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
    
    # 로깅 (50 에폭마다 또는 Early Stopping 임박 시)
    if (epoch + 1) % 50 == 0 or patience_counter >= EARLY_STOPPING_PATIENCE - 10:
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
            f'Epoch {epoch+1:4d} | Train Loss: {avg_train_loss:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e} | '
            f'Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}'
        )
    
    # Early Stopping 발동
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        logging.info(f"⏹️ Early Stopping 발동! (Epoch {epoch+1})")
        break

# 최적의 모델 복원
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    logging.info(f"✅ 최적 모델 복원 (Val Loss: {best_val_loss:.4f})")

# --- 모델 저장 ---
model_path = os.path.join(MODEL_DIR, "vae_celiac.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim,
    'scaler': scaler,
    'columns': df.columns.tolist()
}, model_path)
logging.info(f"✅ 모델 저장: {model_path}")

# --- 합성 데이터 생성 ---
model.eval()
with torch.no_grad():
    z = torch.randn(SYNTHETIC_SAMPLES, LATENT_DIM).to(device) * TEMPERATURE
    syn_scaled = model.decode(z).cpu().numpy()  # sigmoid로 이미 [0,1] 범위
    logging.info(f"🌡️ Temperature {TEMPERATURE} 적용하여 Latent Space 탐색 범위 확대")
    syn_data = scaler.inverse_transform(syn_scaled)
    
    output_path = os.path.join(OUTPUT_DIR, "synthetic_celiac_data.csv")
    pd.DataFrame(syn_data, columns=df.columns).to_csv(output_path, index=False)
    logging.info(f"✅ 합성 데이터 저장: {output_path} ({SYNTHETIC_SAMPLES}개 샘플)")

logging.info("🎉 완료!")