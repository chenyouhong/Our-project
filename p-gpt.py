# -*- coding: utf-8 -*-
import os
import glob
import math
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==================== 全局配置 ====================

# 滑动窗口长度
L = 100   # 历史窗口长度
H = 50    # 未来窗口长度

# 只用 IMU 6 维
D_cond = 6
D_target = 6


# ==================== ALFA 官方故障时间（秒，relative to flight start） ====================
# 仅用于 failure flight 的测试阶段和（如果你把 failure flight 也放进 train_files）训练截断
fault_time_dict = {
    "2018-07-18-15-53-31_1": 116.3,
    "2018-07-18-15-53-31_2": 73.4,
    "2018-07-18-16-22-01":   116.6,
    "2018-07-18-16-37-39_2": 114.2,
    "2018-07-30-16-29-45":   123.1,
    "2018-07-30-16-39-00_1": 116.7,
    "2018-07-30-16-39-00_2": 91.6,
    "2018-07-30-17-10-45":   117.2,
    "2018-07-30-17-20-01":   87.7,
    "2018-07-30-17-36-35":   133.4,
    "2018-07-30-17-46-31":   90.3,
    "2018-09-11-11-56-30":   103.6,
    "2018-09-11-14-22-07_1": 104.8,
    "2018-09-11-14-22-07_2": 49.9,
    "2018-09-11-14-41-51":   117.8,
    "2018-09-11-14-52-54":   105.2,
    "2018-09-11-15-05-11_1": 63.4,
    "2018-09-11-15-06-34_1": 55.5,
    "2018-09-11-15-06-34_2": 51.9,
    "2018-09-11-15-06-34_3": 60.1,
    "2018-09-11-17-27-13_1": 116.3,
    "2018-09-11-17-27-13_2": 65.8,
    "2018-09-11-17-55-30_1": 111.9,
    "2018-09-11-17-55-30_2": 50.0,
    "2018-10-05-14-34-20_2": 152.2,
    "2018-10-05-14-37-22_2": 73.4,
    "2018-10-05-14-37-22_3": 72.4,
    "2018-10-05-15-52-12_3": 49.1,
    "2018-10-05-15-55-10":   100.1,
    "2018-10-05-16-04-46":   76.2,
    "2018-10-18-11-03-57":   104.2,
    "2018-10-18-11-04-00":   111.1,
    "2018-10-18-11-04-08_1": 100.3,
    "2018-10-18-11-04-08_2": 98.2,
    "2018-10-18-11-04-35":   101.3,
    "2018-10-18-11-06-06":   102.5,
}


# ==================== 工具函数 ====================

def get_flight_id(path: str) -> str:
    """
    从路径中抽取 flight 的 Sequence Name，与 README 中一致.

    例：
    data/alfa/train/carbonZ_2018-07-18-15-53-31_1_engine_failure/mavros-imu-data.csv
    -> 2018-07-18-15-53-31_1
    """
    dirname = os.path.basename(os.path.dirname(path))
    if dirname.startswith("carbonZ_"):
        dirname = dirname[len("carbonZ_"):]
    parts = dirname.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0] + "_" + parts[1]
    else:
        return parts[0]


# ==================== Dataset 实现 ====================

class ALFADataset(Dataset):
    """
    - 不跨文件滑窗，每个 flight 单独滑窗
    - train 模式：如果 flight 在 fault_time_dict 中，则剪掉故障后的数据
    - test 模式：不裁剪，保留全序列
    """

    def __init__(self,
                 file_paths,
                 mode: str = 'train',
                 scaler: StandardScaler = None,
                 L: int = 50,
                 H: int = 20,
                 fault_times: dict = None):

        self.L = L
        self.H = H
        self.mode = mode
        self.fault_times = fault_times or {}

        # 只用 IMU 6 维
        self.features = [
            'linear_acceleration.x',
            'linear_acceleration.y',
            'linear_acceleration.z',
            'angular_velocity.x',
            'angular_velocity.y',
            'angular_velocity.z'
        ]

        self.flight_ids = []
        self.time_seqs = []
        raw_seqs = []

        for path in file_paths:
            df = pd.read_csv(path)

            for col in self.features:
                if col not in df.columns:
                    raise ValueError(f"Column {col} not found in {path}")
            if 'Time' not in df.columns:
                raise ValueError(f"Column 'Time' not found in {path}")

            X_full = df[self.features].values.astype(np.float32)
            t_full = df['Time'].values.astype(np.float32)
            fid = get_flight_id(path)

            # train 模式下，如有故障标注，则截取到故障开始前
            if mode == 'train' and fid in self.fault_times:
                fault_time_rel = float(self.fault_times[fid])  # 秒（相对飞行开始）
                t0 = t_full[0]
                fault_time_abs = t0 + fault_time_rel
                fault_idx = int(np.searchsorted(t_full, fault_time_abs, side='left'))
                fault_idx = max(0, min(fault_idx, len(X_full)))
                X = X_full[:fault_idx]
                t = t_full[:fault_idx]
            else:
                X = X_full
                t = t_full

            if len(X) <= (L + H):
                continue

            raw_seqs.append(X)
            self.time_seqs.append(t)
            self.flight_ids.append(fid)

        if len(raw_seqs) == 0:
            raise ValueError("No valid sequences found for ALFADataset.")

        # 归一化
        if mode == 'train':
            full = np.concatenate(raw_seqs, axis=0)
            self.scaler = StandardScaler().fit(full)
        else:
            assert scaler is not None, "Test mode requires a fitted scaler."
            self.scaler = scaler

        self.seqs = []
        self.windows = []

        for idx_flight, X in enumerate(raw_seqs):
            X_norm = self.scaler.transform(X)
            seq_tensor = torch.tensor(X_norm, dtype=torch.float32)
            self.seqs.append(seq_tensor)

            T = len(seq_tensor)
            for start in range(0, T - (L + H)):
                self.windows.append((idx_flight, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        idx_flight, start = self.windows[idx]
        seq = self.seqs[idx_flight]
        C = seq[start:start + self.L]
        Y = seq[start + self.L:start + self.L + self.H]
        return C, Y

    def get_scaler(self):
        return self.scaler


# ==================== 基础模块（位置编码 + 时间步编码） ====================

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TimeEmbedding(nn.Module):
    def __init__(self, num_steps: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_steps, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.emb(t)
        return self.mlp(x)


# ==================== 扩散调度器 ====================

class DiffusionConfig:
    def __init__(self, num_steps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end


class GaussianDiffusionSchedule(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.num_steps = cfg.num_steps

        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise


# ==================== 条件编码器 & 未来 backbone ====================

class CondEncoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_ff: int = 128,
        max_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cond_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(cond_seq)
        x = self.pos_enc(x)
        h = self.encoder(x)
        h = self.norm(h)
        h_pool = h.mean(dim=1)
        return h_pool


class FutureBackbone(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_ff: int = 128,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        h = self.encoder(x)
        h = self.norm(h)
        return h


# ==================== UAV 扩散模型 ====================

class UAVDiffusionModel(nn.Module):
    def __init__(
        self,
        D_cond: int,
        D_target: int,
        diffusion_cfg: DiffusionConfig,
        d_model_cond: int = 32,
        d_model_future: int = 32,
    ):
        super().__init__()
        self.D_cond = D_cond
        self.D_target = D_target
        self.diffusion_cfg = diffusion_cfg

        self.cond_encoder = CondEncoder(
            d_in=D_cond,
            d_model=d_model_cond,
            num_layers=1,
            num_heads=4,
            dim_ff=128,
            max_len=512,
            dropout=0.1,
        )

        self.time_embed = TimeEmbedding(
            num_steps=diffusion_cfg.num_steps,
            d_model=d_model_future
        )

        self.future_proj = nn.Linear(D_target, d_model_future)
        self.fuse_proj = nn.Linear(d_model_future + d_model_cond, d_model_future)

        self.future_backbone = FutureBackbone(
            d_model=d_model_future,
            num_layers=2,
            num_heads=4,
            dim_ff=128,
            max_len=256,
            dropout=0.1,
        )

        self.out_proj = nn.Linear(d_model_future, D_target)

    def forward(self, C: torch.Tensor, Y_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, H, _ = Y_t.shape

        cond_emb = self.cond_encoder(C)         # [B, d_model_cond]
        future_emb = self.future_proj(Y_t)      # [B, H, d_model_future]

        t_emb = self.time_embed(t).unsqueeze(1)
        future_emb = future_emb + t_emb

        cond_expand = cond_emb.unsqueeze(1).expand(B, H, -1)
        x = torch.cat([future_emb, cond_expand], dim=-1)
        x = self.fuse_proj(x)

        h = self.future_backbone(x)
        eps_pred = self.out_proj(h)
        return eps_pred


# ==================== 训练 & 验证 loss ====================

def diffusion_loss(
    model: UAVDiffusionModel,
    schedule: GaussianDiffusionSchedule,
    C_batch: torch.Tensor,
    Y0_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    C_batch = C_batch.to(device)
    Y0_batch = Y0_batch.to(device)

    B = Y0_batch.size(0)
    T = schedule.num_steps

    t = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
    eps = torch.randn_like(Y0_batch)
    Y_t = schedule.q_sample(Y0_batch, t, eps)

    eps_pred = model(C_batch, Y_t, t)
    loss = F.mse_loss(eps_pred, eps)
    return loss


def training_step(
    model: UAVDiffusionModel,
    schedule: GaussianDiffusionSchedule,
    C_batch: torch.Tensor,
    Y0_batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = None,
):
    model.train()
    loss = diffusion_loss(model, schedule, C_batch, Y0_batch, device)

    optimizer.zero_grad()
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return loss.item()


# ==================== 扩散异常分数 ====================

@torch.no_grad()
def diffusion_nll_like_score(
        model: nn.Module,
        schedule: GaussianDiffusionSchedule,
        C: torch.Tensor,      # [B, L, D]
        Y0: torch.Tensor,     # [B, H, D]
        num_t: int = 8
) -> torch.Tensor:
    model.eval()
    device = Y0.device
    B = Y0.size(0)
    T = schedule.num_steps

    total = torch.zeros(B, device=device)

    for _ in range(num_t):
        t = torch.randint(low=0, high=T, size=(B,), device=device)
        eps = torch.randn_like(Y0)
        Y_t = schedule.q_sample(Y0, t, eps)
        eps_pred = model(C, Y_t, t)

        loss = F.mse_loss(eps_pred, eps, reduction='none').mean(dim=[1, 2])
        total += loss

    return total / num_t


def determine_engineering_threshold(
        model: nn.Module,
        schedule: GaussianDiffusionSchedule,
        train_loader: DataLoader,
        device: torch.device,
        score_fn,
        k: float = 3.0
):
    print(f"\n[Threshold Calibration] Computing statistics on {len(train_loader.dataset)} normal samples...")
    model.eval()
    scores = []

    with torch.no_grad():
        for i, (C_batch, Y0_batch) in enumerate(train_loader):
            C_batch, Y0_batch = C_batch.to(device), Y0_batch.to(device)
            batch_scores = score_fn(model, schedule, C_batch, Y0_batch)
            scores.extend(batch_scores.cpu().tolist())

            if (i + 1) % 50 == 0:
                print(f"  -> Scanned batch {i + 1}...")

    scores = np.array(scores)
    mu = np.mean(scores)
    std = np.std(scores)

    threshold_sigma = mu + k * std
    threshold_percentile = np.percentile(scores, 99)

    print(f"[Stats] Mean: {mu:.4f}, Std: {std:.4f}, Max: {np.max(scores):.4f}")
    print(f"[Result] Suggested Threshold (3-Sigma): {threshold_sigma:.4f}")
    print(f"[Result] Suggested Threshold (99-Percentile): {threshold_percentile:.4f}")

    return threshold_sigma


# ==================== main：训练 + 测试 ====================

def main_alfa():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 数据路径（和你现在的一致）
    train_files = glob.glob('data/alfa/train/**/mavros-imu-data.csv', recursive=True)
    test_files = glob.glob('data/alfa/test/**/mavros-imu-data.csv', recursive=True)

    if len(train_files) == 0:
        print("Error: No training files found.")
        return

    print(f"Found {len(train_files)} training IMU files.")
    print("Example paths and flight_ids:")
    for p in train_files[:5]:
        print(f"{p}   -->   {get_flight_id(p)}")

    # 2. 构建训练集（train 模式下会自动根据 fault_time_dict 截断有故障的 flight）
    print("Loading training data...")
    train_dataset_full = ALFADataset(
        train_files,
        mode='train',
        L=L,
        H=H,
        fault_times=fault_time_dict   # 你的 no_failure flight 不在字典里，不会被截断
    )
    scaler = train_dataset_full.get_scaler()
    print(f"Total training windows: {len(train_dataset_full)}")

    num_workers = 0 if os.name == 'nt' else 4

    # 2.1 按窗口随机划分 train / val（8:2）
    n_total = len(train_dataset_full)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    g = torch.Generator()
    g.manual_seed(42)  # 固定随机种子
    train_subset, val_subset = random_split(train_dataset_full, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=num_workers)

    # 3. 模型初始化（更小的 Transformer）
    diffusion_cfg = DiffusionConfig(num_steps=200)
    schedule = GaussianDiffusionSchedule(diffusion_cfg).to(device)
    model = UAVDiffusionModel(
        D_cond=D_cond,
        D_target=D_target,
        diffusion_cfg=diffusion_cfg,
        d_model_cond=32,
        d_model_future=32
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    # 4. 训练 + 验证 + early stopping
    max_epochs = 80
    patience = 10
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    print("\n>>> Phase 1: Training on ALFA nominal segments (with val & early stopping)...")
    for epoch in range(max_epochs):
        # ---- Train ----
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        for C_batch, Y0_batch in train_loader:
            loss_value = training_step(
                model, schedule,
                C_batch, Y0_batch,
                optimizer, device,
                max_grad_norm=1.0
            )
            total_train_loss += loss_value
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(n_train_batches, 1)

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for C_val, Y_val in val_loader:
                loss_val = diffusion_loss(model, schedule, C_val, Y_val, device)
                total_val_loss += loss_val.item()
                n_val_batches += 1
        avg_val_loss = total_val_loss / max(n_val_batches, 1)

        scheduler.step()

        print(f"Epoch {epoch + 1}/{max_epochs}, "
              f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

        # ---- Early stopping ----
        if avg_val_loss + 1e-4 < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. "
                      f"Best val_loss={best_val_loss:.4f}")
                break

    # 加载验证集最优权重
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), "uav_diffusion_best.pt")
        print("Best model saved to uav_diffusion_best.pt")

    # 5. 阈值标定（用全部训练窗口）
    print("\n>>> Phase 1.5: Calibrating threshold on full training data...")
    calib_loader = DataLoader(train_dataset_full, batch_size=64, shuffle=False, num_workers=num_workers)
    engineering_threshold = determine_engineering_threshold(
        model, schedule, calib_loader, device,
        score_fn=lambda m, s, C, Y: diffusion_nll_like_score(m, s, C, Y, num_t=8),
        k=3.0
    )
    print(f"Engineering threshold: {engineering_threshold:.4f}")

    # 6. 测试（和你之前一样，先选一个 failure 文件测试）
    print("\n>>> Phase 2: Testing on failure data...")
    if not test_files:
        print("No test files found.")
        return

    test_file = test_files[0]
    print(f"Testing file: {test_file}")
    df_test = pd.read_csv(test_file)

    if 'linear_acceleration.z' in df_test.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(df_test['linear_acceleration.z'].values, label='Acc Z (raw)')
        plt.title(f"Raw IMU (Z) - {os.path.basename(test_file)}")
        plt.legend()
        plt.grid(True)
        plt.show()

    test_dataset = ALFADataset(
        [test_file],
        mode='test',
        scaler=scaler,
        L=L,
        H=H,
        fault_times=None
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    print("Computing anomaly scores (diffusion_nll_like_score)...")
    anomaly_scores = []
    with torch.no_grad():
        for i, (C_batch, Y0_batch) in enumerate(test_loader):
            C_batch, Y0_batch = C_batch.to(device), Y0_batch.to(device)
            batch_scores = diffusion_nll_like_score(
                model, schedule, C_batch, Y0_batch, num_t=8
            )
            anomaly_scores.extend(batch_scores.cpu().tolist())

    anomaly_scores = np.array(anomaly_scores)
    num_windows = len(anomaly_scores)

    # 构造 GT
    flight_id = get_flight_id(test_file)
    y_true = np.zeros(num_windows, dtype=int)

    if flight_id in fault_time_dict:
        fault_time_rel = float(fault_time_dict[flight_id])
        t_arr = df_test['Time'].values.astype(np.float32)
        t0 = t_arr[0]
        fault_time_abs = t0 + fault_time_rel

        fault_row_idx = int(np.searchsorted(t_arr, fault_time_abs, side='left'))
        failure_start_index = max(0, min(num_windows - 1, fault_row_idx - L))

        y_true[failure_start_index:] = 1

        print(f"\n[GT] flight_id={flight_id}, "
              f"fault_time_rel={fault_time_rel:.2f}s, "
              f"fault_row_idx={fault_row_idx}, "
              f"failure_start_window={failure_start_index}")
    else:
        failure_start_index = None
        print(f"\n[GT] flight_id={flight_id} not in fault_time_dict, treat as nominal.")

    # 用工程阈值评估
    manual_thresh = engineering_threshold
    y_pred = (anomaly_scores > manual_thresh).astype(int)

    print("\n" + "=" * 40)
    print(f"Manual threshold = {manual_thresh:.4f}")
    print("=" * 40)
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    # 扫描最佳阈值
    print("\n[Auto] Searching best threshold by F1...")
    best_f1 = 0.0
    best_thresh = 0.0
    best_metrics = {}

    th_min, th_max = anomaly_scores.min(), anomaly_scores.max()
    thresholds = np.linspace(th_min, th_max, 100)

    for th in thresholds:
        y_tmp = (anomaly_scores > th).astype(int)
        f1 = f1_score(y_true, y_tmp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
            best_metrics = {
                "acc": accuracy_score(y_true, y_tmp),
                "prec": precision_score(y_true, y_tmp, zero_division=0),
                "rec": recall_score(y_true, y_tmp, zero_division=0),
            }

    print("-" * 40)
    print(f"Best threshold: {best_thresh:.4f}")
    print(f"Best F1       : {best_f1:.4f}")
    if best_metrics:
        print(f"  -> Acc : {best_metrics['acc']:.4f}")
        print(f"  -> Prec: {best_metrics['prec']:.4f}")
        print(f"  -> Rec : {best_metrics['rec']:.4f}")
    print("=" * 40)

    # 可视化
    window_size = 15
    smoothed = pd.Series(anomaly_scores).rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(anomaly_scores, alpha=0.3, label="Raw score")
    plt.plot(smoothed, linewidth=2, label=f"Smoothed (MA={window_size})")

    plot_thresh = best_thresh if best_f1 > 0 else manual_thresh
    plt.axhline(plot_thresh, linestyle="--", label=f"Threshold {plot_thresh:.4f}")

    if failure_start_index is not None:
        plt.axvline(failure_start_index, color="black", linestyle=":",
                    label="GT failure start")

    plt.xlabel("Window index")
    plt.ylabel("Diffusion NLL-like score")
    plt.title(f"Anomaly detection: {os.path.basename(test_file)}")
    plt.legend()
    plt.grid(True)

    save_path = "alfa_anomaly_result_v2.png"
    plt.savefig(save_path)
    print(f"\nResult plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main_alfa()
