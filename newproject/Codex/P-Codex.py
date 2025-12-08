# -*- coding: utf-8 -*-
import math
import os
import glob
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataclasses import dataclass
from statistics import NormalDist
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture

# ==================== 1. ݼ ====================
class ALFADataset(Dataset):
    def __init__(self, file_paths, mode='train', scaler=None, L=50, H=20):
        self.L = L
        self.H = H
        self.features = [
            'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
            'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
            'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'
        ]

        raw_data_list = []
        for file in file_paths:
            df = pd.read_csv(file)
            if not set(self.features).issubset(df.columns):
                continue
            data_chunk = df[self.features].values.astype(np.float32)
            raw_data_list.append(data_chunk)

        if not raw_data_list:
            raise ValueError("No valid data found in file paths.")

        full_data = np.concatenate(raw_data_list, axis=0)

        if mode == 'train':
            self.scaler = StandardScaler()
            self.normalized_data = self.scaler.fit_transform(full_data)
        else:
            assert scaler is not None, "Test mode requires a fitted scaler!"
            self.scaler = scaler
            self.normalized_data = self.scaler.transform(full_data)

        self.data = torch.tensor(self.normalized_data, dtype=torch.float32)
        self.valid_indices = len(self.data) - (self.L + self.H)

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        hist_end = idx + self.L
        future_end = hist_end + self.H

        C_seq = self.data[idx: hist_end]  # Condition [L, D]
        Y_seq = self.data[hist_end: future_end]  # Target [H, D]
        return C_seq, Y_seq

    def get_scaler(self):
        return self.scaler


# ==================== 2.  ====================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

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


@dataclass
class DiffusionConfig:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02


class GaussianDiffusionSchedule(nn.Module):
    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.num_steps = cfg.num_steps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x0, t, noise):
        B = x0.size(0)
        alpha_bar_t = self.alpha_bars[t].view(B, 1, 1)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise

    def p_sample_step(self, x_t, t_scalar, eps_pred):
        device = x_t.device
        t = torch.tensor(t_scalar, device=device, dtype=torch.long)
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bars[t]
        alpha_bar_prev = self.alpha_bars_prev[t]

        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        coef1 = torch.sqrt(alpha_bar_prev) * self.betas[t] / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

        while coef1.dim() < x_t.dim():
            coef1 = coef1.unsqueeze(-1)
        while coef2.dim() < x_t.dim():
            coef2 = coef2.unsqueeze(-1)

        mean = coef1 * x0_hat + coef2 * x_t

        if t_scalar == 0:
            return mean

        var = self.posterior_variance[t]
        while var.dim() < x_t.dim():
            var = var.unsqueeze(-1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise


# ==================== 3. ģͼܹ ====================
class CondEncoder(nn.Module):
    def __init__(self, d_in, d_model, num_layers=2, num_heads=4, dim_ff=256, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, cond_seq):
        x = self.input_proj(cond_seq)
        x = self.pos_enc(x)
        h = self.encoder(x)
        return h  # [B, L, d_model]


class FutureBackbone(nn.Module):
    def __init__(self, d_model, num_layers=4, num_heads=4, dim_ff=512, max_len=256):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.pos_enc(x)
        return self.encoder(x)


class UAVDiffusionModel(nn.Module):
    def __init__(self, D_cond, D_target, diffusion_cfg, d_model_cond=128, d_model_future=128):
        super().__init__()
        self.D_cond = D_cond
        self.D_target = D_target

        self.cond_encoder = CondEncoder(D_cond, d_model_cond)
        self.time_embed = TimeEmbedding(diffusion_cfg.num_steps, d_model_future)

        self.future_proj = nn.Linear(D_target * 2, d_model_future)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model_future, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model_future)

        self.future_backbone = FutureBackbone(d_model=d_model_future)
        self.out_proj = nn.Linear(d_model_future, D_target)

    def forward(self, C, Y_t, t, self_cond=None):
        cond_seq = self.cond_encoder(C)

        if self_cond is None:
            self_cond = torch.zeros_like(Y_t)

        x_in = torch.cat([Y_t, self_cond], dim=-1)

        future_emb = self.future_proj(x_in)  # [B, H, d]
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, d]
        query = future_emb + t_emb

        attn_out, _ = self.cross_attn(query, cond_seq, cond_seq)
        x = self.norm1(query + attn_out)

        h = self.future_backbone(x)
        return self.out_proj(h)


# ==================== 4. / M2AD ====================
def _smooth_errors(errors, smoothing_window):
    return pd.DataFrame(errors).ewm(smoothing_window).mean().values


def point_errors(y, pred, smooth=True, smoothing_window=10):
    errors = np.abs(y - pred)
    if smooth:
        errors = _smooth_errors(errors.reshape(-1, errors.shape[-1]), smoothing_window)
    return np.array(errors)


def _trapz(arr, dx):
    # numpy.trapz 已弃用，改用 trapezoid
    return np.trapezoid(arr, dx=dx)


def area_errors(y, pred, score_window=10, dx=100, smooth=True, smoothing_window=10):
    # y/pred: [N, D]
    y_df = pd.DataFrame(y)
    pred_df = pd.DataFrame(pred)
    area_y = y_df.rolling(score_window, center=True, min_periods=score_window // 2).apply(_trapz, kwargs={'dx': dx})
    area_pred = pred_df.rolling(score_window, center=True, min_periods=score_window // 2).apply(_trapz, kwargs={'dx': dx})
    errors = (area_y - area_pred).values
    mu = np.mean(errors)
    std = np.std(errors) if np.std(errors) > 0 else 1.0
    errors = (errors - mu) / std
    if smooth:
        errors = _smooth_errors(errors, smoothing_window)
    return errors


def _compute_cdf(gmm, x):
    means = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    cdf = 0
    for i in range(len(means)):
        cdf += weights[i] * NormalDist(mu=means[i], sigma=sigma[i]).cdf(x)
    return cdf


def _combine_pval(cdf, one_sided=True):
    if one_sided:
        p_val = 1 - cdf
    else:
        p_val = 2 * np.minimum(1 - cdf, cdf)
    p_val = np.clip(p_val, 1e-16, 1)
    fisher_pval = -2 * np.log(p_val)
    return fisher_pval, p_val


class GMMScorer:
    def __init__(self, sensors, n_components=1, covariance_type='spherical', one_sided=True):
        self.sensors = sensors
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.one_sided = one_sided
        self.gmm_list = [None] * len(sensors)
        self.g_scale = None
        self.g_shape = None
        self.weights = [1 / len(sensors)] * len(sensors)

    def fit(self, X):
        # X: [N, D]
        combined = 0
        for i, sensor in enumerate(self.sensors):
            x = X[:, i].reshape(-1, 1)
            gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type)
            gmm.fit(x)
            self.gmm_list[i] = gmm

            cdf = np.vectorize(_compute_cdf)(gmm, x.flatten())
            fisher, _ = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher

        var = np.var(combined)
        if var > 0:
            self.g_scale = var / np.mean(combined)
            self.g_shape = np.mean(combined) ** 2 / var
        else:
            self.g_scale = 1.0
            self.g_shape = 0.0

    def p_values(self, X):
        combined = 0
        p_val_sensors = np.zeros_like(X)
        fisher_values = np.zeros_like(X)
        for i, sensor in enumerate(self.sensors):
            y = X[:, i]
            gmm = self.gmm_list[i]
            cdf = np.vectorize(_compute_cdf)(gmm, y)
            fisher, p_val = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher
            p_val_sensors[:, i] = p_val
            fisher_values[:, i] = fisher

        gamma_p_val = 1 - gamma.cdf(combined, a=self.g_shape, scale=self.g_scale)
        return gamma_p_val, p_val_sensors, combined, fisher_values


# ==================== 5. ѵ/ƶ ====================
def predict_x0_from_xt(schedule, xt, eps_pred, t):
    B = xt.size(0)
    alpha_bar_t = schedule.alpha_bars.to(xt.device)[t].view(B, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    return (xt - sqrt_one_minus * eps_pred) / sqrt_alpha_bar


def training_step(model, schedule, C_batch, Y0_batch, optimizer, device):
    model.train()
    C_batch = C_batch.to(device)
    Y0_batch = Y0_batch.to(device)
    B = Y0_batch.size(0)

    t = torch.randint(0, schedule.num_steps, (B,), device=device).long()
    eps = torch.randn_like(Y0_batch)
    Y_t = schedule.q_sample(Y0_batch, t, eps)

    if torch.rand(1) < 0.5:
        with torch.no_grad():
            self_cond = Y0_batch
    else:
        self_cond = torch.zeros_like(Y0_batch)

    eps_pred = model(C_batch, Y_t, t, self_cond)
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def impute_future_trajectory(model, schedule, C, Y_obs, mask, device):
    model.eval()
    B, H, D = Y_obs.shape
    Y_t = torch.randn_like(Y_obs)
    self_cond = torch.zeros_like(Y_obs)

    for t_scalar in reversed(range(schedule.num_steps)):
        t = torch.full((B,), t_scalar, device=device).long()
        eps_pred = model(C, Y_t, t, self_cond)
        x0_pred = predict_x0_from_xt(schedule, Y_t, eps_pred, t)
        self_cond = x0_pred.detach()
        Y_prev = schedule.p_sample_step(Y_t, t_scalar, eps_pred)

        if t_scalar > 0:
            noise = torch.randn_like(Y_obs)
            t_prev = torch.full((B,), t_scalar - 1, device=device).long()
            Y_obs_t = schedule.q_sample(Y_obs, t_prev, noise)
        else:
            Y_obs_t = Y_obs

        Y_t = mask * Y_obs_t + (1 - mask) * Y_prev

    return Y_t


@torch.no_grad()
def collect_errors(model, schedule, loader, device, mask_ratio=0.5, error_name='point',
                   smoothing_window=10, score_window=10, dx=100, max_batches=None):
    errors_list = []
    for b_idx, (C, Y) in enumerate(loader):
        C, Y = C.to(device), Y.to(device)
        mask = (torch.rand_like(Y) > mask_ratio).float().to(device)
        Y_imputed = impute_future_trajectory(model, schedule, C, Y, mask, device)
        Y_true_np = Y.cpu().numpy()
        Y_pred_np = Y_imputed.cpu().numpy()
        mask_np = mask.cpu().numpy()

        diff = Y_true_np - Y_pred_np  # [B, H, D]
        diff_flat = diff.reshape(-1, diff.shape[-1])  # [B*H, D]
        mask_flat = mask_np.reshape(-1, mask_np.shape[-1])  # [B*H, D]

        # ?????????mask==0???????????
        rows_masked = (mask_flat < 0.5).any(axis=1)
        if rows_masked.sum() == 0:
            rows_masked = np.ones(len(diff_flat), dtype=bool)
        diff_use = diff_flat[rows_masked]

        if error_name == 'point':
            errs = point_errors(np.zeros_like(diff_use), diff_use, smooth=True, smoothing_window=smoothing_window)
        elif error_name == 'area':
            errs = area_errors(np.zeros_like(diff_use), diff_use, score_window=score_window, dx=dx, smooth=True, smoothing_window=smoothing_window)
        else:
            raise ValueError(f"Unknown error_name {error_name}")

        errors_list.append(errs)
        if max_batches is not None and (b_idx + 1) >= max_batches:
            break

    if not errors_list:
        return np.empty((0, loader.dataset.data.shape[-1]))
    return np.concatenate(errors_list, axis=0)


def calibrate_threshold(model, schedule, loader, device, sensors, mask_ratio=0.5, error_name='point',
                        smoothing_window=10, score_window=10, dx=100, threshold_percentile=99.5,
                        gamma_thresh=1e-3, max_batches=10):
    print("\n[Threshold] Calibrating with M2AD-style scoring on normal data...")
    errors = collect_errors(model, schedule, loader, device, mask_ratio, error_name,
                            smoothing_window, score_window, dx, max_batches=max_batches)
    scorer = GMMScorer(sensors=sensors, one_sided=True)
    scorer.fit(errors)
    gamma_p_val, _, combined, _ = scorer.p_values(errors)
    thresh = np.percentile(combined, threshold_percentile)
    print(f"[Stats] Combined fisher threshold (pctl {threshold_percentile}): {thresh:.4f}; gamma p-val mean={gamma_p_val.mean():.4e}")
    return scorer, thresh, gamma_thresh


@torch.no_grad()
def evaluate_loader(model, schedule, loader, device, scorer, fisher_threshold, gamma_thresh, mask_ratio=0.5, error_name='point', smoothing_window=10, score_window=10, dx=100):
    errors = collect_errors(model, schedule, loader, device, mask_ratio, error_name, smoothing_window, score_window, dx)
    gamma_p_val, _, combined, fisher_values = scorer.p_values(errors)
    anomaly_flags = (combined > fisher_threshold) | (gamma_p_val < gamma_thresh)
    return errors, combined, gamma_p_val, anomaly_flags, fisher_values


# ==================== 6.  ====================
def main_alfa():
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"Using device: {device}")

    L, H = 50, 20
    D_cond, D_target = 10, 10
    train_epochs = 30
    mask_ratio = 0.3  # ?????????????
    sensors = [
        'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
        'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
        'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'
    ]

    # 当前文件 P-Codex.py 的绝对路径
    THIS_FILE = Path(__file__).resolve()

    # 仓库根目录: D:\project\our
    ROOT_DIR = THIS_FILE.parents[2]  # Codex -> newproject -> our

    DATA_ROOT = ROOT_DIR / "data" / "alfa"

    train_dir = DATA_ROOT / "train"
    test_dir = DATA_ROOT / "test"

    # 递归查找所有 mavros-imu-data.csv
    train_files = list(train_dir.rglob("mavros-imu-data.csv"))
    test_files = list(test_dir.rglob("mavros-imu-data.csv"))

    if not train_files:
        raise RuntimeError(f"Error: No train files found in {train_dir}")
    if not test_files:
        raise RuntimeError(f"Error: No test files found in {test_dir}")

    if not train_files:
        print("Error: No train files found.")
        return

    print("Loading Data...")
    train_dataset = ALFADataset(train_files, mode='train', L=L, H=H)
    scaler = train_dataset.get_scaler()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    cfg = DiffusionConfig(num_steps=1000)
    schedule = GaussianDiffusionSchedule(cfg).to(device)
    model = UAVDiffusionModel(D_cond, D_target, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("\n>>> Phase 1: Loading Pre-trained Model...")
    if os.path.exists("uav_imputation_model.pth"):
        model.load_state_dict(torch.load("uav_imputation_model.pth"))
        print("Success: Loaded 'uav_imputation_model.pth'")
    else:
        print("Warning: Model file not found, training from scratch (this may be slow)...")
        model.train()
        for epoch in range(train_epochs):
            total_loss = 0
            for C, Y in train_loader:
                loss = training_step(model, schedule, C, Y, optimizer, device)
                total_loss += loss
            print(f"Epoch {epoch + 1}/{train_epochs}, Avg Loss: {total_loss / len(train_loader):.6f}")
        torch.save(model.state_dict(), "uav_imputation_model.pth")
        print("Saved trained model to uav_imputation_model.pth")

    model.eval()

    # ֵ궨ʹѵ/ݣ
    calib_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    scorer, fisher_threshold, gamma_thresh = calibrate_threshold(
        model, schedule, calib_loader, device, sensors, mask_ratio=mask_ratio,
        error_name='area', smoothing_window=10, score_window=10, dx=100,
        threshold_percentile=99.0, gamma_thresh=5e-3, max_batches=5  # 限制标定批次数加快运行
    )

    print("\n>>> Phase 2: Testing...")
    if not test_files:
        print("No test files found.")
        return

    test_file = test_files[0]
    print(f"Testing on: {test_file}")
    test_dataset = ALFADataset([test_file], mode='test', scaler=scaler, L=L, H=H)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    errors, combined, gamma_p_val, anomaly_flags, fisher_values = evaluate_loader(
        model, schedule, test_loader, device, scorer, fisher_threshold, gamma_thresh,
        mask_ratio=mask_ratio, error_name='area', smoothing_window=10, score_window=10, dx=100
    )

    # 򵥱ǩļ failure Ϊ쳣
    test_file_str = str(test_file).lower()
    y_true = np.ones_like(combined) if "failure" in test_file_str else np.zeros_like(combined)
    y_pred = anomaly_flags.astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 45)
    print("?? Final Performance Report (M2AD Scoring)")
    print("=" * 45)
    print(f"Accuracy  : {acc:.2%}")
    print(f"Precision : {prec:.2%}")
    print(f"Recall    : {rec:.2%}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Anomaly rate (flags): {anomaly_flags.mean():.4%}")
    print("=" * 45)

    plt.figure(figsize=(12, 6))
    plt.plot(combined, color='blue', linewidth=1, label='Fisher Combined')
    plt.axhline(fisher_threshold, color='red', linestyle='--', label='Threshold (99.5 pctl)')
    if np.sum(y_true) > 0:
        plt.axvspan(0, len(combined), color='red', alpha=0.1, label='Expected Failure Region')
    plt.title(f"Detection with M2AD Scoring | F1={f1:.3f} | Recall={rec:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_result_m2ad.png')
    plt.show()
    print("Done.")


if __name__ == "__main__":
    main_alfa()
