# -*- coding: utf-8 -*-
"""
整合模型: Diffusion Model (p-gemini) + M2AD 异常检测
核心改进:
1. 保留 Diffusion + Transformer 架构用于轨迹预测
2. 整合 M2AD 的误差计算方法 (Point/Area Error)
3. 整合 M2AD 的 GMM 异常评分系统
4. 整合 M2AD 的统计阈值确定方法
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from dataclasses import dataclass
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture
from statistics import NormalDist
from functools import partial
import logging

LOGGER = logging.getLogger(__name__)


# ==================== 1. 数据集定义 (保持不变) ====================
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
        C_seq = self.data[idx: hist_end]
        Y_seq = self.data[hist_end: future_end]
        return C_seq, Y_seq

    def get_scaler(self):
        return self.scaler


# ==================== 2. 基础组件 (保持不变) ====================
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
        betas = torch.linker(cfg.beta_start, cfg.beta_end, cfg.num_steps)
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

        while coef1.dim() < x_t.dim(): coef1 = coef1.unsqueeze(-1)
        while coef2.dim() < x_t.dim(): coef2 = coef2.unsqueeze(-1)

        mean = coef1 * x0_hat + coef2 * x_t

        if t_scalar == 0:
            return mean

        var = self.posterior_variance[t]
        while var.dim() < x_t.dim(): var = var.unsqueeze(-1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise


# ==================== 3. Diffusion 模型架构 (保持不变) ====================
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
        return h


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
        future_emb = self.future_proj(x_in)
        t_emb = self.time_embed(t).unsqueeze(1)
        query = future_emb + t_emb
        attn_out, _ = self.cross_attn(query, cond_seq, cond_seq)
        x = self.norm1(query + attn_out)
        h = self.future_backbone(x)
        return self.out_proj(h)


# ==================== 4. M2AD 误差计算方法 (新增) ====================
def _smooth(errors, smoothing_window):
    """指数加权移动平均平滑"""
    smoothed_errors = pd.DataFrame(errors).ewm(smoothing_window).mean().values
    return smoothed_errors


def point_errors(y, pred, smooth=True, smoothing_window=10):
    """
    M2AD Point Error 计算
    数学逻辑: error = |y_true - y_pred|, 然后 EWM 平滑
    """
    errors = np.abs(y - pred)
    if smooth:
        errors = _smooth(errors, smoothing_window)
    return np.array(errors)


def area_errors(y, pred, score_window=10, dx=1, smooth=True, smoothing_window=10):
    """
    M2AD Area Error 计算
    数学逻辑:
    1. 对滑动窗口内的 y 和 pred 分别计算曲线下面积 (梯形法则)
    2. error = Area(y) - Area(pred)
    3. 标准化: (error - μ) / σ
    """
    trapz = partial(np.trapz, dx=dx)
    errors = np.empty_like(y)
    num_signals = errors.shape[1]

    for i in range(num_signals):
        area_y = pd.Series(y[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)
        area_pred = pd.Series(pred[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2).apply(trapz)
        error = area_y - area_pred
        if smooth:
            error = _smooth(error, smoothing_window)
        errors[:, i] = error.flatten()

    mu = np.mean(errors)
    std = np.std(errors)
    return (errors - mu) / std


# ==================== 5. M2AD GMM 异常评分系统 (新增) ====================
def _compute_cdf(gmm, x):
    """
    计算 GMM 的累积分布函数 (CDF)
    数学逻辑: CDF(x) = Σ w_i × Φ_i(x)
    其中 Φ_i 是第 i 个高斯分量的 CDF, w_i 是权重
    """
    means = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    cdf = 0
    for i in range(len(means)):
        cdf += weights[i] * NormalDist(mu=means[i], sigma=sigma[i]).cdf(x)
    return cdf


def _combine_pval(cdf, one_sided=True):
    """
    将 CDF 转换为 p-value, 并用 Fisher's Method 组合
    数学逻辑:
    1. p_val = 1 - CDF (单侧) 或 2×min(CDF, 1-CDF) (双侧)
    2. Fisher统计量 = -2 × Σ log(p_val)
    """
    if one_sided:
        p_val = 1 - cdf
    else:
        p_val = 2 * np.array(list(map(np.min, zip(1 - cdf, cdf))))

    p_val[p_val < 1e-16] = 1e-16
    fisher_pval = -2 * np.log(p_val)
    return fisher_pval, p_val


class GMMDetector:
    """
    M2AD 的 GMM 异常检测器
    数学逻辑:
    1. 为每个传感器的误差拟合 GMM
    2. 计算每个传感器的 p-value
    3. 用 Fisher's Method 组合多传感器 p-values
    4. 拟合 Gamma 分布得到最终异常分数
    """

    def __init__(self, n_sensors, n_components=1, covariance_type='spherical', one_sided=True):
        self.n_sensors = n_sensors
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.one_sided = one_sided
        self.gmm_list = [None] * n_sensors
        self.compute_cdf = np.vectorize(_compute_cdf)
        self.g_scale = None
        self.g_shape = None
        self.weights = [1.0 / n_sensors] * n_sensors  # 均等权重

    def fit(self, errors):
        """
        拟合 GMM 模型
        errors: [N, D] 数组, N=样本数, D=传感器数
        """
        combined = 0
        for i in range(self.n_sensors):
            x = errors[:, i].reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type
            )
            gmm.fit(x)
            self.gmm_list[i] = gmm

            # 计算训练集的 Fisher 统计量
            cdf = self.compute_cdf(gmm, x.flatten())
            fisher, _ = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher

        # 拟合 Gamma 分布
        if np.var(combined) > 0:
            self.g_scale = np.var(combined) / np.mean(combined)
            self.g_shape = np.mean(combined) ** 2 / np.var(combined)
        else:
            LOGGER.warning('No variance in Fisher statistics, using default params.')
            self.g_scale = 1
            self.g_shape = 1

    def compute_anomaly_scores(self, errors):
        """
        计算异常分数
        返回: gamma_p_val, sensor_p_vals, fisher_combined
        """
        combined = 0
        sensor_p_vals = np.zeros_like(errors)
        fisher_values = np.zeros_like(errors)

        for i in range(self.n_sensors):
            y = errors[:, i]
            gmm = self.gmm_list[i]
            cdf = self.compute_cdf(gmm, y)
            fisher, p_val = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher
            sensor_p_vals[:, i] = p_val
            fisher_values[:, i] = fisher

        # Gamma 分布的 p-value (最终异常分数)
        gamma_p_val = 1 - gamma.cdf(combined, a=self.g_shape, scale=self.g_scale)
        return gamma_p_val, sensor_p_vals, combined


# ==================== 6. 训练和推理函数 (整合逻辑) ====================
def training_step(model, schedule, C_batch, Y0_batch, optimizer, device):
    """训练步骤 (保持不变)"""
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
def generate_predictions(model, schedule, C, device):
    """
    使用 Diffusion Model 生成预测
    返回完整的去噪轨迹
    """
    model.eval()
    B, H, D = C.size(0), 20, C.size(2)  # H=20 (预测长度)
    Y_t = torch.randn(B, H, D, device=device)
    self_cond = torch.zeros_like(Y_t)

    for t_scalar in reversed(range(schedule.num_steps)):
        t = torch.full((B,), t_scalar, device=device).long()
        eps_pred = model(C, Y_t, t, self_cond)

        # 更新 self_cond
        alpha_bar_t = schedule.alpha_bars[t].view(B, 1, 1)
        x0_pred = (Y_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        self_cond = x0_pred.detach()

        Y_t = schedule.p_sample_step(Y_t, t_scalar, eps_pred)

    return Y_t


def calibrate_gmm_detector(model, schedule, loader, device, error_type='point'):
    """
    在验证集上校准 GMM 检测器
    数学逻辑:
    1. 生成所有样本的预测
    2. 计算误差 (Point 或 Area)
    3. 用 GMM 拟合误差分布
    4. 计算 Fisher 统计量的 99.5 百分位作为阈值
    """
    print("\n[GMM Calibration] Computing errors on validation set...")
    all_errors = []

    for i, (C, Y) in enumerate(loader):
        C, Y = C.to(device), Y.to(device)
        Y_pred = generate_predictions(model, schedule, C, device)

        # 转为 numpy 计算误差
        y_true = Y.cpu().numpy()  # [B, H, D]
        y_pred = Y_pred.cpu().numpy()

        # 展平时间维度: [B*H, D]
        y_true_flat = y_true.reshape(-1, y_true.shape[-1])
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

        if error_type == 'point':
            errors = point_errors(y_true_flat, y_pred_flat)
        else:
            errors = area_errors(y_true_flat, y_pred_flat)

        all_errors.append(errors)
        if i % 20 == 0:
            print(f"  Processed batch {i}/{len(loader)}")

    all_errors = np.vstack(all_errors)

    # 拟合 GMM
    print("[GMM Calibration] Fitting GMM models...")
    n_sensors = all_errors.shape[1]
    gmm_detector = GMMDetector(
        n_sensors=n_sensors,
        n_components=1,
        covariance_type='spherical',
        one_sided=(error_type == 'point')
    )
    gmm_detector.fit(all_errors)

    # 计算阈值
    _, _, fisher_combined = gmm_detector.compute_anomaly_scores(all_errors)
    threshold = np.percentile(fisher_combined, 99.5)

    print(f"[GMM Calibration] Threshold (99.5 percentile): {threshold:.4f}")
    return gmm_detector, threshold


# ==================== 7. 主流程 ====================
def main_integrated():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数配置
    L, H = 50, 20
    D_cond, D_target = 10, 10
    train_epochs = 30
    error_type = 'point'  # 'point' 或 'area'

    # 文件路径
    train_files = glob.glob('data/alfa/train/**/mavros-imu-data.csv', recursive=True)
    test_files = glob.glob('data/alfa/test/**/mavros-imu-data.csv', recursive=True)

    if not train_files:
        print("Error: No train files found.")
        return

    # 数据集
    print("Loading Data...")
    train_dataset = ALFADataset(train_files, mode='train', L=L, H=H)
    scaler = train_dataset.get_scaler()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 初始化模型
    cfg = DiffusionConfig(num_steps=1000)
    schedule = GaussianDiffusionSchedule(cfg).to(device)
    model = UAVDiffusionModel(D_cond, D_target, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Phase 1: 训练 Diffusion Model
    print("\n>>> Phase 1: Training Diffusion Model...")
    if os.path.exists("uav_diffusion_m2ad.pth"):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load("uav_diffusion_m2ad.pth"))
    else:
        for epoch in range(train_epochs):
            total_loss = 0
            for C, Y in train_loader:
                loss = training_step(model, schedule, C, Y, optimizer, device)
                total_loss += loss
            print(f"Epoch {epoch + 1}/{train_epochs}, Loss: {total_loss / len(train_loader):.6f}")
        torch.save(model.state_dict(), "uav_diffusion_m2ad.pth")

    model.eval()

    # Phase 2: 校准 GMM 检测器
    calib_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    gmm_detector, fisher_threshold = calibrate_gmm_detector(
        model, schedule, calib_loader, device, error_type=error_type
    )

    # Phase 3: 测试
    print("\n>>> Phase 3: Testing with M2AD Anomaly Detection...")
    if not test_files:
        print("No test files found.")
        return

    test_file = test_files[0]
    print(f"Testing on: {test_file}")
    test_dataset = ALFADataset([test_file], mode='test', scaler=scaler, L=L, H=H)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_gamma_scores = []
    all_fisher_scores = []

    with torch.no_grad():
        for i, (C, Y) in enumerate(test_loader):
            C, Y = C.to(device), Y.to(device)
            Y_pred = generate_predictions(model, schedule, C, device)

            y_true = Y.cpu().numpy().reshape(-1, D_target)
            y_pred = Y_pred.cpu().numpy().reshape(-1, D_target)

            if error_type == 'point':
                errors = point_errors(y_true, y_pred)
            else:
                errors = area_errors(y_true, y_pred)

            gamma_scores, _, fisher_scores = gmm_detector.compute_anomaly_scores(errors)
            all_gamma_scores.extend(gamma_scores)
            all_fisher_scores.extend(fisher_scores)

            if i % 100 == 0:
                print(f"Step {i}/{len(test_loader)}")

    # Phase 4: 评估
    print("\n>>> Phase 4: Evaluation with M2AD Metrics...")

    # 平滑
    smoothed_fisher = pd.Series(all_fisher_scores).rolling(window=5, min_periods=1).mean().values

    # 预测异常
    y_pred = (smoothed_fisher > fisher_threshold).astype(int)

    # Ground Truth (需要根据实际情况调整)
    y_true = np.zeros(len(all_fisher_scores))
    failure_start = 100  # 根据数据调整
    if "failure" in test_file or "carbon" in test_file:
        y_true[failure_start:] = 1

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 50)
    print("🎯 Diffusion-M2AD Performance Report")
    print("=" * 50)
    print(f"Error Type: {error_type.upper()}")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 50)

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_fisher_scores, alpha=0.3, label='Raw Fisher Score')
    plt.plot(smoothed_fisher, linewidth=2, label='Smoothed')
    plt.axhline(fisher_threshold, color='red', linestyle='--', label='Threshold')
    if np.sum(y_true) > 0:
        plt.axvspan(failure_start, len(y_true), color='red', alpha=0.1, label='True Failure')
    plt.title('M2AD Fisher Statistic')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(all_gamma_scores, label='Gamma p-value')
    plt.axhline(0.001, color='red', linestyle='--', label='p < 0.001')
    plt.title('M2AD Gamma Anomaly Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('diffusion_m2ad_results.png', dpi=150)
    plt.show()
    print("\nResults saved to 'diffusion_m2ad_results.png'")


if __name__ == "__main__":
    main_integrated()