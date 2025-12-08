# -*- coding: utf-8 -*-
"""
P-Gpt.py

UAV anomaly detection with:
- Diffusion + Transformer backbone (from p-gemini.py)
- M2AD-style multi-sensor error modelling, GMM scoring, and Gamma threshold.

This version:
- Uses AREA errors (integral in a sliding window) to capture local energy changes.
- Plots ROC / PR curves per sensor (10 IMU channels).
"""

import math
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from dataclasses import dataclass

# ======== M2AD statistical components ========
from statistics import NormalDist
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture
from functools import partial
from itertools import compress


# ==================== 1. 数据集定义 ====================
class ALFADataset(Dataset):
    def __init__(self, file_paths, mode='train', scaler=None, L=50, H=20):
        self.L = L
        self.H = H
        # 10 维传感器特征，每一维视为一个独立传感器
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

        C_seq = self.data[idx: hist_end]         # [L, D]
        Y_seq = self.data[hist_end: future_end]  # [H, D]
        return C_seq, Y_seq

    def get_scaler(self):
        return self.scaler


# ==================== 2. Diffusion 基础组件 ====================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
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

        # 预测 x0
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # 后验均值系数
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


# ==================== 3. Transformer + Diffusion 架构 ====================
class CondEncoder(nn.Module):
    def __init__(self, d_in, d_model, num_layers=2, num_heads=4,
                 dim_ff=256, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

    def forward(self, cond_seq):
        x = self.input_proj(cond_seq)
        x = self.pos_enc(x)
        h = self.encoder(x)
        return h  # [B, L, d_model]


class FutureBackbone(nn.Module):
    def __init__(self, d_model, num_layers=4, num_heads=4,
                 dim_ff=512, max_len=256):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=dim_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

    def forward(self, x):
        x = self.pos_enc(x)
        return self.encoder(x)


class UAVDiffusionModel(nn.Module):
    def __init__(self, D_cond, D_target, diffusion_cfg,
                 d_model_cond=128, d_model_future=128):
        super().__init__()
        self.D_cond = D_cond
        self.D_target = D_target

        self.cond_encoder = CondEncoder(D_cond, d_model_cond)
        self.time_embed = TimeEmbedding(diffusion_cfg.num_steps,
                                        d_model_future)

        # 输入: [Y_t, self_cond] 维度翻倍
        self.future_proj = nn.Linear(D_target * 2, d_model_future)

        # Cross-Attention: future query history
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model_future, num_heads=4, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model_future)

        self.future_backbone = FutureBackbone(d_model=d_model_future)
        self.out_proj = nn.Linear(d_model_future, D_target)

    def forward(self, C, Y_t, t, self_cond=None):
        # 1. 编码历史 -> [B, L, d]
        cond_seq = self.cond_encoder(C)

        # 2. Self-Conditioning
        if self_cond is None:
            self_cond = torch.zeros_like(Y_t)

        # 3. 拼接未来输入
        x_in = torch.cat([Y_t, self_cond], dim=-1)  # [B, H, 2D]
        future_emb = self.future_proj(x_in)         # [B, H, d]
        t_emb = self.time_embed(t).unsqueeze(1)     # [B, 1, d]
        query = future_emb + t_emb                  # [B, H, d]

        # 4. Cross-Attention
        attn_out, _ = self.cross_attn(query, cond_seq, cond_seq)
        x = self.norm1(query + attn_out)

        # 5. Transformer backbone & 输出噪声预测
        h = self.future_backbone(x)
        return self.out_proj(h)


# ==================== 4. Diffusion 训练 / 采样 ====================
def predict_x0_from_xt(schedule, xt, eps_pred, t):
    """从 x_t 反推 x_0"""
    B = xt.size(0)
    alpha_bar_t = schedule.alpha_bars.to(xt.device)[t].view(B, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    return (xt - sqrt_one_minus * eps_pred) / sqrt_alpha_bar


def training_step(model, schedule, C_batch, Y0_batch, optimizer, device):
    """标准 DDPM 训练步 + Self-conditioning"""
    model.train()
    C_batch = C_batch.to(device)
    Y0_batch = Y0_batch.to(device)
    B = Y0_batch.size(0)

    t = torch.randint(0, schedule.num_steps, (B,), device=device).long()
    eps = torch.randn_like(Y0_batch)
    Y_t = schedule.q_sample(Y0_batch, t, eps)

    # 50% 概率使用真实 Y0 作为 self-cond
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
    """
    RePaint 风格的未来轨迹生成/补全。
    - 如果 mask = 0，全维度由 Diffusion 生成 (只依赖 C)；
    - 如果 mask = 1，对应位置强制保持观测 Y_obs。
    """
    model.eval()
    B, H, D = Y_obs.shape
    Y_t = torch.randn_like(Y_obs)
    self_cond = torch.zeros_like(Y_obs)

    for t_scalar in reversed(range(schedule.num_steps)):
        t = torch.full((B,), t_scalar, device=device).long()

        # 噪声预测
        eps_pred = model(C, Y_t, t, self_cond)

        # 更新 self-cond
        x0_pred = predict_x0_from_xt(schedule, Y_t, eps_pred, t)
        self_cond = x0_pred.detach()

        # 反向采样一步
        Y_prev = schedule.p_sample_step(Y_t, t_scalar, eps_pred)

        # RePaint：已知位置强制注入
        if t_scalar > 0:
            noise = torch.randn_like(Y_obs)
            t_prev = torch.full((B,), t_scalar - 1, device=device).long()
            Y_obs_t = schedule.q_sample(Y_obs, t_prev, noise)
        else:
            Y_obs_t = Y_obs

        Y_t = mask * Y_obs_t + (1.0 - mask) * Y_prev

    return Y_t


# ==================== 5. M2AD 风格多传感器统计模块 ====================
def _get_sum(name, sensors):
    return sum([1 for sensor in sensors if name in sensor])


def _divide(x, y):
    return x / y if y else 0.0


def _get_default(sensors):
    # 对每个传感器统一权重 1
    return {sensor: 1 for sensor in sensors}


def _find_weights(sensors, prefix=None) -> list:
    prefix = prefix or _get_default(sensors)
    pre_weights = {
        sensor_type: _divide(sensor_weight, _get_sum(sensor_type, sensors))
        for sensor_type, sensor_weight in prefix.items()
    }
    weights = [pre_weights[k] for sensor in sensors for k in pre_weights if k in sensor]
    return weights


def _compute_cdf(gmm, x):
    means = gmm.means_.flatten()
    sigma = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()
    cdf = 0.0
    for i in range(len(means)):
        cdf += weights[i] * NormalDist(mu=means[i], sigma=sigma[i]).cdf(x)
    return cdf


def _combine_pval(cdf, side=True):
    # side=True: 单边；side=False: 双边 (area error 使用)
    if side:
        p_val = 1 - cdf
    else:
        p_val = 2 * np.array(list(map(np.min, zip(1 - cdf, cdf))))
    p_val[p_val < 1e-16] = 1e-16
    fisher_pval = -2 * np.log(p_val)
    return fisher_pval, p_val


class GMM:
    """Multi-sensor Gaussian Mixture with Fisher + Gamma aggregation."""

    def _parse_components(self, n_components, sensors, default=1):
        if sensors is None:
            if isinstance(n_components, dict):
                raise ValueError("Unknown list of sensors but specified in components.")
            elif isinstance(n_components, int):
                return n_components
            return default

        if isinstance(n_components, dict):
            n_components = {
                **n_components,
                **{k: default for k in sensors if k not in n_components},
            }
        elif isinstance(n_components, int):
            n_components = dict(zip(sensors, [n_components] * len(sensors)))
        return n_components

    def __init__(self, sensors, n_components=1, covariance_type='spherical',
                 one_sided=False, weights=None):
        self.sensors = sensors
        self.n_components = self._parse_components(n_components, sensors)
        self.covariance_type = covariance_type
        self.one_sided = one_sided

        self.gmm = [None] * len(self.sensors)
        self.compute_cdf = np.vectorize(_compute_cdf)

        self.g_scale = None
        self.g_shape = None

        self.weights = weights or _find_weights(self.sensors)

    def fit(self, X):
        """
        X: [T, D] 误差矩阵，D = 传感器数 (这里为 10)
        """
        combined = 0.0
        num_sensors = X.shape[1]
        assert num_sensors == len(self.sensors)
        for i, sensor in enumerate(self.sensors):
            x = X[:, i].reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=self.n_components[sensor],
                covariance_type=self.covariance_type
            )
            gmm.fit(x)
            self.gmm[i] = gmm

            cdf = self.compute_cdf(gmm, x.flatten())
            fisher, _ = _combine_pval(cdf, self.one_sided)
            combined += self.weights[i] * fisher

        if np.var(combined) > 0:
            self.g_scale = np.var(combined) / np.mean(combined)
            self.g_shape = (np.mean(combined) ** 2) / np.var(combined)
        else:
            print(f"[GMM] Warning: no variance in combined fisher values ({np.var(combined)}).")
            self.g_scale = 1.0
            self.g_shape = 0.0

    def p_values(self, X):
        """
        输入同维度误差矩阵 X[T, D]，输出：
        - gamma_p_val: [T] Gamma p-value (anomaly score, 越小越异常)
        - p_val_sensors: [T, D] 各传感器 p-value
        - fisher_values: [T, D] 各传感器 Fisher 统计量
        """
        combined = 0.0
        p_val_sensors = np.zeros_like(X)
        fisher_values = np.zeros_like(X)

        for i, sensor in enumerate(self.sensors):
            y = X[:, i]
            gmm = self.gmm[i]
            cdf = self.compute_cdf(gmm, y)
            fisher, p_val = _combine_pval(cdf, self.one_sided)

            combined += self.weights[i] * fisher
            p_val_sensors[:, i] = p_val
            fisher_values[:, i] = fisher

        gamma_p_val = 1 - gamma.cdf(combined, a=self.g_shape, scale=self.g_scale)
        return gamma_p_val, p_val_sensors, fisher_values


def _smooth(errors, smoothing_window):
    smoothed_errors = pd.DataFrame(errors).ewm(smoothing_window).mean().values
    return smoothed_errors


def point_errors(y, pred, smooth=False, smoothing_window=10):
    """
    y, pred: [T, D] (flatten 后的时间序列)
    这里保留 point_errors 以防需要对比实验。
    """
    errors = np.abs(y - pred)
    if smooth:
        errors = _smooth(errors, smoothing_window)
    return np.array(errors)


def area_errors(y, pred, score_window=10, dx=1.0,
                smooth=False, smoothing_window=10):
    """
    滑动积分误差，适合关注局部区域能量变化。
    - 对每个维度的时间序列做滚动窗口积分 (Trapezoidal rule)，再做差值。
    """
    trapz = partial(np.trapz, dx=dx)
    errors = np.empty_like(y)
    num_signals = errors.shape[1]

    for i in range(num_signals):
        area_y = pd.Series(y[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2
        ).apply(trapz)
        area_pred = pd.Series(pred[:, i]).rolling(
            score_window, center=True, min_periods=score_window // 2
        ).apply(trapz)

        error = area_y - area_pred

        if smooth:
            error = _smooth(error, smoothing_window)

        errors[:, i] = error.flatten()

    mu = np.mean(errors)
    std = np.std(errors) + 1e-8
    return (errors - mu) / std


def _merge_sequences(sequences):
    """
    合并相邻异常区间 (按 index 而非真实时间)
    """
    if len(sequences) == 0:
        return np.array([])

    sorted_sequences = sorted(sequences, key=lambda s: s[0])
    new_sequences = [sorted_sequences[0]]
    score = [sorted_sequences[0][2]]

    for sequence in sorted_sequences[1:]:
        prev_sequence = new_sequences[-1]
        if sequence[0] <= prev_sequence[1] + 1:
            score.append(sequence[2])
            average = np.mean(score)
            new_sequences[-1] = (
                prev_sequence[0],
                max(prev_sequence[1], sequence[1]),
                average,
            )
        else:
            score = [sequence[2]]
            new_sequences.append(sequence)

    return np.array(new_sequences)


def create_intervals(anomalies, index, score, anomaly_padding=50):
    """
    根据二值 anomalies 序列 + 分数，合并成异常区间。
    index: 任意可索引序列（这里直接用整数 0..T-1 即可）
    """
    intervals = []
    length = len(anomalies)
    anomalies_index = list(compress(range(length), anomalies))
    for idx in anomalies_index:
        start = max(0, idx - anomaly_padding)
        end = min(idx + anomaly_padding + 1, length)
        value = np.mean(score[start: end])
        intervals.append([index[start], index[end - 1], value])

    intervals = _merge_sequences(intervals)
    return intervals  # shape [K, 3], 每行为 [start_idx, end_idx, avg_score]


# ==================== 6. 与 Diffusion 结合的误差/评分/阈值逻辑 ====================
def collect_diffusion_errors(model, schedule, loader, device,
                             error_name='area',
                             score_window=10,
                             smoothing_window=10):
    """
    用 Diffusion 模型在整个 loader 上生成未来轨迹预测，
    与真实 Y 对比，得到 [T, D] 的误差矩阵（按时间展开）。
    """
    all_y = []
    all_pred = []

    model.eval()
    with torch.no_grad():
        for C, Y_true in loader:
            C = C.to(device)           # [B, L, D]
            Y_true = Y_true.to(device) # [B, H, D]

            # 这里不使用观测值，做「纯预测」：Y_obs=0, mask=0 表示全维由 Diffusion 生成
            Y_obs = torch.zeros_like(Y_true)
            mask = torch.zeros_like(Y_true)

            Y_pred = impute_future_trajectory(model, schedule, C, Y_obs, mask, device)

            # 展开到 [B*H, D]，拼接到全局时间序列
            B, H, D = Y_true.shape
            all_y.append(Y_true.reshape(B * H, D).cpu().numpy())
            all_pred.append(Y_pred.reshape(B * H, D).cpu().numpy())

    y = np.concatenate(all_y, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    if error_name == 'point':
        errors = point_errors(y, pred, smooth=True, smoothing_window=smoothing_window)
    elif error_name == 'area':
        errors = area_errors(y, pred, score_window=score_window,
                             smooth=True, smoothing_window=smoothing_window)
    else:
        raise ValueError(f"Unknown error_name={error_name}")

    return y, pred, errors


def fit_gmm_on_diffusion_errors(model, schedule, loader, device,
                                sensors, error_name='area',
                                n_components=1,
                                covariance_type='spherical',
                                score_window=10,
                                smoothing_window=10):
    """
    在训练/校准数据上：
      1) 用 Diffusion 做未来预测；
      2) 计算 AREA 误差矩阵；
      3) 对每个传感器建一维 GMM；
      4) 拟合 Gamma 聚合分布，并给出 Fisher 阈值（诊断用）。
    """
    print("\n[Calibration] Collecting diffusion prediction errors for GMM fitting...")
    _, _, errors = collect_diffusion_errors(
        model, schedule, loader, device,
        error_name=error_name,
        score_window=score_window,
        smoothing_window=smoothing_window
    )
    print(f"[Calibration] Error matrix shape: {errors.shape} (T, D)")

    # area 误差使用双侧检验
    one_sided = (error_name == 'point')
    gmm_model = GMM(
        sensors=sensors,
        n_components=n_components,
        covariance_type=covariance_type,
        one_sided=one_sided
    )
    gmm_model.fit(errors)
    print("[Calibration] GMM fitted for all sensors.")

    gamma_p_val, _, fisher_values = gmm_model.p_values(errors)
    fisher_scalar = np.sum(fisher_values, axis=1)
    fisher_threshold = np.percentile(fisher_scalar, 99.5)
    print(f"[Calibration] 99.5% Fisher threshold (diagnostic only): {fisher_threshold:.4f}")

    return gmm_model, fisher_threshold, (errors, gamma_p_val, fisher_values)


def detect_with_diffusion_gmm(model, schedule, loader, device,
                              gmm_model,
                              error_name='area',
                              gamma_thresh=1e-3,
                              score_window=10,
                              smoothing_window=10):
    """
    在测试数据上检测异常：
      1) Diffusion 预测未来；
      2) 计算 AREA 误差矩阵；
      3) 通过 GMM+Gamma 得到 p-value；
      4) 应用 gamma_thresh 得到异常标签；
      5) 返回时间序列级别的分数和异常区间。
    """
    print("\n[Detection] Collecting diffusion prediction errors on test data...")
    y, pred, errors = collect_diffusion_errors(
        model, schedule, loader, device,
        error_name=error_name,
        score_window=score_window,
        smoothing_window=smoothing_window
    )

    print("[Detection] Computing Gamma p-values via GMM...")
    gamma_p_val, p_val_sensors, fisher_values = gmm_model.p_values(errors)

    anomalies_bool = gamma_p_val < gamma_thresh
    indices = np.arange(len(gamma_p_val))

    intervals = create_intervals(
        anomalies=anomalies_bool,
        index=indices,
        score=gamma_p_val,
        anomaly_padding=50
    )

    print(f"[Detection] Total time steps: {len(gamma_p_val)}, "
          f"anomalies: {anomalies_bool.sum()}")

    results = {
        "y": y,
        "pred": pred,
        "errors": errors,
        "gamma_p_val": gamma_p_val,
        "p_val_sensors": p_val_sensors,
        "fisher_values": fisher_values,
        "anomalies_bool": anomalies_bool,
        "intervals": intervals,
    }
    return results


# ==================== 7. 主流程：训练 / 校准 / 测试 ====================
def main_alfa():
    device = torch.device('xpu' if torch.xpu.is_available()
                          else 'cuda' if torch.cuda.is_available()
                          else 'cpu')
    print(f"Using device: {device}")

    # 序列长度
    L, H = 50, 20
    D_cond, D_target = 10, 10
    train_epochs = 30

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

    if not train_files:
        print("Error: No train files found.")
        return
    if not test_files:
        print("Warning: No test files found.")
        return

    # ---------- 数据集 ----------
    print("Loading Data.")
    train_dataset = ALFADataset(train_files, mode='train', L=L, H=H)
    scaler = train_dataset.get_scaler()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0)

    # 测试用单文件（也可以遍历所有 test_files）
    test_file = test_files[0]
    print(f"Testing on: {test_file}")
    test_dataset = ALFADataset([test_file], mode='test', scaler=scaler,
                               L=L, H=H)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=0)

    # ---------- 模型与 Diffusion ----------
    cfg = DiffusionConfig(num_steps=1000)
    schedule = GaussianDiffusionSchedule(cfg).to(device)
    model = UAVDiffusionModel(D_cond, D_target, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---------- 训练 or 加载 ----------
    model_path = "uav_imputation_model.pth"

    if os.path.exists(model_path):
        print("\n>>> Phase 1: Loading pre-trained Diffusion model.")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded '{model_path}'.")
    else:
        print("\n>>> Phase 1: Training Diffusion model from scratch.")
        for epoch in range(train_epochs):
            total_loss = 0.0
            for C_batch, Y_batch in train_loader:
                loss = training_step(model, schedule, C_batch, Y_batch,
                                     optimizer, device)
                total_loss += loss
            print(f"Epoch {epoch + 1}/{train_epochs}, "
                  f"Avg Loss: {total_loss / len(train_loader):.6f}")

        torch.save(model.state_dict(), model_path)
        print(f"Saved trained model to '{model_path}'.")

    model.eval()

    # ---------- Phase 2: 使用 M2AD 风格 AREA 误差进行阈值标定 ----------
    print("\n>>> Phase 2: Calibrating GMM + Gamma on normal-like train data.")
    calib_loader = DataLoader(train_dataset, batch_size=32, shuffle=False,
                              num_workers=0)

    # 这里切换为 area 误差
    error_name = 'area'
    gamma_thresh = 1e-3  # 完全沿用 M2AD 的 gamma 阈值

    gmm_model, fisher_thresh, calib_stats = fit_gmm_on_diffusion_errors(
        model=model,
        schedule=schedule,
        loader=calib_loader,
        device=device,
        sensors=train_dataset.features,  # 10 个传感器
        error_name=error_name,
        n_components=1,
        covariance_type='spherical',
        score_window=10,
        smoothing_window=10,
    )
    print(f"[Threshold] Using gamma_thresh={gamma_thresh:.1e} as anomaly threshold on p-values.")

    # ---------- Phase 3: 测试检测 ----------
    print("\n>>> Phase 3: Detection on test file.")
    det_results = detect_with_diffusion_gmm(
        model=model,
        schedule=schedule,
        loader=test_loader,
        device=device,
        gmm_model=gmm_model,
        error_name=error_name,
        gamma_thresh=gamma_thresh,
        score_window=10,
        smoothing_window=10,
    )

    gamma_p_val = det_results["gamma_p_val"]
    anomalies_bool = det_results["anomalies_bool"]
    p_val_sensors = det_results["p_val_sensors"]

    # 将 p-value 转成「分数」，便于可视化：score = -log10(p)
    scores_global = -np.log10(gamma_p_val + 1e-16)
    score_threshold = -np.log10(gamma_thresh)

    # --------- 构造 Ground Truth（和原 p-gemini 简单逻辑保持一致）---------
    num_steps = len(scores_global)
    y_true = np.zeros(num_steps, dtype=int)

    # 需要根据具体文件调整 failure_start_index
    failure_start_index = 100
    if "failure" in test_file or "carbon" in test_file:
        y_true[failure_start_index:] = 1
        print(f"[Ground Truth] Failure marked starting at index {failure_start_index}")
    else:
        print("[Ground Truth] Normal flight assumed (no failure label in filename).")

    y_pred = anomalies_bool.astype(int)

    # ---------- Overall 指标 ----------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("Global Performance Report (M2AD-area scoring on Diffusion backbone)")
    print("=" * 60)
    print(f"Accuracy  : {acc:.2%}")
    print(f"Precision : {prec:.2%}")
    print(f"Recall    : {rec:.2%}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 60)

    # ---------- 全局分数可视化 ----------
    window_size = 5
    smoothed_scores = pd.Series(scores_global).rolling(
        window=window_size, min_periods=1
    ).mean().values

    plt.figure(figsize=(12, 6))
    plt.plot(scores_global, label='Raw Score (-log10 p_gamma)')
    plt.plot(smoothed_scores, linewidth=2,
             label=f'Smoothed Score (MA={window_size})')
    plt.axhline(score_threshold, linestyle='--',
                label=f'Threshold (-log10 {gamma_thresh:.0e})')

    # 标记真实故障区域
    if np.sum(y_true) > 0:
        plt.axvspan(failure_start_index, num_steps, alpha=0.1,
                    label='Ground Truth Failure')

    plt.title(f"Diffusion+M2AD-area Detection: F1={f1:.3f} | Recall={rec:.3f}")
    plt.xlabel("Time Index (sliding windows)")
    plt.ylabel("-log10 Gamma p-value")
    plt.legend()
    plt.tight_layout()
    plt.savefig('final_result_p-gpt_m2ad_area.png')
    plt.show()
    print("Global visualization saved as 'final_result_p-gpt_m2ad_area.png'.")

    # ---------- 每个传感器单独的 ROC / PR 曲线 ----------
    print("\n>>> Per-sensor ROC / PR analysis (based on sensor p-values).")
    # 传感器级分数：score_sensor = -log10(p_val_sensor)
    sensor_scores = -np.log10(p_val_sensors + 1e-16)  # [T, D]

    for i, sensor in enumerate(train_dataset.features):
        s = sensor_scores[:, i]

        # ROC
        fpr, tpr, _ = roc_curve(y_true, s)
        roc_auc = auc(fpr, tpr)

        # PR
        precision_s, recall_s, _ = precision_recall_curve(y_true, s)
        ap = average_precision_score(y_true, s)

        print(f"\n[Sensor: {sensor}] ROC-AUC={roc_auc:.4f}, AP={ap:.4f}")

        plt.figure(figsize=(10, 4))

        # ROC 曲线
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {sensor}')
        plt.legend()

        # PR 曲线
        plt.subplot(1, 2, 2)
        plt.plot(recall_s, precision_s, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR - {sensor}')
        plt.legend()

        plt.tight_layout()
        fname = f'roc_pr_{sensor.replace(".", "_")}.png'
        plt.savefig(fname)
        plt.close()
        print(f"Saved per-sensor ROC/PR figure to '{fname}'.")


if __name__ == "__main__":
    main_alfa()
