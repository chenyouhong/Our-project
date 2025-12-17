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
import itertools
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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score
# ======== M2AD statistical components ========
from statistics import NormalDist
from scipy.stats import gamma
from sklearn.mixture import GaussianMixture
from functools import partial
from itertools import compress
from __future__ import annotations
import sys

THIS_FILE = Path(__file__).resolve()
# P-Gpt.py 在 D:\project\our\newproject\GPT
# parents[0] = GPT, parents[1] = newproject, parents[2] = our
ROOT_DIR = THIS_FILE.parents[2]   # D:\project\our

# 把 D:\project\our 加到 sys.path 里
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 现在就可以 from MTCL_UAV... 了
from MTCL_UAV.layers.MoE import MoE
from MTCL_UAV.layers.RevIN import RevIN
from tqdm import tqdm

from typing import Optional
# ======================================================================
# 映射：测试集中每个 flight -> 对应的故障 IMU 行号（从 0 开始）
# 你需要根据 ALFA 官方工具 / 自己整理的标注，把下面的字典填完整。
# key 可以是文件名的一部分（子串匹配），value 是整数 fault_row。
# ======================================================================
FAULT_ROW_DICT = {
    "carbonZ_2018-07-18-15-53-31_1_engine_failure": 286,
    "carbonZ_2018-07-18-15-53-31_2_engine_failure": 187,
    "carbonZ_2018-07-18-16-22-01_engine_failure_with_emr_traj": 288,
    "carbonZ_2018-07-18-16-37-39_2_engine_failure_with_emr_traj": 279,
    "carbonZ_2018-07-30-16-29-45_engine_failure_with_emr_traj": 311,
    "carbonZ_2018-07-30-16-39-00_1_engine_failure": 294,
    "carbonZ_2018-07-30-16-39-00_2_engine_failure": 234,
    "carbonZ_2018-07-30-17-10-45_engine_failure_with_emr_traj": 300,
    "carbonZ_2018-07-30-17-20-01_engine_failure_with_emr_traj": 218,
    "carbonZ_2018-07-30-17-36-35_engine_failure_with_emr_traj": 330,
    "carbonZ_2018-07-30-17-46-31_engine_failure_with_emr_traj": 224,
    "carbonZ_2018-09-11-11-56-30_engine_failure": 264,
    "carbonZ_2018-09-11-14-22-07_1_engine_failure": 256,
    "carbonZ_2018-09-11-14-22-07_2_engine_failure": 124,
    "carbonZ_2018-09-11-14-41-51_elevator_failure": 299,
    "carbonZ_2018-09-11-14-52-54_left_aileron__right_aileron__failure": 256,
    "carbonZ_2018-09-11-15-05-11_1_elevator_failure": 164,
    "carbonZ_2018-09-11-15-06-34_1_rudder_right_failure": 140,
    "carbonZ_2018-09-11-15-06-34_2_rudder_right_failure": 133,
    "carbonZ_2018-09-11-15-06-34_3_rudder_left_failure": 155,
    "carbonZ_2018-09-11-17-27-13_1_rudder_zero__left_aileron_failure": 299,
    "carbonZ_2018-09-11-17-27-13_2_both_ailerons_failure": 169,
    "carbonZ_2018-09-11-17-55-30_1_right_aileron_failure": 287,
    "carbonZ_2018-09-11-17-55-30_2_left_aileron_failure": 125,
    "carbonZ_2018-10-05-14-34-20_2_right_aileron_failure_with_emr_traj": 383,
    "carbonZ_2018-10-05-14-37-22_2_right_aileron_failure": 191,
    "carbonZ_2018-10-05-14-37-22_3_left_aileron_failure": 188,
    "carbonZ_2018-10-05-15-52-12_3_engine_failure_with_emr_traj": 128,
    "carbonZ_2018-10-05-15-55-10_engine_failure_with_emr_traj": 257,
    "carbonZ_2018-10-05-16-04-46_engine_failure_with_emr_traj": 193,
    "carbonZ_2018-10-18-11-03-57_engine_failure_with_emr_traj": 269,
    "carbonZ_2018-10-18-11-04-00_engine_failure_with_emr_traj": 284,
    "carbonZ_2018-10-18-11-04-08_1_engine_failure_with_emr_traj": 253,
    "carbonZ_2018-10-18-11-04-08_2_engine_failure_with_emr_traj": 248,
    "carbonZ_2018-10-18-11-04-35_engine_failure_with_emr_traj": 256,
    "carbonZ_2018-10-18-11-06-06_engine_failure_with_emr_traj": 266,
}


def get_fault_row_for_file(file_path: Path):
    """根据文件名子串在 FAULT_ROW_DICT 中查找故障行号。找不到则返回 None。"""
    s = str(file_path)
    for key, row in FAULT_ROW_DICT.items():
        if key in s:
            return row
    return None


# ==================== 1. 数据集定义 ====================
# ==================== 1. 数据集定义（支持分类标签） ====================
class ALFADataset(Dataset):
    def __init__(
        self,
        file_paths,
        mode: str = 'train',
        scaler: StandardScaler = None,
        L: int = 50,
        H: int = 20,
        return_label: bool = False,
    ):
        """
        file_paths : 一组 mavros-imu-data.csv 的路径
        mode       : 'train' / 'test'，这里只用于日志，缩放逻辑只看 scaler 是否为 None
        scaler     : 若为 None，则在 full_data 上 fit；否则直接使用传入的 scaler
        L, H       : 历史长度和预测长度
        return_label:
            False -> __getitem__ 返回 (C_seq, Y_seq)
            True  -> __getitem__ 返回 (C_seq, Y_seq, y_cls)，其中 y_cls 为标量 0/1
        """
        self.L = L
        self.H = H
        self.return_label = return_label

        # 10 维传感器特征，每一维视为一个独立传感器
        self.features = [
            'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
            'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
            'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'
        ]

        self.file_paths = [Path(p) for p in file_paths]

        # 先把所有 flight 的原始数据读出来
        raw_data_list = []
        for file in self.file_paths:
            df = pd.read_csv(file)
            if not set(self.features).issubset(df.columns):
                continue
            data_chunk = df[self.features].values.astype(np.float32)
            raw_data_list.append(data_chunk)

        if not raw_data_list:
            raise ValueError("No valid data found in file paths.")

        full_data = np.concatenate(raw_data_list, axis=0)

        # 统一用 full_data 拟合/复用 scaler，保证与之前逻辑一致
        if scaler is None:
            # 仅在第一次构造（训练集）时拟合
            self.scaler = StandardScaler()
            self.scaler.fit(full_data)
        else:
            self.scaler = scaler

        # 对每个 flight 单独做 transform，保留边界，用于构造「不跨 flight」的窗口
        self.seqs = []          # list of [N_f, D] 的 tensor，每个 flight 一条
        self.lengths = []       # 每个 flight 的长度
        for data_chunk in raw_data_list:
            norm_chunk = self.scaler.transform(data_chunk)
            tensor_chunk = torch.tensor(norm_chunk, dtype=torch.float32)
            self.seqs.append(tensor_chunk)
            self.lengths.append(tensor_chunk.shape[0])

        # 为每个 flight 构造滑动窗口起点 (file_idx, start_row_local)，不跨 flight
        self.samples = []       # list of (file_idx, start_row_local)
        for f_idx, N in enumerate(self.lengths):
            max_start = N - (self.L + self.H)
            if max_start <= 0:
                continue
            for start in range(max_start):
                self.samples.append((f_idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回:
          return_label=False:  C_seq [L,D], Y_seq [H,D]
          return_label=True :  C_seq [L,D], Y_seq [H,D], y_cls [标量 0/1]
        """
        file_idx, start = self.samples[idx]
        seq = self.seqs[file_idx]               # [N_f, D]
        hist_end = start + self.L
        future_end = hist_end + self.H

        C_seq = seq[start: hist_end]           # [L, D]
        Y_seq = seq[hist_end: future_end]      # [H, D]

        if not self.return_label:
            return C_seq, Y_seq

        # ---------- 计算该窗口的二分类标签 ----------
        file_path = self.file_paths[file_idx]
        fault_row = get_fault_row_for_file(file_path)

        if fault_row is None:
            # 没有标注的 flight，一律当作正常（纯负样本）
            label = 0.0
        else:
            # 窗口未来 H 步对应的原始行号（在该 flight 内是局部 0-based）
            future_rows = np.arange(hist_end, future_end)
            # 只要未来窗口中出现任何 >= fault_row 的行号，就标记为正
            label = float(np.any(future_rows >= fault_row))

        y_cls = torch.tensor(label, dtype=torch.float32)

        return C_seq, Y_seq, y_cls

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


@torch.no_grad()
def ddim_sample(
        model,
        schedule,
        C,
        shape,
        device,
        steps=50,
        eta=0.0
):
    """
    DDIM Accelerated Sampling
    Args:
        model: UAVDiffusionModel
        schedule: GaussianDiffusionSchedule instance
        C: Condition (History) [B, L, D]
        shape: Output shape (B, H, D)
        device: torch device
        steps: Sampling steps (e.g., 50)
        eta: 0.0 for deterministic DDIM
    Returns:
        Y_pred: Predicted future trajectory [B, H, D]
    """
    model.eval()
    B, H, D = shape

    # 1. 生成跳步时间序列 (e.g., [999, 979, ..., 0])
    total_steps = schedule.num_steps
    times = torch.linspace(0, total_steps - 1, steps=steps).long().to(device)
    times = torch.flip(times, [0])  # 倒序

    # 2. 初始化噪声
    img = torch.randn(shape, device=device)

    # Self-conditioning placeholder
    self_cond = torch.zeros_like(img)

    # 3. DDIM Loop
    # 使用 tqdm 显示进度 (可选，为了速度可移除 tqdm)
    for i, step in enumerate(times):
        # 当前时间步 t
        t = torch.full((B,), step, device=device, dtype=torch.long)

        # 下一个时间步 t_prev (即 t-1 在子序列中的位置)
        prev_step = times[i + 1] if i < len(times) - 1 else -1

        # 获取 alpha_bar 参数
        alpha_bar_t = schedule.alpha_bars[step]
        alpha_bar_prev = schedule.alpha_bars[prev_step] if prev_step >= 0 else torch.tensor(1.0, device=device)

        # 4. 模型预测噪声 epsilon
        # 注意：P-Gpt 模型 forward 需要 (C, Y_t, t, self_cond)
        noise_pred = model(C, img, t, self_cond)

        # 5. 预测 x0 (Predicted Clean Data)
        pred_x0 = (img - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # 更新 self_cond (如果模型训练时用了 self-conditioning)
        self_cond = pred_x0.detach()

        # 6. 计算方向 (Direction pointing to x_t)
        sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
        # 对于 DDIM (eta=0)，sigma_t 为 0

        pred_dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t ** 2) * noise_pred

        # 7. 更新 x_{t-1}
        noise = torch.randn_like(img) if sigma_t > 0 else 0.
        img = torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir_xt + sigma_t * noise

    return img
# ==================== 3. Transformer + Diffusion 架构 ====================
class AMSCondEncoder(nn.Module):
    """
    条件编码器：使用 MTCL_UAV 的 AMS 结构 (RevIN + 多层 MoE) 对历史序列 C 进行特征提取。

    输入:
        C : [B, L, D]  (L=历史长度, D=传感器数=10)

    输出:
        cond_seq       : [B, L, d_model]  作为 Diffusion 的 cross-attn key/value
        balance_loss   : 标准 MoE load-balance 正则 (累加所有层)
        contrast_loss  : AMS 中所有层的对比学习损失平均值
    """
    def __init__(
        self,
        num_nodes: int,
        seq_len: int,
        d_model: int = 128,
        d_ff: int = 256,
        layer_nums: int = 2,
        k: int = 2,
        num_experts_list=None,
        patch_size_list=None,
        residual_connection: bool = True,
        use_revin: bool = True,
        batch_norm: bool = True,
        temp: float = 2.0,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.num_nodes = num_nodes      # = D_cond = 10
        self.seq_len = seq_len          # = L
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_nums = layer_nums
        self.k = k
        self.residual_connection = int(residual_connection)
        self.batch_norm = batch_norm
        self.temp = temp
        self.device = device

        # 1) RevIN: 按 feature 维度做归一化（与 MTCL 一致）
        self.revin = RevIN(
            num_features=self.num_nodes,
            affine=False,
            subtract_last=False
        ) if use_revin else None

        # 2) 起始线性层: 把每个传感器的一维值映射到 d_model
        #    输入形状为 [B, L, D, 1] -> 输出 [B, L, D, d_model]
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)

        # 3) AMS: 多层 MoE，每层上面挂多个 “patch 尺度” 的专家
        if num_experts_list is None:
            num_experts_list = [4] * layer_nums  # 每层 4 个专家
        assert len(num_experts_list) == layer_nums

        # patch_size_list 形状: [layer_nums, n_patches_per_layer]
        # 如果没给，就按 MTCL 的风格构造一个多尺度列表
        if patch_size_list is None:
            # 这里简单生成几种尺度，保证 <= seq_len
            base_patches = [16, 12, 8, 4]
            patch_size_list = []
            for li in range(layer_nums):
                scales = [max(2, min(self.seq_len, p // (2**li)))
                          for p in base_patches]
                patch_size_list.append(scales)
        else:
            # 支持传入 flat list（和 MTCL run.py 一样 reshape）
            if isinstance(patch_size_list[0], int):
                # e.g. [16, 12, 8, 32, 12, 8, 6, 4, ...]
                patch_size_list = np.array(patch_size_list).reshape(
                    layer_nums, -1
                ).tolist()

        self.ams_layers = nn.ModuleList()
        for li in range(layer_nums):
            ams = MoE(
                input_size=self.seq_len,
                output_size=self.seq_len,
                num_experts=num_experts_list[li],
                device=self.device,
                num_nodes=self.num_nodes,
                d_model=self.d_model,
                d_ff=self.d_ff,
                dynamic=False,
                patch_size=patch_size_list[li],
                noisy_gating=True,
                k=self.k,
                layer_number=li + 1,
                residual_connection=self.residual_connection,
                batch_norm=self.batch_norm,
                temp=self.temp,
            )
            self.ams_layers.append(ams)

    def forward(self, C: torch.Tensor):
        """
        C: [B, L, D]
        返回:
            cond_seq      : [B, L, d_model]
            balance_loss  : scalar 张量
            contrast_loss : scalar 张量
        """
        x = C  # [B, L, D]

        # 1) RevIN 归一化（与 MTCL 一致）
        if self.revin is not None:
            x = self.revin(x, mode='norm')  # [B, L, D]

        # 2) 映射到 AMS 的输入形状 [B, L, D, d_model]
        #    先在最后添一维，再通过全连接层
        out = self.start_fc(x.unsqueeze(-1))  # [B, L, D, d_model]

        balance_total = out.new_tensor(0.0)
        contrast_list = []

        # 3) 逐层 AMS (MoE)
        for ams in self.ams_layers:
            # MoE.forward(out) -> (out, balance_loss, contrast_loss)
            out, bal_loss, con_loss = ams(out)
            balance_total = balance_total + bal_loss
            contrast_list.append(con_loss)

        # 4) 对所有专家层的对比损失求平均
        if len(contrast_list) > 0:
            contrast_loss = torch.stack(contrast_list).mean()
        else:
            contrast_loss = out.new_tensor(0.0)

        # 5) 融合 10 个 IMU 传感器：对 sensor 维度做平均池化
        #    out: [B, L, D, d_model] -> [B, L, d_model]
        cond_seq = out.mean(dim=2)

        return cond_seq, balance_total, contrast_loss



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
    def __init__(
        self,
        D_cond,
        D_target,
        diffusion_cfg,
        L_hist: int,
        ams_cfg: dict = None,
        d_model_future: int = 128,
    ):
        """
        D_cond   : 条件维度 (10)
        D_target : 预测维度 (10)
        L_hist   : 历史长度 L，用于 AMS 的 seq_len
        ams_cfg  : dict 中可以覆盖 AMSCondEncoder 的默认超参
        """
        super().__init__()
        self.D_cond = D_cond
        self.D_target = D_target

        if ams_cfg is None:
            ams_cfg = {}

        device = ams_cfg.get("device", torch.device("cpu"))



        # ========= 1) 用 AMSCondEncoder 替代原 CondEncoder =========
        self.cond_encoder = AMSCondEncoder(
            num_nodes=D_cond,
            seq_len=L_hist,
            d_model=ams_cfg.get("d_model", d_model_future),
            d_ff=ams_cfg.get("d_ff", 256),
            layer_nums=ams_cfg.get("layer_nums", 2),
            k=ams_cfg.get("k", 2),
            num_experts_list=ams_cfg.get("num_experts_list", [4, 4]),
            patch_size_list=ams_cfg.get("patch_size_list", None),
            residual_connection=ams_cfg.get("residual_connection", True),
            use_revin=ams_cfg.get("use_revin", True),
            batch_norm=ams_cfg.get("batch_norm", True),
            temp=ams_cfg.get("temp", 2.0),
            device=device,
        )

        d_model_future = ams_cfg.get("d_model", d_model_future)

        # ========= 2) 时间步嵌入 (DDPM) =========
        self.time_embed = TimeEmbedding(diffusion_cfg.num_steps,
                                        d_model_future)

        # ========= 3) 未来轨迹分支 (和你原来一致) =========
        # 输入: [Y_t, self_cond] -> d_model_future
        self.future_proj = nn.Linear(D_target * 2, d_model_future)

        # Cross-Attention: future query 历史 AMS 特征
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model_future,
            num_heads=4,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model_future)

        # Transformer backbone in future domain
        self.future_backbone = FutureBackbone(d_model=d_model_future)
        self.out_proj = nn.Linear(d_model_future, D_target)
        # ======== 4) 时序二分类头 (MTCL 风格的 task-level 分支) ========
        # 这里采用一个简单策略：用 AMS 编码后的历史序列 cond_seq 的最后一个时间步
        # 作为「当前时刻」的表示，然后接一个二分类 head。
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model_future),
            nn.Linear(d_model_future, 1)   # 输出 logit 标量
        )

    def forward(self, C, Y_t, t, self_cond=None, return_aux: bool = False):
        """
        C       : [B, L, D]
        Y_t     : [B, H, D]
        t       : [B]  时间步
        self_cond : [B, H, D] 或 None
        return_aux: True 时返回 (eps_pred, balance_loss, contrast_loss, cls_logits)
        """
        # 1) AMS 编码历史（结构级融合）
        cond_seq, balance_loss, contrast_loss = self.cond_encoder(C)
        # cond_seq: [B, L, d_model_future]

        # 2) self-conditioning
        if self_cond is None:
            self_cond = torch.zeros_like(Y_t)

        # 3) 未来分支输入
        x_in = torch.cat([Y_t, self_cond], dim=-1)  # [B, H, 2D]
        future_emb = self.future_proj(x_in)  # [B, H, d_model]

        # 4) 时间步嵌入
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, d_model]
        query = future_emb + t_emb  # [B, H, d_model]

        # 5) Cross-Attention: query=未来，key/value=AMS 历史
        attn_out, _ = self.cross_attn(query, cond_seq, cond_seq)
        x = self.norm1(query + attn_out)

        # 6) 未来 Transformer backbone & 噪声预测
        h = self.future_backbone(x)
        eps_pred = self.out_proj(h)  # [B, H, D_target]

        # 7) 时序二分类 head：使用 cond_seq 的最后一个时间步作为 summary
        #    cls_feat: [B, d_model_future], cls_logits: [B]
        cls_feat = cond_seq[:, -1, :]  # 取历史最后一个时刻
        cls_logits = self.cls_head(cls_feat).squeeze(-1)

        if return_aux:
            return eps_pred, balance_loss, contrast_loss, cls_logits
        else:
            return eps_pred


# ==================== 4. Diffusion 训练 / 采样 ====================
def predict_x0_from_xt(schedule, xt, eps_pred, t):
    """从 x_t 反推 x_0"""
    B = xt.size(0)
    alpha_bar_t = schedule.alpha_bars.to(xt.device)[t].view(B, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    return (xt - sqrt_one_minus * eps_pred) / sqrt_alpha_bar


def training_step(
    model,
    schedule,
    C_batch,
    Y0_batch,
    optimizer,
    device,
    lambda_balance: float = 1.0,
    lambda_contrast: float = 0.1,
    lambda_cls: float = 1.0,
    y_cls: torch.Tensor = None,
):
    """
    标准 DDPM 训练步 + AMS 的 load-balance & contrastive 正则 + 时序二分类任务

    总损失:
        L = L_diffusion + λ_b * L_balance + λ_c * L_contrast + λ_cls * L_cls

    其中:
        L_diffusion = E[||eps_pred - eps||^2]
        L_cls       = BCEWithLogits(cls_logits, y_cls)
    """
    model.train()
    C_batch = C_batch.to(device)    # [B, L, D]
    Y0_batch = Y0_batch.to(device)  # [B, H, D]
    B = Y0_batch.size(0)

    # 随机时间步
    t = torch.randint(0, schedule.num_steps, (B,), device=device).long()
    eps = torch.randn_like(Y0_batch)
    Y_t = schedule.q_sample(Y0_batch, t, eps)

    # 50% 概率使用真实 Y0 作为 self-conditioning
    if torch.rand(1) < 0.5:
        with torch.no_grad():
            self_cond = Y0_batch
    else:
        self_cond = torch.zeros_like(Y0_batch)

    # ===== 调用 Joint backbone (带 AMS 辅助损失 + 分类 logit) =====
    eps_pred, balance_loss, contrast_loss, cls_logits = model(
        C_batch, Y_t, t, self_cond, return_aux=True
    )

    # 1) Diffusion 噪声回归损失
    diff_loss = F.mse_loss(eps_pred, eps)

    # 2) 分类损失（若提供标签）
    if (y_cls is not None) and (lambda_cls > 0.0):
        y_cls = y_cls.to(device)              # [B]
        cls_loss = F.binary_cross_entropy_with_logits(cls_logits, y_cls)
    else:
        cls_loss = torch.tensor(0.0, device=device)

    # 3) Joint loss
    loss = diff_loss \
           + lambda_balance * balance_loss \
           + lambda_contrast * contrast_loss \
           + lambda_cls * cls_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 方便监控：把四个子项都返回
    return {
        "loss_total":    loss.item(),
        "loss_diff":     diff_loss.item(),
        "loss_balance":  balance_loss.item(),
        "loss_contrast": contrast_loss.item(),
        "loss_cls":      cls_loss.item(),
    }



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
@torch.no_grad()
def impute_future_trajectory_ddim(
    model: torch.nn.Module,
    schedule,  # GaussianDiffusionSchedule
    C: torch.Tensor,
    Y_obs: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    sample_steps: int = 50,
    use_autocast: bool = True,
) -> torch.Tensor:
    """
    Fast future trajectory generation using strided DDIM sampling (eta=0).

    Args:
        model: UAVDiffusionModel-like module. Signature: model(C, Y_t, t, self_cond) -> eps_pred.
        schedule: GaussianDiffusionSchedule, must provide alpha_bars (buffer) and num_steps.
        C: [B, L, D] history sequence.
        Y_obs: [B, H, D] observed future (can be zeros).
        mask: [B, H, D] 1 for observed dims, 0 for unknown dims.
        device: torch.device.
        sample_steps: number of DDIM steps used in sampling (<= schedule.num_steps).
        use_autocast: enable AMP autocast for cuda/cpu.

    Returns:
        Y0_hat: [B, H, D] sampled future sequence.
    """
    assert C.device == device and Y_obs.device == device and mask.device == device, \
        "C/Y_obs/mask must be moved to `device` before calling."

    num_steps = int(schedule.num_steps)
    sample_steps = int(sample_steps)
    if sample_steps <= 0:
        raise ValueError("sample_steps must be positive.")
    if sample_steps > num_steps:
        sample_steps = num_steps

    B, H, D = Y_obs.shape
    Y_t = torch.randn_like(Y_obs)  # x_T
    self_cond = torch.zeros_like(Y_obs)

    # Strided timesteps: e.g., 999 -> ... -> 0
    t_seq = torch.linspace(num_steps - 1, 0, steps=sample_steps, device=device).long()
    alpha_bars = schedule.alpha_bars  # buffer already on device if schedule.to(device) was called

    mask_is_all_zero = bool(torch.count_nonzero(mask).item() == 0)

    # AMP context (safe only for cuda/cpu; xpu 环境建议先关掉 use_autocast)
    amp_enabled = use_autocast and device.type in ("cuda", "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    def _step_body(t_now: int, t_prev: int, y_t: torch.Tensor, sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.full((B,), t_now, device=device, dtype=torch.long)
        eps_pred = model(C, y_t, t, sc)

        a_t = alpha_bars[t_now]
        a_prev = torch.tensor(1.0, device=device) if t_prev < 0 else alpha_bars[t_prev]

        # x0 = (x_t - sqrt(1-a_t)*eps) / sqrt(a_t)
        x0 = (y_t - torch.sqrt(1.0 - a_t) * eps_pred) / torch.sqrt(a_t)

        # DDIM (eta=0): x_{t_prev} = sqrt(a_prev)*x0 + sqrt(1-a_prev)*eps_pred
        y_prev = torch.sqrt(a_prev) * x0 + torch.sqrt(1.0 - a_prev) * eps_pred
        return y_prev, x0

    for i in range(len(t_seq)):
        t_now = int(t_seq[i].item())
        t_prev = -1 if i == len(t_seq) - 1 else int(t_seq[i + 1].item())

        if amp_enabled:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                Y_prev, x0_pred = _step_body(t_now, t_prev, Y_t, self_cond)
        else:
            Y_prev, x0_pred = _step_body(t_now, t_prev, Y_t, self_cond)

        self_cond = x0_pred.detach()

        # 如果是纯预测（mask 全 0），直接用 Y_prev（最快）
        if mask_is_all_zero:
            Y_t = Y_prev
            continue

        # 否则保持与你原 RePaint 一致：在 t_prev 注入观测（注意：t_prev=-1 表示最终 x0）
        if t_prev >= 0:
            noise = torch.randn_like(Y_obs)
            t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
            Y_obs_t = schedule.q_sample(Y_obs, t_prev_tensor, noise)
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

    mu = np.mean(errors, axis=0, keepdims=True)
    std = np.std(errors, axis=0, keepdims=True) + 1e-8
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
        # 【修改】用 tqdm 包裹 loader，显示进度条
        print(f"[Info] Starting inference on {len(loader)} batches...")
        for C, Y_true in loader:
            C = C.to(device)           # [B, L, D]
            Y_true = Y_true.to(device) # [B, H, D]

            # 这里不使用观测值，做「纯预测」：Y_obs=0, mask=0 表示全维由 Diffusion 生成
            Y_obs = torch.zeros_like(Y_true)
            mask = torch.zeros_like(Y_true)

            # Y_pred = impute_future_trajectory(model, schedule, C, Y_obs, mask, device)
            Y_pred = impute_future_trajectory_ddim(
                model=model,
                schedule=schedule,
                C=C,
                Y_obs=Y_obs,
                mask=mask,
                device=device,
                sample_steps=50,  # 关键：从 1000 降到 50
                use_autocast=False,  # 若你是 xpu 环境，建议先 False；cuda 可改 True
            )

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


def tune_gamma_by_f1(y_true, gamma_p_val):
    """
    Oracle thresholding: pick gamma p-value (via score=-log10 p) maximizing F1.
    返回:
        best_gamma, best_f1, best_precision, best_recall
    """
    scores = -np.log10(gamma_p_val + 1e-16)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # thresholds 长度比 precision/recall 少 1，所以只对前 len(thresholds) 个点算 F1
    p = precision[:-1]
    r = recall[:-1]
    f1 = 2 * p * r / (p + r + 1e-8)

    best_idx = np.nanargmax(f1)
    best_score = thresholds[best_idx]
    best_gamma = 10 ** (-best_score)

    return best_gamma, f1[best_idx], p[best_idx], r[best_idx]


# ==================== 7. 主流程：训练 / 校准 / 测试 ====================
def main_alfa():
    args = parse_args()
    set_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Hyperparams
    L, H = 50, 20
    D_cond, D_target = 10, 10
    train_epochs = 300
    diffusion_steps = 1000

    # Paths
    THIS_FILE = Path(__file__).resolve()
    ROOT_DIR = THIS_FILE.parents[2]
    DATA_ROOT = ROOT_DIR / "data" / "alfa"
    train_dir = DATA_ROOT / "train"
    test_dir = DATA_ROOT / "test"

    train_files = list(train_dir.rglob("mavros-imu-data.csv"))
    test_files = list(test_dir.rglob("mavros-imu-data.csv"))
    if not train_files:
        raise RuntimeError(f"Error: No train files found in {train_dir}")
    if not test_files:
        raise RuntimeError(f"Error: No test files found in {test_dir}")
    print(f"Found {len(train_files)} train files, {len(test_files)} test files.")

    # pick faulty test flights
    faulty_test_files = []
    for f in test_files:
        s = str(f)
        if any(key in s for key in FAULT_ROW_DICT.keys()):
            faulty_test_files.append(f)
    faulty_test_files = list(dict.fromkeys(faulty_test_files))
    if len(faulty_test_files) == 0:
        raise RuntimeError("No faulty test flights found that match FAULT_ROW_DICT keys.")
    test_files = faulty_test_files[: min(args.num_test_flights, len(faulty_test_files))]
    print("Debug mode: using the following test flights:")
    for f in test_files:
        print(" ", f)

    # Datasets & Loaders
    print("Loading train data and fitting scaler...")
    train_dataset = ALFADataset(train_files, mode='train', scaler=None, L=L, H=H, return_label=True)
    scaler = train_dataset.get_scaler()
    num_workers = max(0, args.num_workers)
    persistent = True if num_workers > 0 else False

    def _worker_init_fn(worker_id):
        seed = args.seed + worker_id + 1
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    calib_dataset = ALFADataset(train_files, mode='train', scaler=scaler, L=L, H=H, return_label=False)
    # model & schedule
    cfg = DiffusionConfig(num_steps=diffusion_steps)
    schedule = GaussianDiffusionSchedule(cfg).to(device)
    ams_cfg = {"device": device, "d_model": 128, "d_ff": 256, "layer_nums": 2, "k": 2,
               "num_experts_list": [4, 4], "patch_size_list": [10,5,2,1,10,5,2,1],
               "residual_connection": True, "use_revin": True, "batch_norm": True, "temp": 2.0}

    model = UAVDiffusionModel(D_cond=D_cond, D_target=D_target, diffusion_cfg=cfg,
                              L_hist=L, ams_cfg=ams_cfg, d_model_future=ams_cfg["d_model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model_path = "uav_imputation_model.pth"

    # Load or train
    if os.path.exists(model_path):
        print(">>> Phase 1: Loading pre-trained Diffusion model.")
        state_dict = torch.load(model_path, map_location=device)
        if hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print(f"Loaded '{model_path}'.")
    else:
        print(">>> Phase 1: Training Diffusion+AMS+Classifier joint model from scratch.")
        for epoch in range(train_epochs):
            total_loss = 0.0
            for C_batch, Y_batch, y_cls in train_loader:
                stats = training_step(model, schedule, C_batch, Y_batch, optimizer, device,
                                      lambda_balance=1.0, lambda_contrast=0.1, lambda_cls=0.5, y_cls=y_cls)
                total_loss += stats["loss_total"]
            print(f"Epoch {epoch + 1}/{train_epochs}, Avg Loss: {total_loss / len(train_loader):.6f}")
        torch.save(model.state_dict(), model_path)
        print(f"Saved trained model to '{model_path}'.")

    # Calibration: use limited batches or disk-based collection
    print(">>> Phase 2: Calibrating GMM + Gamma on normal-like train data.")
    calib_loader = DataLoader(calib_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    limited_loader = list(itertools.islice(calib_loader, args.max_calib_batches))
    gmm_model, fisher_thresh, calib_stats = fit_gmm_on_diffusion_errors(
        model=model, schedule=schedule, loader=limited_loader, device=device,
        sensors=train_dataset.features, error_name='area', n_components=1,
        covariance_type='spherical', score_window=10, smoothing_window=10
    )
    # ... 后续保持不变 ...


    calib_errors, calib_gamma_pval, calib_fisher = calib_stats
    # 在“正常”校准数据上希望的假报警率（可尝试多档）
    target_fprs = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]    # 0.1% / 0.5% / 1%
    gamma_thresh_list = [
        (fpr, np.quantile(calib_gamma_pval, fpr)) for fpr in target_fprs
    ]
    gamma_thresh_default = gamma_thresh_list[len(gamma_thresh_list) // 2][1]

    print(f"[Calibration] 99.5% Fisher threshold (diagnostic only): "
          f"{fisher_thresh:.4f}")
    for fpr, gthr in gamma_thresh_list:
        print(f"[Threshold] Auto-calibrated gamma_thresh={gthr:.3e} "
              f"(target FPR={fpr:.3%} on calibration data)")

    # -------------------- Phase 3: 在 test 上逐 flight 检测与评估 --------------------
    all_metrics = {fpr: [] for fpr, _ in gamma_thresh_list}

    print("\n>>> Phase 3: Detection on test flights.")
    for test_file in test_files:
        print("\n" + "#" * 80)
        print(f"[Test Flight] {test_file}")
        print("#" * 80)

        # 构建该 flight 的 Dataset / DataLoader
        test_dataset = ALFADataset(
            [test_file], mode='test', scaler=scaler, L=L, H=H
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0,
            pin_memory=True
        )

        # ---- 检测 ----
        det_results = detect_with_diffusion_gmm(
            model=model,
            schedule=schedule,
            loader=test_loader,
            device=device,
            gmm_model=gmm_model,
            error_name=error_name,
            gamma_thresh=gamma_thresh_default,
            score_window=10,
            smoothing_window=10,
        )

        gamma_p_val = det_results["gamma_p_val"]
        p_val_sensors = det_results["p_val_sensors"]

        scores_global = -np.log10(gamma_p_val + 1e-16)
        score_threshold = -np.log10(gamma_thresh_default)

        num_steps = len(scores_global)
        H_forecast = test_dataset.H

        # ---- 构造 Ground Truth：根据 fault_row 映射 flatten index -> 原始 IMU 行号 ----
        fault_row = get_fault_row_for_file(Path(test_file))
        if fault_row is None:
            # 如果没有标注，默认视作纯正常飞行
            y_true = np.zeros(num_steps, dtype=int)
            print("[Ground Truth] No fault_row for this file. "
                  "Treat as normal flight.")
        else:
            idx = np.arange(num_steps)          # flatten 时间索引
            i = idx // H_forecast               # 样本索引
            h = idx % H_forecast                # 窗口内未来步索引
            t = i + L + h                       # 对应原始 IMU 行号

            y_true = (t >= fault_row).astype(int)
            print(f"[Ground Truth] fault_row = {fault_row}, "
                  f"positive ratio = {y_true.mean():.3f}")
        # ---- 先看全局 score 排序的上限能力 ----
        scores_global = -np.log10(gamma_p_val + 1e-16)

        auc_global = roc_auc_score(y_true, scores_global)
        ap_global = average_precision_score(y_true, scores_global)

        best_gamma, best_f1, best_p, best_r = tune_gamma_by_f1(y_true, gamma_p_val)

        neg_scores = scores_global[y_true == 0]
        pos_scores = scores_global[y_true == 1]

        print(f"[Global scores] ROC-AUC={auc_global:.4f}, "
              f"AP={ap_global:.4f}")
        print(f"[Score mean]   neg={neg_scores.mean():.3f}, "
              f"pos={pos_scores.mean():.3f}")
        print(f"[Oracle thr]   best_gamma={best_gamma:.3e}, "
              f"F1={best_f1:.3f}, P={best_p:.3f}, R={best_r:.3f}")

        # ---- 计算全局指标（多档 FPR） ----
        print("\n" + "=" * 60)
        print(f"Performance on {Path(test_file).name}")
        print("=" * 60)
        for fpr, gthr in gamma_thresh_list:
            y_pred = (gamma_p_val < gthr).astype(int)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            all_metrics[fpr].append((acc, prec, rec, f1))
            print(f"FPR={fpr:.3%} | thr={gthr:.3e} | "
                  f"Acc={acc:.2%} | P={prec:.2%} | R={rec:.2%} | F1={f1:.2%}")
        print("=" * 60)

        # 默认绘图阈值使用中档 FPR（gamma_thresh_default）
        y_pred_default = (gamma_p_val < gamma_thresh_default).astype(int)
        f1_default = f1_score(y_true, y_pred_default, zero_division=0)
        rec_default = recall_score(y_true, y_pred_default, zero_division=0)

        # ---- 全局分数可视化 ----
        window_size = 5
        smoothed_scores = pd.Series(scores_global).rolling(
            window=window_size, min_periods=1
        ).mean().values

        plt.figure(figsize=(12, 6))
        plt.plot(scores_global, label='Raw Score (-log10 p_gamma)')
        plt.plot(smoothed_scores, linewidth=2,
                 label=f'Smoothed Score (MA={window_size})')
        plt.axhline(score_threshold, linestyle='--',
                    label=f'Threshold (-log10 {gamma_thresh_default:.0e})')

        # 标记真实故障区域（若存在）
        if y_true.sum() > 0:
            pos_indices = np.where(y_true == 1)[0]
            first_pos = pos_indices[0]
            last_pos = pos_indices[-1]
            plt.axvspan(first_pos, last_pos, alpha=0.1,
                        label='Ground Truth Failure')

        plt.title(f"Diffusion+M2AD-area Detection: {Path(test_file).name} "
                  f"| F1={f1_default:.3f} | Recall={rec_default:.3f}")
        plt.xlabel("Time Index (sliding windows)")
        plt.ylabel("-log10 Gamma p-value")
        plt.legend()
        plt.tight_layout()
        fig_name = f'final_result_{Path(test_file).stem}_m2ad_area.png'
        plt.savefig(fig_name)
        plt.close()
        print(f"Global visualization saved as '{fig_name}'.")

        # ---- 每个传感器 ROC / PR 曲线 ----
        print("\n>>> Per-sensor ROC / PR analysis "
              f"(file={Path(test_file).name}).")

        sensor_scores = -np.log10(p_val_sensors + 1e-16)  # [T, D]

        for i_sensor, sensor in enumerate(train_dataset.features):
            s = sensor_scores[:, i_sensor]

            # ROC
            fpr, tpr, _ = roc_curve(y_true, s)
            roc_auc_val = auc(fpr, tpr)

            # PR
            precision_s, recall_s, _ = precision_recall_curve(y_true, s)
            ap_val = average_precision_score(y_true, s)

            print(f"[Sensor: {sensor}] ROC-AUC={roc_auc_val:.4f}, "
                  f"AP={ap_val:.4f}")

            plt.figure(figsize=(10, 4))

            # ROC 曲线
            plt.subplot(1, 2, 1)
            plt.plot(fpr, tpr, label=f'AUC={roc_auc_val:.3f}')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - {sensor}')
            plt.legend()

            # PR 曲线
            plt.subplot(1, 2, 2)
            plt.plot(recall_s, precision_s, label=f'AP={ap_val:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR - {sensor}')
            plt.legend()

            plt.tight_layout()
            sensor_fig = (
                f'roc_pr_{Path(test_file).stem}_'
                f'{sensor.replace(".", "_")}.png'
            )
            plt.savefig(sensor_fig)
            plt.close()
            print(f"Saved per-sensor ROC/PR figure to '{sensor_fig}'.")

    # -------------------- 全部测试 flight 的总体统计 --------------------
    if all_metrics:
        print("\n" + "=" * 60)
        print("Overall performance on all test flights")
        print("=" * 60)
        for fpr, metrics in all_metrics.items():
            if not metrics:
                continue
            accs = [m[0] for m in metrics]
            precs = [m[1] for m in metrics]
            recs = [m[2] for m in metrics]
            f1s = [m[3] for m in metrics]
            print(f"FPR={fpr:.3%} | Acc={np.mean(accs):.2%} | "
                  f"P={np.mean(precs):.2%} | R={np.mean(recs):.2%} | "
                  f"F1={np.mean(f1s):.2%}")
        print("=" * 60)


if __name__ == "__main__":
    main_alfa()
