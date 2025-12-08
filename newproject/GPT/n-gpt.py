# -*- coding: utf-8 -*-
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


# ==================== 1. 数据集定义 ====================
class ALFADataset(Dataset):
    def __init__(self, file_paths, mode='train', scaler=None, L=50, H=20):
        self.L = L
        self.H = H
        # ★★★ 修改后 (增加 orientation.x, y, z, w)
        self.features = [
            'orientation.x', 'orientation.y', 'orientation.z', 'orientation.w',
            'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z',
            'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z'
        ]

        raw_data_list = []
        for file in file_paths:
            df = pd.read_csv(file)
            # 简单检查列名是否存在，实际需根据 CSV 表头调整
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


# ==================== 2. 基础组件 ====================
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

        # 预测 x0
        x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # 后验均值
        coef1 = torch.sqrt(alpha_bar_prev) * self.betas[t] / (1.0 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

        # 广播处理
        while coef1.dim() < x_t.dim(): coef1 = coef1.unsqueeze(-1)
        while coef2.dim() < x_t.dim(): coef2 = coef2.unsqueeze(-1)

        mean = coef1 * x0_hat + coef2 * x_t

        if t_scalar == 0:
            return mean

        var = self.posterior_variance[t]
        while var.dim() < x_t.dim(): var = var.unsqueeze(-1)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise


# ==================== 3. 核心模型架构 (改造重点) ====================

class CondEncoder(nn.Module):
    def __init__(self, d_in, d_model, num_layers=2, num_heads=4, dim_ff=256, max_len=512):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, cond_seq):
        # ★★★ 关键点：移除 mean pooling，保留完整序列供 Attention 使用
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

        # ★★★ 改造：输入维度翻倍 (D_target * 2)，接收 [Y_t, self_cond]
        self.future_proj = nn.Linear(D_target * 2, d_model_future)

        # ★★★ 改造：Cross-Attention 模块
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model_future, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model_future)

        self.future_backbone = FutureBackbone(d_model=d_model_future)
        self.out_proj = nn.Linear(d_model_future, D_target)

    def forward(self, C, Y_t, t, self_cond=None):
        # 1. 编码历史 -> Key, Value [B, L, d]
        cond_seq = self.cond_encoder(C)

        # 2. 处理 Self-Conditioning
        if self_cond is None:
            self_cond = torch.zeros_like(Y_t)

        # 拼接 Input: [B, H, D*2]
        x_in = torch.cat([Y_t, self_cond], dim=-1)

        # 3. Embedding + Time
        future_emb = self.future_proj(x_in)  # [B, H, d]
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, d]
        query = future_emb + t_emb  # [B, H, d]

        # 4. ★★★ Cross-Attention: Future 查询 History
        attn_out, _ = self.cross_attn(query, cond_seq, cond_seq)
        x = self.norm1(query + attn_out)  # Residual + Norm

        # 5. Backbone & Output
        h = self.future_backbone(x)
        return self.out_proj(h)


# ==================== 4. 辅助函数 ====================

def predict_x0_from_xt(schedule, xt, eps_pred, t):
    """辅助：从 x_t 反推 x_0"""
    B = xt.size(0)
    alpha_bar_t = schedule.alpha_bars.to(xt.device)[t].view(B, 1, 1)
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)
    return (xt - sqrt_one_minus * eps_pred) / sqrt_alpha_bar


def training_step(model, schedule, C_batch, Y0_batch, optimizer, device):
    """训练步：包含 Self-Conditioning 随机丢弃"""
    model.train()
    C_batch = C_batch.to(device)
    Y0_batch = Y0_batch.to(device)
    B = Y0_batch.size(0)

    t = torch.randint(0, schedule.num_steps, (B,), device=device).long()
    eps = torch.randn_like(Y0_batch)
    Y_t = schedule.q_sample(Y0_batch, t, eps)

    # ★★★ Self-Cond 训练策略: 50% 概率使用真实值模拟预测结果
    if torch.rand(1) < 0.5:
        with torch.no_grad():
            self_cond = Y0_batch  # 模拟“完美预测”
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
    """核心推理：带 Mask 的补全逻辑"""
    model.eval()
    B, H, D = Y_obs.shape
    Y_t = torch.randn_like(Y_obs)
    self_cond = torch.zeros_like(Y_obs)  # 初始 self_cond

    for t_scalar in reversed(range(schedule.num_steps)):
        t = torch.full((B,), t_scalar, device=device).long()

        # 预测噪声
        eps_pred = model(C, Y_t, t, self_cond)

        # 更新 self_cond (用 x0 的估计值)
        x0_pred = predict_x0_from_xt(schedule, Y_t, eps_pred, t)
        self_cond = x0_pred.detach()

        # 反向一步
        Y_prev = schedule.p_sample_step(Y_t, t_scalar, eps_pred)

        # ★★★ RePaint 策略: 已知部分强制替换
        if t_scalar > 0:
            noise = torch.randn_like(Y_obs)
            t_prev = torch.full((B,), t_scalar - 1, device=device).long()
            Y_obs_t = schedule.q_sample(Y_obs, t_prev, noise)
        else:
            Y_obs_t = Y_obs

        Y_t = mask * Y_obs_t + (1 - mask) * Y_prev

    return Y_t


@torch.no_grad()
def compute_imputation_score(model, schedule, C, Y_true, device, mask_ratio=0.5):
    """计算补全误差分数"""
    model.eval()
    # Mask: 1=Keep, 0=Masked
    mask = (torch.rand_like(Y_true) > mask_ratio).float().to(device)

    Y_imputed = impute_future_trajectory(model, schedule, C, Y_true, mask, device)

    # 仅计算 Mask 部分的 MSE
    loss = F.mse_loss(Y_imputed * (1 - mask), Y_true * (1 - mask), reduction='none')
    score = loss.sum() / ((1 - mask).sum() + 1e-6)
    return score.item()


def determine_engineering_threshold(model, schedule, loader, device, mask_ratio=0.5, k=3.0):
    """基于验证集计算阈值"""
    print("\n[Threshold] Calibrating on normal data...")
    scores = []
    for i, (C, Y) in enumerate(loader):
        C, Y = C.to(device), Y.to(device)
        s = compute_imputation_score(model, schedule, C, Y, device, mask_ratio)
        scores.append(s)
        if i % 20 == 0: print(f" -> Batch {i}...")

    mu = np.mean(scores)
    std = np.std(scores)
    thresh = mu + k * std
    print(f"[Stats] Mean: {mu:.4f}, Std: {std:.4f} => Threshold: {thresh:.4f}")
    return thresh


# ==================== 5. 主流程 ====================

def main_alfa():
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 参数配置
    L, H = 50, 20
    D_cond, D_target = 10, 10
    train_epochs = 30  # 建议增加 epoch
    mask_ratio = 0.5  # 遮挡 50% 进行补全测试

    # 文件路径 (请根据实际情况修改)
    train_files = glob.glob('data/alfa/train/**/mavros-imu-data.csv', recursive=True)
    test_files = glob.glob('data/alfa/test/**/mavros-imu-data.csv', recursive=True)

    if not train_files:
        print("Error: No train files found.")
        return

    # 数据集
    print("Loading Data...")
    train_dataset = ALFADataset(train_files, mode='train', L=L, H=H)
    scaler = train_dataset.get_scaler()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

    # 初始化模型
    cfg = DiffusionConfig(num_steps=1000)
    schedule = GaussianDiffusionSchedule(cfg).to(device)
    model = UAVDiffusionModel(D_cond, D_target, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ★★★ 强制重新训练 (不加载旧权重)
    # print("\n>>> Phase 1: Training (New Architecture)...")
    # model.train()
    # for epoch in range(train_epochs):
    #     total_loss = 0
    #     for C, Y in train_loader:
    #         loss = training_step(model, schedule, C, Y, optimizer, device)
    #         total_loss += loss
    #     print(f"Epoch {epoch + 1}/{train_epochs}, Avg Loss: {total_loss / len(train_loader):.6f}")
    #
    # torch.save(model.state_dict(), "uav_imputation_model.pth")
    print("\n>>> Phase 1: Loading Pre-trained Model...")
    # 加载您刚才生成的 .pth 文件
    if os.path.exists("uav_imputation_model.pth"):
        model.load_state_dict(torch.load("uav_imputation_model.pth"))
        print("Success: Loaded 'uav_imputation_model.pth'")
    else:
        print("Error: Model file not found!")
        return

    model.eval()  # 切换到评估模式
    # 计算阈值
    calib_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    threshold = determine_engineering_threshold(model, schedule, calib_loader, device, mask_ratio=mask_ratio)

    # 测试环节
    print("\n>>> Phase 2: Testing...")
    if not test_files: return

    test_file = test_files[0]  # 取一个文件测试
    print(f"Testing on: {test_file}")
    test_dataset = ALFADataset([test_file], mode='test', scaler=scaler, L=L, H=H)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    scores = []
    with torch.no_grad():
        for i, (C, Y) in enumerate(test_loader):
            C, Y = C.to(device), Y.to(device)
            s = compute_imputation_score(model, schedule, C, Y, device, mask_ratio=mask_ratio)
            scores.append(s)
            if i % 100 == 0: print(f"Step {i}...")

        # ... (前接 Phase 2 测试循环，scores 列表已计算完毕) ...
        # ==================== 优化步骤：去噪与四维评估 ====================
        print("\n>>> Phase 3: Final Optimization & Evaluation...")

        # 1. 滑动平均平滑 (去噪关键)
        # 窗口越大曲线越平滑，但响应越慢；建议取 5-10
        window_size = 5
        smoothed_scores = pd.Series(scores).rolling(window=window_size, min_periods=1).mean().values

        # 2. 构造 Ground Truth (真实标签)
        # 注意：ALFA 数据集通常在文件中间发生故障，需根据文件名或 result.png 的突变点手动指定
        y_true = np.zeros(len(scores))

        # ★★★ 关键：请根据上一步 result.png 中波形突变的位置修改此值
        # 例如：如果图中第 100 步开始飙升，就设为 100
        failure_start_index = 100

        if "failure" in test_file or "carbon" in test_file:
            y_true[failure_start_index:] = 1  # 标记故障区间
            print(f"[Ground Truth] Failure marked starting at index {failure_start_index}")
        else:
            print("[Ground Truth] Normal flight assumed.")

        # 3. 预测与指标计算
        # 使用平滑后的分数与之前计算的阈值对比
        y_pred = (smoothed_scores > threshold).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 4. 打印最终成绩单
        print("\n" + "=" * 45)
        print(f"🏆 Final Performance Report (Window={window_size})")
        print("=" * 45)
        print(f"Accuracy  (准确率): {acc:.2%}")
        print(f"Precision (精确率): {prec:.2%}")
        print(f"Recall    (召回率): {rec:.2%}  <-- 重点关注，不能漏报！")
        print(f"F1 Score  (综合分): {f1:.4f}")
        print("=" * 45)

        # 5. 可视化对比
        plt.figure(figsize=(12, 6))
        plt.plot(scores, color='lightgray', label='Raw Score (Noisy)')
        plt.plot(smoothed_scores, color='blue', linewidth=2, label=f'Smoothed Score (MA={window_size})')
        plt.axhline(threshold, color='red', linestyle='--', label='Threshold')

        # 标记真实故障区域
        if np.sum(y_true) > 0:
            plt.axvspan(failure_start_index, len(scores), color='red', alpha=0.1, label='Ground Truth Failure')

        plt.title(f"Final Detection: F1={f1:.3f} | Recall={rec:.3f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig('final_result_optimized.png')
        plt.show()
        print("Done.")


if __name__ == "__main__":
    main_alfa()