import torch

torch.cuda.empty_cache()  # 清空缓存
import torch.nn as nn
import torch.nn.functional as F
import pywt  # PyWavelets for filter initialization
import math
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import typing as t
import random
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt
import torch
import numpy as np
import typing as t
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
# from soft_dtw_cuda import SoftDTW
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def visualize_segments_with_tsne(segments_np, perplexity=30, subsample_size=5000):
    """
    使用 t-SNE 对提取的波形片段进行降维并可视化。

    参数:
    - segments_np (np.ndarray): 从主导子带提取的所有片段，形状为 (n_segments, shapelet_length)。
    - perplexity (int): t-SNE 的 perplexity 参数，与每个点的近邻数有关。
    - subsample_size (int): 如果片段数量过多，为了加速计算，只对一个随机子集进行可视化。
    """
    print(f"\n--- Running t-SNE for visualization ---")

    # --- 数据子采样 ---
    # t-SNE 在大数据集上计算非常缓慢，如果片段过多，建议进行随机子采样
    if len(segments_np) > subsample_size:
        print(f"Dataset is large ({len(segments_np)} segments). Subsampling to {subsample_size} for t-SNE.")
        indices = np.random.choice(len(segments_np), subsample_size, replace=False)
        data_to_visualize = segments_np[indices]
    else:
        data_to_visualize = segments_np

    # # --- 执行 t-SNE ---
    # # n_components=2 表示我们将高维数据降到二维平面上
    # tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    # print("Fitting t-SNE... (this may take a moment)")
    # tsne_results = tsne.fit_transform(data_to_visualize)
    # print("t-SNE fitting complete.")
    #
    # # --- 绘图 ---
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # plt.figure(figsize=(10, 8))
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5, s=10)
    # plt.title('提取片段的t-SNE降维可视化结果')
    # plt.xlabel('t-SNE 维度 1')
    # plt.ylabel('t-SNE 维度 2')
    # plt.grid(True)
    # plt.show()


def normalize_signals(signals):
    # 计算每个信号的最大绝对值，保持维度以便广播
    max_vals = torch.max(torch.abs(signals), dim=1, keepdim=True)[0]
    # 防止除以零
    max_vals[max_vals == 0] = 1.0
    return signals / max_vals


# --------- 正弦时间 / 步长嵌入（供 diffusion 与 DSG 使用）---------
def sinusoidal_time_embedding(t: torch.Tensor, emb_dim: int):
    """
    t : (B,)  int64 / int32
    返回 (B, emb_dim)  float32
    """
    device = t.device
    half = emb_dim // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(half, device=device) / half
    )  # (half,)
    ang = t.float().unsqueeze(1) * freq.unsqueeze(0)  # (B, half)
    emb = torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)
    return emb


def _idwt_single_band(center, dominant_level, wavelet, target_len):
    """只让一个细节子带非零，其余—including A0—填 0，然后逆变换。"""
    levels = dominant_level  # L  = 最大分解层数
    coeffs = [None] * (levels + 1)  # [A_L, D_L … D1]
    coeffs[0] = np.zeros_like(center)  # ← A_L 先放 0（任意占位）
    coeffs[-(dominant_level + 1)] = center  # 把目标子带放进去

    rec = pywt.waverec(coeffs, wavelet)  # 时域波形
    # 裁/补到 shapelet_length
    if len(rec) < target_len:
        rec = np.pad(rec, (0, target_len - len(rec)))
    else:
        rec = rec[:target_len]
    return rec


class SoftDTW(nn.Module):
    def __init__(self, gamma: float = 1.0, normalize: bool = False):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x : (N, Lx, D)
        y : (K, Ly, D)
        return : (N, K)
        """
        N, Lx, D = x.shape
        K, Ly, _ = y.shape

        # 点积：  (N,K,Lx,Ly)
        dot = torch.matmul(
            x.unsqueeze(1),  # (N,1,Lx,D)
            y.unsqueeze(0).transpose(-1, -2)  # (1,K,D,Ly)
        )

        # R 的上一行
        prev_row = x.new_full((N, K, Ly + 1), float('-inf'))
        prev_row[:, :, 0] = 0.  # R[0,0] = 0

        for i in range(1, Lx + 1):
            curr_row = x.new_full((N, K, Ly + 1), float('-inf'))

            # j = 1…Ly
            for j in range(1, Ly + 1):
                # **立刻 clone，彻底切断对后续写入的别名**
                diag = prev_row[:, :, j - 1].clone()
                up = prev_row[:, :, j].clone()
                left = curr_row[:, :, j - 1].clone()

                cand = torch.stack((diag, up, left), dim=-1)  # (N,K,3)

                max_val = cand.max(-1, keepdim=True)[0]
                lse = (max_val.squeeze(-1) +
                       self.gamma *
                       ((cand - max_val) / self.gamma).exp().sum(-1).log())

                curr_row[:, :, j] = dot[:, :, i - 1, j - 1] + lse

            # **clone()，保证新一轮循环不会改旧行的存储**
            prev_row = curr_row.clone()

        dist = -self.gamma * prev_row[:, :, -1]
        if self.normalize:
            dist = dist / (Lx + Ly)
        return dist


# ======================================================================
# 修订版 1：PIMT_Net   —— 添加软阈值 + 更完整的正交 / 能量正则
# ======================================================================
class PIMT_Net(nn.Module):
    def __init__(self, wavelet_type='db4', levels=4, learnable=True):
        super(PIMT_Net, self).__init__()
        self.levels = levels

        wavelet = pywt.Wavelet(wavelet_type)
        kernel_size = len(wavelet.dec_lo)

        lo_filter = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32).view(1, 1, -1)
        hi_filter = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32).view(1, 1, -1)

        self.decomposition_layers, self.reconstruction_layers = nn.ModuleList(), nn.ModuleList()
        self.theta = nn.Parameter(torch.full((levels,), 1e-3))  # learnable soft-threshold

        for _ in range(levels):
            # stride=1 + 后续 AvgPool 实现论文里的 “先卷积再下采样”
            conv_lo = nn.Conv1d(1, 1, kernel_size,
                                stride=1, padding='same', bias=True)
            conv_hi = nn.Conv1d(1, 1, kernel_size,
                                stride=1, padding='same', bias=True)
            conv_lo.weight = nn.Parameter(lo_filter.clone(), requires_grad=learnable)
            conv_hi.weight = nn.Parameter(hi_filter.clone(), requires_grad=learnable)
            self.decomposition_layers.append(nn.ModuleDict({'lo': conv_lo, 'hi': conv_hi}))

            # 重构保持原 stride=2 的反卷积，与插 0 上采样等价
            pad = kernel_size // 2 - 1
            deconv_lo = nn.ConvTranspose1d(
                1, 1, kernel_size, stride=2, padding=pad, output_padding=0, bias=True)
            deconv_hi = nn.ConvTranspose1d(
                1, 1, kernel_size, stride=2, padding=pad, output_padding=0, bias=True)
            deconv_lo.weight = nn.Parameter(lo_filter.clone(), requires_grad=learnable)
            deconv_hi.weight = nn.Parameter(hi_filter.clone(), requires_grad=learnable)
            self.reconstruction_layers.append(nn.ModuleDict({'lo': deconv_lo, 'hi': deconv_hi}))

    def _soft_threshold(self, x, lam):
        return torch.sign(x) * F.relu(torch.abs(x) - lam)

    def forward(self, x):
        if x.dim() == 2:  # (B,T) -> (B,1,T)
            x = x.unsqueeze(1)

        detail_coeffs, approx_coeff = [], x
        for l, layer in enumerate(self.decomposition_layers):
            # 2)  PIMT_Net.forward() —— 若想保持论文“先卷积再池化”且完全无视图，可选这样改
            lo = layer['lo'](approx_coeff)
            hi = layer['hi'](approx_coeff)
            lo, hi = self._soft_threshold(lo, self.theta[l]), self._soft_threshold(hi, self.theta[l])
            # 改为 functional.avg_pool1d 不生成 view；或直接 stride=2 卷积（见上）
            lo = F.avg_pool1d(lo.contiguous(), 2)
            hi = F.avg_pool1d(hi.contiguous(), 2)

            detail_coeffs.append(hi)
            approx_coeff = lo
        return approx_coeff, detail_coeffs

    # --- 论文公式 (3) 正交正则 更完整的正则 ---
    def get_ortho_regularization_loss(self):
        loss = 0.0
        device = next(self.parameters()).device
        for layer in self.decomposition_layers:
            w_lo = layer['lo'].weight.view(-1)
            w_hi = layer['hi'].weight.view(-1)

            cross = torch.sum(w_lo * w_hi)
            autoc_lo = torch.sum(w_lo * w_lo)
            autoc_hi = torch.sum(w_hi * w_hi)

            # 目标张量用同形状，避免广播警告
            loss += F.mse_loss(cross.unsqueeze(0), torch.zeros(1, device=device))
            loss += F.mse_loss(autoc_lo.unsqueeze(0), torch.ones(1, device=device))
            loss += F.mse_loss(autoc_hi.unsqueeze(0), torch.ones(1, device=device))

        return loss

    def reconstruct(self, approx_coeff, detail_coeffs):

        if isinstance(detail_coeffs, tuple):
            detail_coeffs = list(detail_coeffs)

        # 逆序遍历：L → 1
        x = approx_coeff
        for l in reversed(range(self.levels)):
            lo_up = self.reconstruction_layers[l]['lo'](x)  # (B,1,2*T_l)
            hi_up = self.reconstruction_layers[l]['hi'](detail_coeffs[l])

            # # 由于边界填充，两个分量可能长度相差 1 ―― 对齐后再相加
            # min_len = min(lo_up.shape[-1], hi_up.shape[-1])
            # x = lo_up[..., :min_len] + hi_up[..., :min_len]  # (B,1,T_{l-1})

            diff = lo_up.size(-1) - hi_up.size(-1)
            if diff > 0:  # hi_up 短
                hi_up = F.pad(hi_up, (0, diff))
            elif diff < 0:  # lo_up 短
                lo_up = F.pad(lo_up, (0, -diff))

            x = lo_up + hi_up
        return x  # (B,1,T_orig)


# ======================================================================
# 修订版 2：DynamicShapeletGating —— forward方法已修正
# ======================================================================
class DynamicShapeletGating(nn.Module):
    def __init__(self,
                 levels: int,
                 num_shapelets: int = 20,  # 现在作为KMeans搜索的上限
                 shapelet_length: int = 20,
                 soft_dtw_gamma: float = 0.1,
                 local_window_delta: int = 5,
                 lambda_min: float = 0.2,
                 lambda_max: float = 1.5):
        super().__init__()
        self.levels = levels
        # num_shapelets 现在是K的上限和初始占位符
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.delta = local_window_delta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Shapelets参数现在是动态的，先用占位符初始化
        # 其真实数量将在initialize_shapelets中由肘部法则确定
        self.shapelets = nn.Parameter(torch.randn(num_shapelets, 1, shapelet_length))

        self.gamma_gating = nn.Parameter(torch.tensor(1.0))
        # w_k权重也需要动态创建
        self.w_k = nn.Parameter(torch.full((num_shapelets,), 1.0 / num_shapelets))
        self.v_l = nn.Parameter(torch.ones(levels))

        self.soft_dtw = SoftDTW(gamma=soft_dtw_gamma, normalize=False)
        self.smoothness_loss = torch.tensor(0.0)

    @torch.no_grad()
    # 【修改】函数参数名从 clean_loader 改为 initialization_loader，以反映其现在处理的是带噪数据
    def initialize_shapelets(self, initialization_loader: DataLoader, pim_net: nn.Module):
        """
        流程:
        1. 分解信号，找到能量主导子带。
        2. 对主导子带中的每个信号滑窗，提取所有局部片段 pi。
        3. 使用肘部法则确定最佳聚类簇数 K。
        4. 使用最佳K值进行K-Means++聚类。
        5. 对每个簇，将其内部所有片段 pi 逐一进行逆变换（使用pim_net.reconstruct），
           然后将得到的时域信号求平均，得到最终的 shapelet sk。
        """
        print("Initializing shapelets with dynamic K and literal reconstruction...")
        device = next(pim_net.parameters()).device
        pim_net.eval()

        # === 步骤 1 & 2: 提取所有 pi 片段 ===
        print("Step 1 & 2: Decomposing signals and extracting segments...")
        coeff_buffers = [[] for _ in range(self.levels)]
        # 从初始化数据加载器中获取样本信号，并确保取第一个元素（即带噪信号）
        x_sample = next(iter(initialization_loader))[0].to(device)
        approx_sample, detail_samples = pim_net(x_sample)

        approx_coeff_shape = (1, *approx_sample.shape[1:])  # (1, C, T)
        detail_coeff_shapes = [(1, *d.shape[1:]) for d in detail_samples]

        # 【修改】遍历初始化加载器，并确保使用batch中的第一个元素（带噪信号）
        for batch in initialization_loader:
            x = batch[0].to(device)  # batch[0] 是带噪信号, batch[1] 是干净信号
            _, detail_coeffs = pim_net(x)
            for l, d in enumerate(detail_coeffs):
                coeff_buffers[l].append(d.squeeze(1).cpu().numpy())

        stacks = [np.concatenate(b, 0) for b in coeff_buffers if b]
        energies = [np.linalg.norm(s) for s in stacks]
        dominant_idx = int(np.argmax(energies))
        dominant_data = stacks[dominant_idx]  # (N_signals, T_dom)

        # 假设 dominant_data 是一个 (n_shapelets, time_steps) 的二维数组或 tensor
        data1 = dominant_data.cpu().numpy() if hasattr(dominant_data, "cpu") else dominant_data
        x = range(data1.shape[1])

        plt.figure(figsize=(12, 4))
        for i in range(2):  # 遍历每个 shapelet
            plt.plot(x, data1[i], label=f'Shapelet {i}')
        plt.title('Shapelets')
        plt.xlabel('Time step')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        segments_list = []
        # 【核心修改】
        # 将步长 step 从 1 改为 self.shapelet_length，实现无重叠切割
        step = self.shapelet_length

        for single_signal_coeffs in dominant_data:
            if single_signal_coeffs.shape[-1] >= self.shapelet_length:
                # range函数的第三个参数step控制了窗口的移动步长
                for i in range(0, single_signal_coeffs.shape[-1] - self.shapelet_length + 1, step):
                    segments_list.append(single_signal_coeffs[i:i + self.shapelet_length])
        if not segments_list:
            raise ValueError("No segments were extracted. The signal length may be shorter than shapelet_length.")

        segments_np = np.stack(segments_list, 0)
        print(f"Extracted {len(segments_np)} non-overlapping segments from dominant level {dominant_idx}.")

        # # === 【核心修改】步骤 3: 使用肘部法则确定最佳 K ===
        # print("Step 3: Determining optimal number of shapelets (K) using the Elbow Method...")
        # inertias = []
        # # K值的搜索范围从2到您设定的上限
        # k_range = range(2, self.num_shapelets + 1)
        # for k in k_range:
        #     kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42).fit(segments_np)
        #     inertias.append(kmeans.inertia_)
        #
        # # 计算每个点到首尾连线的距离，距离最大的点即为“肘部”
        # p1 = np.array([k_range[0], inertias[0]])
        # p2 = np.array([k_range[-1], inertias[-1]])
        # distances = []
        # for i in range(len(k_range)):
        #     p3 = np.array([k_range[i], inertias[i]])
        #     dist = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        #     distances.append(dist)
        #
        # optimal_k = k_range[np.argmax(distances)]
        # print(f"Optimal K found via Elbow Method: {optimal_k}")
        #

        visualize_segments_with_tsne(segments_np)
        # === 步骤 3: 使用最佳 K 进行最终聚类 ===
        print(f"Step 4: Performing final clustering with K={self.num_shapelets}...")
        kmeans = KMeans(n_clusters=self.num_shapelets, init='k-means++', n_init=100, random_state=42).fit(segments_np)
        cluster_labels = kmeans.labels_
        print("Cluster labels:", np.bincount(cluster_labels))
        print("Cluster centers shape:", kmeans.cluster_centers_.shape)

        # === 步骤 4: 逐簇进行逆变换和平均 ===
        print("Step 5: Reconstructing shapelets for each cluster...")
        final_shapelets_list = []
        for k in range(self.num_shapelets):
            cluster_member_segments = segments_np[cluster_labels == k]

            time_domain_members = []
            for segment_w in cluster_member_segments:
                approx_coeff_template = torch.zeros(approx_coeff_shape, device=device)
                detail_coeffs_template = [torch.zeros(s, device=device) for s in detail_coeff_shapes]

                center_tensor = torch.tensor(segment_w, dtype=torch.float32, device=device)
                target_len = detail_coeff_shapes[dominant_idx][2]
                dominant_band_sparse = torch.zeros((1, 1, target_len), device=device)
                dominant_band_sparse[0, 0, :self.shapelet_length] = center_tensor
                detail_coeffs_template[dominant_idx] = dominant_band_sparse

                time_version = pim_net.reconstruct(approx_coeff_template, detail_coeffs_template)

                time_domain_members.append(time_version[0, 0, :self.shapelet_length])

            if time_domain_members:
                sk = torch.stack(time_domain_members).mean(dim=0)

                print(f"Shapelet {k} mean:", sk.mean().item(), "std:", sk.std().item())
                final_shapelets_list.append(sk.unsqueeze(0))

        if final_shapelets_list:
            final_shapelets_tensor = torch.cat(final_shapelets_list, dim=0)

            data = final_shapelets_tensor.detach().cpu().numpy()
            x = range(data.shape[1])  # 时间步（横轴）
            plt.figure(figsize=(12, 4))
            for i in range(data.shape[0]):  # 遍历每个 shapelet
                plt.plot(x, data[i], label=f'Shapelet {i}')
            plt.title('Shapelets')
            plt.xlabel('Time step')
            plt.ylabel('Value')
            plt.legend(loc='upper right', ncol=2, fontsize='small')  # 多 shapelet 时更紧凑
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            self.shapelets = nn.Parameter(final_shapelets_tensor)

        else:
            print("Warning: No shapelets were generated.")

        print("Shapelet initialization complete.")
        pim_net.train()

    # ### 以下是修改的核心部分 ###
    def forward(
            self,
            detail_coeffs: t.List[torch.Tensor],
            approx_coeffs: t.List[torch.Tensor],
            time_step: torch.Tensor,
    ):
        """
        前向传播部分，现在会使用动态确定的 self.num_shapelets
        """
        device = detail_coeffs[0].device
        # 由于 self.num_shapelets 是动态的，每次前向传播时从 self.shapelets 获取K值
        K = self.shapelets.shape[0]
        M = self.shapelet_length

        if K == 0:  # 如果没有初始化出shapelet，则直接返回
            return detail_coeffs, torch.zeros(detail_coeffs[0].shape[0], self.levels, 0, device=device)

        # (K, 1, M) -> (K, M, 1) 以匹配 soft_dtw
        shapelets_for_dtw = self.shapelets.permute(0, 1).unsqueeze(2)

        denoised, feats = [], []
        smooth_loss = 0.0

        for l, d_l in enumerate(detail_coeffs):
            B, _, T_l = d_l.shape

            if T_l < M:
                denoised.append(d_l)
                feats.append(torch.zeros(B, K, device=device))
                continue

            segs = d_l.unfold(2, M, 1).permute(0, 2, 3, 1)
            B, N, _, _ = segs.shape
            segs_flat = segs.reshape(-1, M, 1)

            # DTW 计算
            gating_tensor_flat = torch.exp(
                -torch.abs(self.gamma_gating) * self.soft_dtw(
                    segs_flat,
                    shapelets_for_dtw
                )
            )
            G = gating_tensor_flat.view(B, N, K).permute(0, 2, 1)

            max_pool = F.max_pool1d(G, 2 * self.delta + 1, 1, self.delta)
            w = F.softmax(self.w_k, dim=0)
            A = (max_pool * w.view(1, -1, 1)).sum(1)
            tau = torch.sigmoid(A)

            diff = tau[:, 1:] - tau[:, :-1]
            grad_norm = diff.norm(p=2, dim=1)
            smooth_loss += (F.relu(self.lambda_min - grad_norm) + F.relu(grad_norm - self.lambda_max)).mean()

            center = (M - 1) // 2
            d_center = d_l[:, 0, center: center + N]
            thr = self.v_l[l] * tau
            d_tilde_center = torch.sign(d_center) * F.relu(d_center.abs() - thr)

            # --- 【报错修复】开始 ---
            # 原始的原地操作代码 (导致报错)
            d_new = d_l.clone()
            d_new[:, 0, center: center + N] = d_tilde_center

            # # 替换为非原地操作：通过拼接构建一个新张量
            # prefix = d_l[:, :, :center]
            # suffix = d_l[:, :, center + N:]
            #
            # # 需要确保 d_tilde_center 有正确的通道维度 (B, 1, N)
            # d_tilde_center_reshaped = d_tilde_center.unsqueeze(1)
            #
            # d_new = torch.cat([prefix, d_tilde_center_reshaped, suffix], dim=2)
            # --- 【报错修复】结束 ---

            denoised.append(d_new)
            feats.append(G.mean(-1))

        self.smoothness_loss = smooth_loss
        F_S = torch.stack(feats, 1)

        return denoised, F_S


# ======================================================================
# 修正版 3.1：ConditionedUNet (替换 SimpleUNet)
# 更符合论文描述的U-Net结构，用于噪声预测
# ======================================================================
class UNetResBlock(nn.Module):
    """一个带时间步和条件嵌入的U-Net残差块"""

    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_dim, out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.residual_conv = nn.Conv1d(in_channels, out_channels,
                                       kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb, cond):
        res = self.residual_conv(x)
        h = self.relu(self.conv1(x))

        # 注入时间和条件信息
        time_cond = self.relu(self.time_mlp(t_emb)).unsqueeze(-1)
        cond_info = self.relu(self.cond_mlp(cond)).unsqueeze(-1)
        h = h + time_cond + cond_info

        h = self.relu(self.conv2(h))
        return h + res


class ConditionedUNet(nn.Module):
    """
    【标量扩散版】
    U-Net的结构不变，但在最后增加一个输出头，使其输出一个标量。
    """

    def __init__(self, in_channels, cond_channels, out_channels, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, time_emb_dim), nn.ReLU())
        self.cond_proj = nn.Linear(cond_channels, time_emb_dim)

        # U-Net 核心结构 (与之前版本相同)
        self.down1 = UNetResBlock(in_channels, 64, time_emb_dim, time_emb_dim)
        self.down2 = UNetResBlock(64, 128, time_emb_dim, time_emb_dim)
        self.pool = nn.AvgPool1d(2)
        self.mid = UNetResBlock(128, 256, time_emb_dim, time_emb_dim)
        self.up1 = UNetResBlock(256 + 128, 128, time_emb_dim, time_emb_dim)
        self.up2 = UNetResBlock(128 + 64, 64, time_emb_dim, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        # 【核心修改】增加一个输出头，将最终的特征图映射为单个标量值
        # 这里的 out_channels 实际上是 64，因为是最终卷积前的通道数
        self.final_feature_channels = 64
        self.output_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化，将长度维度压缩为1
            nn.Flatten(),  # 展平
            nn.Linear(self.final_feature_channels, 1)  # 线性层输出单个值
        )

    def forward(self, x, t_emb, cond_vec):
        t_emb = self.time_mlp(t_emb)
        cond_emb = self.cond_proj(cond_vec)

        h1 = self.down1(x, t_emb, cond_emb)
        h2 = self.down2(self.pool(h1), t_emb, cond_emb)
        h_mid = self.mid(self.pool(h2), t_emb, cond_emb)

        h = self.upsample(h_mid)
        if h.shape[2] != h2.shape[2]:
            h = F.pad(h, (0, h2.shape[2] - h.shape[2]))
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, t_emb, cond_emb)

        h = self.upsample(h)
        if h.shape[2] != h1.shape[2]:
            h = F.pad(h, (0, h1.shape[2] - h.shape[2]))
        h = torch.cat([h, h1], dim=1)
        final_features = self.up2(h, t_emb, cond_emb)  # (B, 64, L)

        # 【核心修改】通过输出头得到标量预测
        scalar_output = self.output_head(final_features)  # (B, 1)
        return scalar_output.squeeze(-1)  # 返回 (B,) 形状的标量


# ======================================================================
# 修正版 4.2：WaveletConditionedDiffusionGenerator
# ======================================================================
class WaveletConditionedDiffusionGenerator(nn.Module):
    """
    【标量扩散版】
    完全重写以实现基于标量（总能量）的扩散模型。
    """

    def __init__(self,
                 levels: int,
                 num_shapelets: int,
                 timesteps: int = 50,
                 time_emb_dim: int = 32,
                 tau: float = 0.5):  # tau 不再用于此版本的调度
        super().__init__()
        self.levels = levels
        self.timesteps = timesteps
        self.time_emb_dim = time_emb_dim
        self.cond_channels = levels * num_shapelets

        # U-Net现在输出标量
        self.epsilon_theta = ConditionedUNet(
            in_channels=1,
            cond_channels=self.cond_channels,
            out_channels=1,  # 输出通道参数在此模型中不再直接使用
            time_emb_dim=time_emb_dim,
        )

        # 【核心修改】扩散调度现在是针对标量的，恢复为标准形式
        s = 0.008
        t_steps = torch.arange(timesteps)
        ft = torch.cos(((t_steps / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = ft / ft[0]
        betas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        betas_padded = F.pad(betas, (1, 0), value=0.0001)

        # 注册所有需要的调度表，它们现在都是一维的
        self.register_buffer('betas', betas_padded)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # self.register_buffer('betas', betas)
        # self.register_buffer('alphas_cumprod', alphas_cumprod)
        # self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

    def flatten_coeffs(self, approx_coeff, detail_coeffs):
        flat_list = [approx_coeff.flatten(1)] + [c.flatten(1) for c in detail_coeffs]
        return torch.cat(flat_list, 1).unsqueeze(1)

    def unflatten_coeffs(self, flat, shapes):
        flat = flat.squeeze(1)
        outs, idx = [], 0
        for s in shapes:
            n = s[1] * s[2]
            outs.append(flat[:, idx:idx + n].view(-1, *s[1:]))
            idx += n
        return outs[0], outs[1:]

    def aggregate_energy(self, approx, details):
        """【新增】聚合函数：计算所有小波系数的总能量"""
        total_energy = torch.pow(approx, 2).sum(dim=[1, 2])
        for d in details:
            total_energy += torch.pow(d, 2).sum(dim=[1, 2])
        return total_energy  # 返回 (B,) 形状的标量

    def q_sample(self, d0_scalar, t, noise=None):
        """【修改】前向过程：在标量上加噪"""
        if noise is None:
            noise = torch.randn_like(d0_scalar)

        # 从预计算的buffer中获取对应时间步t的参数
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]

        # 返回加噪后的标量和所加的噪声标量
        return sqrt_alpha_bar_t * d0_scalar + sqrt_one_minus_alpha_bar_t * noise, noise

    def disaggregate_by_energy(self, energy_scalar, template_approx, template_details):
        """【新增】解聚函数：根据目标能量缩放模板系数"""
        with torch.no_grad():
            current_energy = self.aggregate_energy(template_approx, template_details)
            # 防止除以零，并增加稳定性
            safe_current_energy = torch.clamp(current_energy, min=1e-8)
            safe_target_energy = torch.clamp(energy_scalar, min=0.0)

            # 能量是振幅的平方，因此振幅缩放因子是能量比的平方根
            scale_factor = torch.sqrt(safe_target_energy / safe_current_energy)

        # 将缩放因子应用到每个系数张量上
        pred_approx = template_approx * scale_factor.view(-1, 1, 1)
        pred_details = [d * scale_factor.view(-1, 1, 1) for d in template_details]

        return pred_approx, pred_details

    def compute_loss(self,
                     diffusion_target_approx: torch.Tensor,
                     diffusion_target_detail: t.List[torch.Tensor],
                     clean_detail_for_reg: t.List[torch.Tensor],
                     F_S: torch.Tensor,
                     Lambda: float = 0.1,
                     lambda_1: float = 0.1,
                     lambda_2: float = 0.05):
        B = diffusion_target_approx.size(0)
        device = diffusion_target_approx.device

        # 【标量扩散核心逻辑】
        # 1. 聚合：将输入的系数张量D0_dsg聚合为初始能量标量d0
        d0_scalar = self.aggregate_energy(diffusion_target_approx, diffusion_target_detail)

        # 2. 标量扩散：对能量标量d0进行加噪，得到dt_scalar和真实噪声ε_true_scalar
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        dt_scalar, ε_true_scalar = self.q_sample(d0_scalar, t)

        # 3. U-Net预测：U-Net的输入依然是高维的带噪系数张量。
        #    我们通过缩放DSG的输出来构造一个能量与dt_scalar匹配的带噪张量。
        dt_approx, dt_details = self.disaggregate_by_energy(dt_scalar, diffusion_target_approx, diffusion_target_detail)
        dt_tensor = self.flatten_coeffs(dt_approx, dt_details)  # U-Net需要扁平化输入

        time_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
        cond_vec = F_S.reshape(B, -1)
        ε_unet_scalar = self.epsilon_theta(dt_tensor, time_emb, cond_vec)

        # 4. 计算物理正则项：
        #    首先需要通过解聚得到预测的干净高维信号 D0_pred
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
        # 从标量预测中得到预测的干净能量 d0_pred_scalar
        d0_pred_scalar = (dt_scalar - sqrt_one_minus_alpha_bar_t * ε_unet_scalar) / sqrt_alpha_bar_t

        # 解聚得到高维系数 D0_pred
        pred_approx, pred_details = self.disaggregate_by_energy(d0_pred_scalar, diffusion_target_approx,
                                                                diffusion_target_detail)

        # 计算 R_cross 和 R_energy
        R_cross = 0.0
        for l in range(self.levels - 1):
            upsampled = F.interpolate(pred_details[l + 1], scale_factor=2)
            target = pred_details[l]
            min_len = min(upsampled.shape[-1], target.shape[-1])
            R_cross += F.mse_loss(upsampled[..., :min_len], target[..., :min_len])

        R_energy = 0.0
        # clean_approx_for_reg 需要从外部传入，这里我们暂时只用detail
        clean_approx_for_reg = diffusion_target_approx  # 假设近似部分不变
        true_clean_energy = self.aggregate_energy(clean_approx_for_reg, clean_detail_for_reg)
        R_energy = torch.mean(torch.abs(d0_pred_scalar - true_clean_energy))

        # 5. 计算最终损失
        # 严格遵循公式(13)和(16)的“双重贡献”逻辑
        ε_predicted_total = ε_unet_scalar + Lambda * (R_cross + R_energy)
        loss_mse = F.mse_loss(ε_predicted_total, ε_true_scalar)
        total_loss = loss_mse + lambda_1 * R_cross + lambda_2 * R_energy

        print(
            f" total_loss: {total_loss.item():.6f}, loss_mse: {loss_mse.item():.6f}, R_cross: {R_cross.item():.6f}, R_energy: {R_energy.item():.6f}")
        return total_loss

    @torch.no_grad()
    def sample(self, approx_in, detail_in, F_S):
        """
        【修正版】
        1. 使用更数值稳定的 DDIM 采样公式。
        2. 在最终解聚前，确保预测的能量为非负数。
        """
        B = approx_in.size(0)
        device = approx_in.device

        # 初始能量标量从纯高斯噪声中采样
        d_t_scalar = torch.randn(B, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((B,), i, device=device, dtype=torch.long)

            # 构造高维输入给U-Net
            # d_t_scalar**2 确保了作为能量模板的输入恒为正
            dt_approx, dt_details = self.disaggregate_by_energy(d_t_scalar ** 2, approx_in, detail_in)
            dt_tensor = self.flatten_coeffs(dt_approx, dt_details)

            time_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
            cond_vec = F_S.reshape(B, -1)

            # U-Net预测标量噪声
            ε_unet_scalar = self.epsilon_theta(dt_tensor, time_emb, cond_vec)

            # 在采样阶段，我们不使用物理正则项修正
            ε_predicted = ε_unet_scalar

            # --- 【核心修正 1】改用更稳定的 DDIM 采样公式 ---
            sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]

            # 1. 预测干净能量 d0_pred (x0_pred)
            d0_pred_scalar = (d_t_scalar - sqrt_one_minus_alpha_bar_t * ε_predicted) / sqrt_alpha_bar_t

            if i > 0:
                # 2. 准备下一步的 alpha 参数
                sqrt_alpha_bar_prev = self.sqrt_alphas_cumprod[i - 1]

                # 3. 计算 d_{t-1}
                # 这个公式避免了除以可能为零的项，因此更稳定
                d_t_scalar = sqrt_alpha_bar_prev * d0_pred_scalar + \
                             torch.sqrt(1. - sqrt_alpha_bar_prev ** 2) * ε_predicted
            else:
                # 当 i=0 时，直接得到最终的去噪结果
                d_t_scalar = d0_pred_scalar

        # --- 【核心修正 2】在最终解聚前，确保预测的能量 d0 为非负数 ---
        # 使用 F.relu() 将所有可能为负的预测能量值修正为0
        final_clean_energy = F.relu(d_t_scalar)

        # 用预测出的、确保为正的干净能量标量来解聚，得到最终的高维信号
        final_approx, final_details = self.disaggregate_by_energy(final_clean_energy, approx_in, detail_in)

        return final_approx, final_details


# ======================================================================
# DGSAD_Model  ·  顶层封装
# ======================================================================
class DGSAD_Model(nn.Module):
    """
    Physics-Informed  Wavelet  →  Dynamic Shapelet Gating  →  Diffusion  全流程封装
    """

    def __init__(
            self,
            levels: int = 4,
            wavelet_type: str = "db4",
            num_shapelets: int = 4,
            shapelet_length: int = 100,
            diffusion_steps: int = 50,
            time_emb_dim: int = 32,
    ):
        super().__init__()

        self.levels = levels
        self.diffusion_steps = diffusion_steps
        self.time_emb_dim = time_emb_dim

        self.pim_net = PIMT_Net(wavelet_type=wavelet_type, levels=levels)
        self.dsg = DynamicShapeletGating(
            levels=levels,
            num_shapelets=num_shapelets,
            shapelet_length=shapelet_length,
        )
        self.wcdg = WaveletConditionedDiffusionGenerator(
            levels=levels,
            num_shapelets=num_shapelets,
            timesteps=diffusion_steps,
            time_emb_dim=time_emb_dim,
        )

    def finalize_wcdg_initialization(self):
        """
        根据DSG动态确定的K值，延迟实例化WCDG模块。
        此方法必须在 `dsg.initialize_shapelets()` 之后、创建优化器之前调用。
        """
        true_num_shapelets = self.dsg.num_shapelets
        print(f"[Model Finalization] WCDG will be initialized with true K = {true_num_shapelets}")

        self.wcdg = WaveletConditionedDiffusionGenerator(
            levels=self.levels,
            num_shapelets=true_num_shapelets,
            timesteps=self.diffusion_steps,
            time_emb_dim=self.time_emb_dim,
        )
        self.wcdg.to(next(self.parameters()).device)

    # ------------------------------------------------------------------
    # forward  (训练 / 推理 两种模式)
    # ------------------------------------------------------------------
    def forward(
            self,
            x_noisy: torch.Tensor,
            x_clean: t.Optional[torch.Tensor] = None,
            *,
            training: bool = False,
    ):
        """
        Parameters
        ----------
        x_noisy : (B,T) or (B,1,T)   输入带噪信号
        x_clean : (B,T) or (B,1,T)   仅在 training=True 时必填
        training: bool                True→返回总 loss；False→返回去噪波形
        """
        if self.wcdg is None:
            raise RuntimeError(
                "WCDG module has not been initialized. "
                "Please call `model.finalize_wcdg_initialization()` after shapelet initialization and before training/inference."
            )

        # ----------- 推理模式 -----------
        if not training:
            approx_noisy, detail_noisy = self.pim_net(x_noisy)
            detail_dsg, F_S = self.dsg(
                detail_noisy,
                approx_coeffs=[approx_noisy] * len(detail_noisy),
                time_step=torch.zeros(x_noisy.size(0), device=x_noisy.device, dtype=torch.long),
            )
            approx_hat, detail_hat = self.wcdg.sample(approx_noisy, detail_dsg, F_S)
            x_denoised = self.pim_net.reconstruct(approx_hat, detail_hat)
            return x_denoised

        # ----------- 训练模式 -----------
        if x_clean is None:
            raise ValueError("`x_clean` must be provided when training=True")

        # 1) 分解干净 & 带噪信号
        approx_clean, detail_clean = self.pim_net(x_clean)
        approx_noisy, detail_noisy = self.pim_net(x_noisy)

        # 2) DSG处理带噪系数，获取初步去噪结果和条件特征
        detail_dsg, F_S = self.dsg(
            detail_noisy,
            approx_coeffs=[approx_noisy] * len(detail_noisy),
            time_step=torch.zeros(x_noisy.size(0), device=x_noisy.device, dtype=torch.long),
        )

        # 3) 【修改】计算 WCDG 损失
        #    - 扩散目标是 DSG 的输出 (approx_noisy, detail_dsg)
        #    - 能量正则化的基准是干净信号的系数 (detail_clean)
        loss = self.wcdg.compute_loss(
            diffusion_target_approx=approx_noisy,
            diffusion_target_detail=detail_dsg,
            clean_detail_for_reg=detail_clean,
            F_S=F_S
        )
        return loss


# ==============================================================================
# 主函数: 演示如何运行模型
# ==============================================================================
def train_one_epoch(model, data_loader, optimizer, device, print_every=10):
    model.train()
    running = 0.0
    for step, (noisy, clean) in enumerate(data_loader, 1):
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss = model(noisy, clean, training=True)

        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % print_every == 0 or step == len(data_loader):
            print(f"  step {step:3d}/{len(data_loader)} | mean loss = {running / step:.4f}")


import os

if __name__ == '__main__':
    # ---------------- hyper-params & device ----------------
    BATCH_SIZE = 25
    LEVELS = 4
    WAVELET = 'db4'
    NUM_SHAPELETS = 3
    SHAPELET_LENGTH = 100
    num_epochs = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")
    # ---------------- 1) build model ----------------
    model = DGSAD_Model(
        levels=LEVELS,
        wavelet_type=WAVELET,
        num_shapelets=NUM_SHAPELETS,
        shapelet_length=SHAPELET_LENGTH,
    ).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---------------- 2) toy dataset ----------------

    clean_signal = torch.tensor(np.load("D:/single/noisy_signals/ex_synth.npy")[:1000]).T.float()
    noisy_signals_eval = 0.2 * torch.randn_like(clean_signal)
    num_samples = 400
    base_clean_signal = clean_signal[0]
    clean_signals = base_clean_signal.unsqueeze(0).repeat(num_samples, 1)

    noisy_signal = torch.tensor(np.load("D:/single/noisy_signals/ex_synth+Noise.npy")).T.float()

    noisy_signals = clean_signals + 0.1 * torch.randn_like(clean_signals)

    # folder_path = r"D:\single\Landslide_dataset\test_split"
    # # 获取所有 .npy 文件名
    # file_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])
    # # 加载并堆叠为一个大的 numpy 数组
    # data = np.stack([np.load(os.path.join(folder_path, fname)) for fname in file_list])
    # noisy_signals = torch.tensor(data,dtype=torch.float32).to(DEVICE)
    # indices = torch.randperm(noisy_signals.size(0))[:200]
    # noisy_sampled = noisy_signals[indices]

    clean_signals = normalize_signals(clean_signals)
    noisy_signals = normalize_signals(noisy_signals)

    dataset = TensorDataset(noisy_signals, clean_signals)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ---------------- 3) one-shot shapelet --------

    model.dsg.initialize_shapelets(data_loader, model.pim_net)

    torch.autograd.set_detect_anomaly(True)

    # ---------------- 4) train one epoch ----------------
    for epoch in range(1, num_epochs + 1):
        print(f"\n---  Training · epoch {epoch:3d}/{num_epochs} ---")
        train_one_epoch(model, data_loader, optim, DEVICE, print_every=10)

    print("---  Inference demo  ---")
    model.eval()

    with torch.no_grad():
        denoised = model(noisy_signals_eval, training=False)  # (B, 1, T)

    print("Output shape :", denoised.squeeze(1).shape)
