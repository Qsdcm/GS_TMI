# 3DGSMR 性能核心因素分析

> 基于论文 *"Three-Dimensional MRI Reconstruction with 3D Gaussian Representations: Tackling the Undersampling Problem"* (TMI 2025) 及本仓库代码的完整回顾。

---

## 一、整体计算流程概览

```
欠采样 k-space ──iFFT──> zero-filled 图像 ──初始化──> 3D 高斯点云 (N个点)
                                                          │
                                                     ┌────▼────┐
                                                     │ Voxelizer │  高斯 → 体素
                                                     └────┬────┘
                                                          │ volume [D,H,W] complex
                                                          ▼
                                            volume × sensitivity_maps (逐线圈)
                                                          │
                                                     ┌────▼────┐
                                                     │  3D FFT  │  图像域 → k-space
                                                     └────┬────┘
                                                          │
                                              与欠采样 k-space 比较 (L2 Loss)
                                                          │
                                                     反向传播 → 更新高斯参数
                                                          │
                                                  每100步 → 自适应密度控制 (分裂/剪枝)
```

---

## 二、论文最优参数 vs 当前代码配置

论文 Table I 在 **8x 加速**下的消融实验最优值（高亮行）：

| 参数 | 论文最优 | 当前 config | 代码位置 | 是否一致 |
|------|---------|-----------|---------|---------|
| M (最大点数) | **200,000** | 400,000 | `configs/default.yaml → max_num_points` | 不一致 |
| k (密度初始化系数) | **0.2** | 0.15 | `gaussian_model.py:253` (硬编码) | 不一致 |
| s (分裂梯度阈值) | **0.01** | 0.005 | `configs/default.yaml → grad_threshold` | 不一致 |
| λ (TV正则权重) | **0.1** | 0.01 | `configs/default.yaml → tv_weight` | 不一致 |

> **注意**：这四个值都与论文不一致，可能需要调整。

---

## 三、各模块关键参数详解

### 3.1 Voxelizer（高斯 → 体素，显存和精度的核心）

**文件**: `gaussian/voxelizer.py`

Voxelizer 将每个高斯点的贡献"溅射"(splatting) 到周围体素上，是整个算法的物理核心。

#### `max_r`（影响半径上限，单位：体素）

计算链路：
```
scale (归一化空间 [-1,1])
  → scale_vox = scale × (D,H,W)/2          转到体素空间
  → radius_vox = max(scale_vox, axis) × 3   3-sigma 范围
  → max_r = ceil(radius_vox), 硬上限 20
```

每个高斯点以自身中心为原点，在 `(2×max_r+1)³` 的立方体网格内计算贡献：

| max_r | 网格边长 | 每点采样数 | 显存影响 |
|-------|--------|----------|---------|
| 3 | 7 | 343 | 极小 |
| 5 | 11 | 1,331 | 小 |
| 10 | 21 | 9,261 | 中等 |
| 20 | 41 | 68,921 | 巨大 |

**降低 max_r 会直接截断大高斯的贡献尾部**。在训练初期（仅501个点，scale 很大），每个高斯需要覆盖大范围，max_r=10 会让体积覆盖不完整，导致性能严重下降。

**论文说法**：使用 3-sigma rule 限制影响范围（Section II-E），半径取最大标准差方向的 3σ。

#### `mahal <= 9.0`（马氏距离截断阈值）

`voxelizer.py` 中 `mask_final = valid_mask & (mahal <= 9.0)`。

- `9.0` 对应 3σ 截断，此处高斯值约为峰值的 1.1%（`exp(-0.5×9) ≈ 0.011`）
- 降低此值（如 4.0 = 2σ）可减少计算量但丢失更多尾部信息
- 论文明确使用 3σ，不建议修改

#### `chunk_size`（分块大小）

- 控制每批并行处理的高斯点数，**只影响显存，不影响精度**
- 当前实现会根据 kernel 大小动态调整 sub-chunk，大 kernel 时自动缩小

---

### 3.2 高斯模型初始化

**文件**: `gaussian/gaussian_model.py`

#### 初始点数 (`initial_num_points`)

- **配置值**: 500（命令行传 501）
- **论文 Table II**: 高加速下用 M=500 初始点 + long-axis splitting 效果最好
- 初始点数少 → 每个点 scale 大 → 覆盖范围广 → max_r 大 → 显存压力大
- 初始点数多 → scale 小 → 细节更好 → 但可能欠拟合低频

#### Scale 初始化方式

`from_image` 方法（`gaussian_model.py:198-264`）：
1. 从 zero-filled 图像中取信号前 10% 的区域采样位置
2. 计算每个点到 3 个最近邻的平均距离 → 作为初始 scale
3. **未使用** config 中的 `initial_scale: 2.0` 参数

> 501 个点分布在 220×256×171 的体积中，邻居距离在归一化空间约 0.1-0.3，对应体素空间约 10-30 个体素，因此初始 radius_vox 可达 30-90，会被 cap 到 max_r=20。

#### 密度初始化系数 k

- **代码位置**: `gaussian_model.py:253` — `densities * 0.15`（硬编码）
- **论文最优**: k = 0.2
- 从 zero-filled 图像对应位置取值，乘以 k 作为初始密度（实部和虚部分别）
- k 太大 → 初始 volume 过亮，可能梯度爆炸
- k 太小 → 梯度信号弱，收敛慢

---

### 3.3 自适应密度控制（分裂与剪枝）

**文件**: `trainers/trainer.py` → `adaptive_density_control()`

这是模型从 500 个点增长到数十万个点的唯一机制，直接决定模型容量。

#### 分裂条件

一个高斯点被分裂必须**同时满足**：
1. **位置梯度范数 > `grad_threshold`**：说明该区域欠拟合
2. **最大 scale > `scale_threshold`**：说明该点足够大，值得一分为二

| 参数 | 配置值 | 论文最优 | 含义 |
|------|--------|---------|------|
| `grad_threshold` | 0.005 | **0.01** | 越小 → 分裂越激进 → 点增长越快 |
| `scale_threshold` | 0.0005 | — | 越小 → 允许更小的点分裂 |

#### 分裂方式：Long-axis Splitting（论文核心贡献之一）

**论文 Section II-D, Fig 2(b)**：

```
分裂前:        ████████████  (一个椭球)
                    ↓
分裂后:    ████           ████  (两个更小的椭球)
           沿最长轴方向分开
```

- 长轴缩放 ×0.6，其余两轴缩放 ×0.85
- 密度值 ×0.6（论文："central values are scaled down by a factor of 0.6"）
- 两个子点沿长轴方向各偏移 1 个 scale 距离

**为什么有效**：避免了原始随机分裂的过多重叠，减少模糊伪影，在高加速因子下尤其明显。

#### 剪枝条件

- 密度绝对值 < `opacity_threshold` (0.01) → 删除
- 最大 scale > `max_scale` (0.5) → 删除（过大的高斯无意义）
- 至少保留 100 个点

#### 密度控制时间窗口

| 参数 | 值 | 含义 |
|------|-----|------|
| `densify_from_iter` | 100 | 前 100 步不分裂，让优化稳定 |
| `densify_until_iter` | 2500 | 2500 步后停止分裂，只做微调 |
| `densify_every` | 100 | 每 100 步检查一次 |
| `max_num_points` | 400,000 | 点数硬上限 |

**增长曲线**：在 100-2500 步之间，每 100 步分裂一次（共 24 次机会），从 500 点增长到最多 400k 点。

---

### 3.4 Loss 函数

**文件**: `losses/losses.py`

#### 公式（论文 Section II-G）

```
L = ||A(X) - b||² + λ · TV(|X|)
```

其中 A = FFT · sensitivity_maps · mask（前向模型），b = 欠采样 k-space。

#### Sum vs Mean Reduction（极其关键）

- **当前实现**: `sum()` reduction
- **为什么必须用 sum**：Loss 用 mean() 后梯度被体素数（~9.6M）除，量级约 1e-7，远小于 `grad_threshold`（0.005-0.01），**导致永远无法触发分裂**，点数停留在 500
- **论文 Fig 5**：Loss 值在 10⁵-10⁶ 量级，验证了使用 sum reduction

#### 各 Loss 项权重

| 参数 | 配置值 | 论文最优 | 作用 |
|------|--------|---------|------|
| `kspace_weight` | 1.0 | 1.0 | k-space 数据一致性（主损失） |
| `image_weight` | 0.0 | 0.0 | 图像域损失（当前关闭） |
| `tv_weight` | 0.01 | **0.1** | TV 正则化，抑制噪声/纹理伪影 |
| `loss_type` | "l2" | L2 | L2 配合 sum 保证正确梯度量级 |

> **TV 注意**：TV loss 也用了 sum reduction，所以 λ=0.1 相对 kspace loss 的实际比重取决于两者的绝对值量级。论文消融实验表明 λ=0.1 是最优平衡点。

---

### 3.5 学习率与优化器

**文件**: `trainers/trainer.py`，`configs/default.yaml`

| 参数 | 值 | 敏感度 | 说明 |
|------|-----|--------|------|
| `position_lr` | 0.001 | 中 | 位置移动速度，太大会震荡 |
| `scale_lr` | 0.005 | 中 | scale 变化速度 |
| `rotation_lr` | 0.001 | 低 | 旋转更新通常较慢 |
| `density_lr` | 0.01 | **高** | 直接决定体素亮度值，最影响 PSNR |
| `lr_scheduler.gamma` | 0.95 | 中 | 每步 lr ×= 0.95，衰减很快 |
| `max_grad_norm` | 1.0 | 中 | 梯度裁剪，防止训练初期梯度爆炸 |

优化器为 **Adam**。分裂/剪枝后需要**重建优化器**（因为参数数量变了），此时 Adam 的动量和方差状态被重置。

---

### 3.6 数据处理与前向模型

**文件**: `data/dataset.py`, `data/transforms.py`

#### Sensitivity Map (CSM) 估计

两种模式（`dataset.py:169-197`）：

1. **无外部 CSM**（当前配置 `csm_path: null`）：
   - GT 生成：从全采样 k-space 估计高分辨率 CSM（用于生成 ground truth，不泄露）
   - 重建输入：从 ACS 区域估计低分辨率 CSM（自校准，避免数据泄露）
2. **有外部 CSM**：直接加载，GT 和重建都用同一份

**CSM 质量直接影响前向模型精度**。ACS 估计的 CSM 分辨率有限，`acs_lines` 越大估计越准但占用更多 k-space 中心区域。

#### 欠采样 Mask

| 参数 | 值 | 含义 |
|------|-----|------|
| `mask_type` | "poisson" | 泊松盘采样，比 random/gaussian 更均匀 |
| `acceleration_factor` | 4 (默认) / 8 (命令行) | 欠采样倍率 |
| `acs_lines` | 32 | ACS 全采样中心区域大小 |

- Mask 在 ky-kz 平面（相位编码方向）欠采样，kx（读出方向）全采样
- 这是标准的 3D MRI 采样模式

#### 前向模型（A 算子）

训练中每步计算：
```
volume → volume × CSM[c] → FFT → pred_kspace[c]  (对每个线圈 c)
```
当前实现为**逐线圈**计算 + gradient checkpointing，峰值显存 O(D×H×W) 而非 O(Coils×D×H×W)。

---

### 3.7 训练调度

| 参数 | 值 | 含义 |
|------|-----|------|
| `max_iterations` | 3000 | 总训练步数 |
| `eval_every` | 100 | 每 100 步评估 PSNR/SSIM |
| `save_every` | 500 | 每 500 步保存 checkpoint |
| `seed` | 42 | 随机种子（影响 mask 生成和初始化采样） |

**论文 Fig 5**：8x 加速约 600 步收敛，10x 加速约 1200 步收敛。当前设 3000 步已足够。

---

## 四、影响性能的因素优先级排序

按对最终 PSNR/SSIM 的影响程度排列：

```
第一梯队（决定能否正常训练）:
  ① Loss reduction: sum vs mean  ← 用 mean 则完全无法分裂，训练失败
  ② grad_threshold             ← 决定是否能触发分裂，值不对则模型容量不足

第二梯队（直接影响重建质量）:
  ③ max_num_points (M)         ← 模型容量上限
  ④ tv_weight (λ)              ← 正则强度，影响细节保留 vs 噪声抑制
  ⑤ density_lr                 ← 密度更新速度，最影响收敛后的 PSNR
  ⑥ 密度初始化系数 k            ← 影响初始质量和收敛速度

第三梯队（影响收敛和细节）:
  ⑦ long-axis splitting vs 随机分裂  ← 高加速下差异显著
  ⑧ max_r (voxelizer)          ← 影响大高��的覆盖完整性
  ⑨ acceleration_factor        ← 任务难度本身
  ⑩ CSM 估计质量 (acs_lines)    ← 影响前向模型精度

第四梯队（微调级别）:
  ⑪ position_lr / scale_lr     ← 影响收敛速度
  ⑫ densify_from/until_iter    ← 分裂时间窗口
  ⑬ scale_threshold            ← 小点分裂门槛
  ⑭ opacity_threshold          ← 剪枝灵敏度
```

---

## 五、代码实现 vs 论文原版的差异

| 方面 | 论文原版 | 本仓库实现 | 影响 |
|------|---------|-----------|------|
| Voxelizer | CUDA kernel + tile-based 并行 | PyTorch 纯 Python + scatter_add_ | 显存高 10 倍以上，需要 gradient checkpointing |
| 显存 | 训练 < 10GB，推理 ~2.4GB | 训练 ~20-40GB（优化后） | 需要 48GB GPU |
| 多线圈前向 | 一次性计算所有线圈 | 逐线圈 + gradient checkpointing | 等价精度，以时间换显存 |
| k (密度系数) | 0.2 | 0.15 (硬编码) | 可能影响初始收敛 |
| grad_threshold (s) | 0.01 | 0.005 | 分裂更激进 |
| tv_weight (λ) | 0.1 | 0.01 | TV 正则弱 10 倍 |
| max_num_points (M) | 200,000 | 400,000 | 可能过多导致过拟合或训练变慢 |
| 评估指标 | SSIM + PSNR + LPIPS | SSIM + PSNR + NMSE（无 LPIPS） | 缺少感知质量评估 |

---

## 六、调参建议

### 快速对齐论文设定

```yaml
# configs/default.yaml 修改建议
gaussian:
  max_num_points: 200000    # 论文 M=200k

adaptive_control:
  grad_threshold: 0.01      # 论文 s=0.01

loss:
  tv_weight: 0.1            # 论文 λ=0.1
```

```python
# gaussian_model.py:253 修改
init_den_r = densities_real.flatten()[indices] * 0.2    # 论文 k=0.2
init_den_i = densities_imag.flatten()[indices] * 0.2
```

### 显存不足时的调整

1. **已实施**: Voxelizer 动态分块 + gradient checkpointing
2. **已实施**: 逐线圈前向 + gradient checkpointing
3. **已实施**: 释放 kspace_full 和 zero_filled
4. **如仍 OOM**: 可降低 `max_num_points` 到 200k（也对齐论文）

---

## 七、关键代码文件索引

| 文件 | 核心功能 |
|------|---------|
| `gaussian/voxelizer.py` | 高斯 → 体素的 splatting 过程 |
| `gaussian/gaussian_model.py` | 高斯参数管理、初始化、分裂/剪枝 |
| `trainers/trainer.py` | 训练循环、前向模型、自适应控制 |
| `losses/losses.py` | K-space Loss + Image Loss + TV Loss |
| `data/dataset.py` | 数据加载、CSM 估计、Mask 生成、欠采样 |
| `data/transforms.py` | 3D FFT/iFFT |
| `data/mask_generator.py` | Variable Density Poisson-Disc 采样 |
| `metrics/metrics.py` | PSNR / SSIM / NMSE 评估 |
| `configs/default.yaml` | 所有可配置参数 |
