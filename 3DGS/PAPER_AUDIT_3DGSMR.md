# 3DGSMR 论文对照差异审计报告

本文档记录本仓库在改造前后的论文对齐情况。表中 `Current Behavior` 指本次改造前仓库行为，`Required Change` 指为论文忠实复现所执行或必须执行的改动方向。

| 1. Module / File Path | 2. Current Behavior | 3. Target Behavior from Paper | 4. Gap / Risk | 5. Required Change | 6. Whether paper explicitly specifies it or not | 7. Confidence level |
| --- | --- | --- | --- | --- | --- | --- |
| `3dgsVC/gaussian/gaussian_model.py` | 已有 12 参数高斯与复数 density，但默认 `importance` 初始化；大规模 scale 初始化是近似；original split 用工程化位移 | iFFT 初始化；随机采样 M 个 grid points；3-NN scale；无旋转；density 乘 `k`；保留 low-mid original 与 high-acc long-axis 两分支 | 默认初始化不忠实；部分 split 位移是推断实现 | 默认切到 `random`；补 exact grid 3-NN scale；保留 `importance` 仅作非论文模式；long-axis offset 作为 inferred 配置项 | 初始化主干明确；child 位移未明确 | High |
| `3dgsVC/gaussian/tile_voxelizer.py` | 旧版 `tile_cuda` 可静默回退，且 CUDA 路径不是真正 tile 主路径 | Tile 划分、3σ bbox、tile-Gaussian pairs、排序/分组、per-tile 聚合、正式训练主路径走 CUDA | 名称与真实行为不一致，存在“假 CUDA”风险 | 重写成 strict `tile_cuda` 加 reference `tile`；论文模式下禁止静默回退 | 明确 | High |
| `3dgsVC/gaussian/cuda_ext/tile_voxelize_cuda_kernel.cu` | 旧版是 per-Gaussian scatter/gather | 真正按 tile block 组织 CUDA 前后向 | 与 Section E 核心设计不符 | 重写 kernel；使用 tile block + staged Gaussian batches；保留 reference backend 做 parity | 明确 | High |
| `3dgsVC/gaussian/voxelizer.py` | chunk/global sparse accumulation | 仅作为 reference naive/chunk backend | 若主路径落到这里就是替代实现 | 降级为 reference/debug/test backend | 明确 | High |
| `3dgsVC/data/dataset.py` | multicoil 与 CSM 存在，但默认 mask 不是论文主路径；readout/phase 轴未显式配置 | 默认 stacked 2D Gaussian mask；readout fully sampled；phase undersampled；ACS fully sampled；保留 sensitivity maps | 默认配置与论文不符；轴语义不透明 | 显式 `readout_axis` / `phase_axes`；默认 mask 改为 `stacked_2d_gaussian`；增加几何校验 | mask 几何明确；CSM 估计细节未明确 | Medium-High |
| `3dgsVC/data/mask_generator.py` | legacy Poisson 保留 | 可保留为额外功能，但不能冒充论文主路径 | 若仍是默认会误导复现入口 | 仅保留为 non-paper 附加模式 | 论文未要求 Poisson | High |
| `3dgsVC/data/transforms.py` | centered FFT/iFFT 已基本正确 | complex 3D FFT / iFFT 用于 forward model 和初始化 | 主要风险是缺测试 | 补 forward / dtype / complex consistency 测试 | 明确 | High |
| `3dgsVC/losses/losses.py` / `3dgsVC/losses/kspace_loss.py` | 旧版模块组织混合，image loss 入口仍暴露 | 主损失应为 k-space consistency + `λ TV(|X|)` | 非论文 image loss 若进入 paper mode 会污染实验 | 重构为显式 paper loss；paper mode 禁用 image loss；保留 legacy alias | 明确 | High |
| `3dgsVC/metrics/metrics.py` | 只有 PSNR / SSIM / NMSE；无 LPIPS；无 plateau 监控 | 复现实验记录 SSIM / PSNR / LPIPS，并用于 plateau stopping | 指标不完整，无法按论文语义停机 | 增加 LPIPS 接口与 plateau stopper；LPIPS 依赖缺失时报清晰错误 | 指标集合明确；LPIPS 3D 聚合方式未明确 | High |
| `3dgsVC/trainers/trainer.py` | 旧逻辑混合 GT 实验评估与通用训练；8x 可被 `auto + M<=1000` 误切到高加速分支 | 分清 paper reproduction 和 self-supervised deploy；low-mid / high-acc 两条策略可追溯；GT plateau 与无 GT 模式分离 | 语义混杂会导致“用非论文入口声称复现论文” | 新增 mode 控制、strict CUDA、metric plateau、loss plateau；8x 论文默认回到 `M=200k + original` | 分支语义明确；plateau delta/patience 未明确 | High |
| `3dgsVC/train.py` / `3dgsVC/test.py` | 训练后默认自动跑 GT 测试；CUDA tuple 输出在测试入口不稳；post-train 行为未区分 mode | 训练/测试入口要跟随 mode 与 paper configs，不能把 GT 评估强绑到 self-supervised 流程 | 入口误导严重 | 改成 mode-controlled post-train test；推理入口支持 `tile_cuda` tuple 与 LPIPS | 明确 | High |
| `3dgsVC/configs/default.yaml` / `paper_lowmid.yaml` / `paper_highacc.yaml` / `legacy_nonpaper.yaml` | 旧默认是 `poisson + chunk + importance` | 默认应是论文 low-mid 主路径；高加速另设独立 profile；legacy 单列 | 默认入口会误导实验结论 | 新增 paper configs，默认切到 `paper_lowmid` | 明确 | High |
| `train.sh` / `test.sh` / `3dgsVC/scripts/*.sh` | 旧脚本示例和硬编码路径不符合当前仓库 | 一键入口应指向 paper configs，并保留 legacy override 的可追踪性 | 会让用户默认跑偏 | 改成 paper-lowmid / paper-highacc 风格入口；移除仓库外硬编码路径 | 明确 | High |
| `tests/` / `benchmarks/` | 改造前几乎没有正式单测和 benchmark | 必须有 complex/init/mask/forward/voxelizer/adaptive/config/CUDA parity+perf | 无法证明“忠实复现 + 主路径可用” | 新增最小有效测试与 benchmark | 明确 | High |

## 已符合论文的部分
- 复数高斯参数化主线已经存在。
- centered 3D FFT / iFFT 主线已经存在。
- multicoil forward model 中的 sensitivity maps 未被删除。
- TV 已经在实现上可作用到 magnitude，而不是强行拆成 real/imag 单独 TV。

## 改造前不符合论文的关键点
- 默认配置不是论文主路径。
- `tile_cuda` 名称与真实行为不一致，存在静默回退。
- 8x 默认入口可能被 `M<=1000` 拉到高加速策略。
- 缺少 LPIPS 与 plateau stopping。
- 缺少 paper reproduction vs self-supervised deploy 语义隔离。

## 推断项（paper unspecified）
- long-axis split 子点精确位移量：当前通过 `long_axis_offset_factor` 暴露为配置，默认取 1.0，相对 child longest-axis scale。
- plateau 停机的 `patience_evals`、`metric_min_delta`、`min_iterations_before_stop`。
- LPIPS 的 3D 聚合方式：当前采用逐切片平均，属于实现推断项。
- 仓库内 CSM 估计沿用 external CSM 优先、ACS fallback；论文未给出完整 estimation pipeline 细节。
