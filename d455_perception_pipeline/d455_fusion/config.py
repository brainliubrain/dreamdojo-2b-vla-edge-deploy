"""管线全局配置(距离单位:米)。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WEIGHTS = Path(
    os.environ.get("D455_FUSION_WEIGHTS", "D:/AI/d455_fusion/models/yolo26m-depth.pt")
)


@dataclass
class FusionConfig:
    # ---- D455 高置信锚定掩码(物理深度可信区) ----
    anchor_dmin: float = 0.5        # D455 近盲区以下不锚定
    anchor_dmax: float = 4.0        # 远场双目衰减,锚定只取 0.5-4m
    max_rel_gradient: float = 0.15  # |∇d|/d 上限,剔除斜面/深度边缘像素
    mask_erode_px: int = 3          # 掩码腐蚀半径,防边缘渗色
    min_anchor_pixels: int = 800    # 少于此数视为锚定失效,沿用上一帧参数

    # ---- RANSAC 尺度/偏移校准 ----
    fit_mode: str = "linear"        # "linear": d = S·y + t(Gemini 原方案,外推有界,默认)
                                    # "log": log d = a·log y + b(与深度头校准同形,
                                    #        域内锚定区略优,但域外远场外推会爆炸,慎用)
    ransac_iters: int = 512
    ransac_rel_thresh: float = 0.03 # log 模式内点阈值(相对误差 3%)
    ransac_abs_thresh: float = 0.05 # linear 模式内点阈值(绝对误差 5cm)
    ransac_subsample: int = 8192    # RANSAC 评分像素上限(float32 矩阵,保实时)
    min_exp: float = 0.25           # log 模式指数 a 合理区间(仅拒退化,不预设模型好坏)
    max_exp: float = 3.5
    min_scale: float = 0.1          # linear 模式 S 下限
    max_scale: float = 5.0          # linear 模式 S 上限(防止病态外推爆炸,容纳合成走廊跨度)
    rng_seed: int = 0

    # ---- 时间平滑(EMA + 跳变守卫) ----
    ema_alpha: float = 0.2          # 平滑系数:越小越稳,越大越跟手
    jump_guard: float = 0.25        # 参考点读数跳变超 25% 视为异常帧
    jump_guard_consecutive: int = 3 # 连续 3 帧跳变才认定为场景真实变化并接受

    # ---- 分区混合 ----
    near_end: float = 4.0           # 0-4m:100% D455(有洞时可选由 YOLO 补)
    far_start: float = 7.0          # 4-7m:smoothstep 权重过渡;7m+:100% YOLO 校准值
    fill_near_holes: bool = True    # 近场 D455 失效像素用校准后 YOLO 深度填补

    # ---- 防重影(D455 孤立噪声斑块 / 深度边缘双轮廓 / 动态残影) ----
    density_gate: bool = True       # 有效密度门控:孤立小岛权重趋 0,让位给 YOLO
    density_ksize: int = 31         # 邻域有效率统计窗(像素)
    density_lo: float = 0.10        # 有效率 ≤ 此值 → D455 权重 0
    density_hi: float = 0.45        # 有效率 ≥ 此值 → 不衰减
    fusion_edge_feather_px: float = 4.0  # D455 空洞边界羽化半径(0=关)

    # 近端盲区平滑过渡(治 <0.6m 近盲区硬壳断崖与超视差飞点)
    near_blind_min: float = 0.32    # 低于此距离 D455 物理不可信, 100% 由 YOLO 单目接管
    near_blind_max: float = 0.50    # 0.32~0.50m 之间 smoothstep 平滑过渡

    # 孤立碎块斑点剔除(Speckle Rejection, 消除散乱斑点浮在 YOLO 上)
    speckle_filter: bool = True     # 开启连通域斑点过滤
    min_speckle_area: int = 150     # 像素数小于此阈值的孤立连通块置零让位 YOLO

    # 深度阶跃边缘羽化(治物体轮廓/快速运动物体与背景交界处的双目拉扯飞点和双重虚影)
    depth_jump_feather: bool = True # 开启前景物体轮廓边缘阶跃羽化
    depth_jump_thresh: float = 0.20 # 相对深度阶跃阈值 |Δd|/d > 20% 视为遮挡边缘
    depth_jump_dilate_px: int = 3   # 遮挡过渡带剥离半径(像素)

    # ---- 模型 ----
    weights: Path = field(default_factory=lambda: DEFAULT_WEIGHTS)
    imgsz: int = 768                # YOLO26-Depth 出厂训练/推理分辨率
    device: str = ""                # ""=自动(cuda 可用则 0,否则 cpu)

    # ---- 可视化 ----
    viz_dmax: float = 12.0          # 彩色图深度上限(米)

    def __post_init__(self):
        assert self.fit_mode in ("log", "linear"), self.fit_mode
