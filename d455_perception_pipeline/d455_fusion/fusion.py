"""分层空间融合(Zone-based Multiplexing)。

近场核心区(0-near_end):D455 物理深度 100% 优先(毫米级安全);
过渡融合区(near_end-far_start):按 smoothstep 权重混合,无断崖;
远场规划区(far_start 以上):100% 校准后 YOLO26 单目深度。
D455 失效像素(0/NaN)权重置 0 → 由校准后 YOLO 补洞(可关)。

防重影(治 D455 孤立噪声斑块浮在 YOLO 场上、深度边缘双轮廓):
1) 有效密度门控:D455 权重 × 邻域有效率,孤立小岛(周围大量空洞)自动让位给 YOLO;
2) 有效区边界腐蚀羽化:斑块边缘若干像素权重递减,过渡交给 YOLO,消除双轮廓。
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import FusionConfig


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def zone_weights(d455_m: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    """D455 权重图:
    - 远近分区: 近场 1 → 4-7m smoothstep 递减 → 远场 0;
    - 近端盲区平滑: <0.5m 平滑让步向 YOLO, 彻底消除 <0.6m 硬壳断崖;
    - 孤立小斑点抑制: 连通域面积过滤 (<min_speckle_area) 权重归零;
    - 有效密度门控: 邻域有效率低的小岛衰减;
    - 深度阶跃边缘羽化: 物体边缘/运动轮廓大跳变处剥离 2-4px, 消除双轮廓与拉扯重影;
    - 空洞边界羽化: 距无效空洞边缘平滑递减。
    """
    d = np.asarray(d455_m, np.float32)
    valid = np.isfinite(d) & (d > 0.05)
    w = np.zeros_like(d)
    if not valid.any():
        return w

    # 1. 远近分区主权重 (4-7m smoothstep 线性过渡)
    w[valid] = _smoothstep((cfg.far_start - d[valid]) / (cfg.far_start - cfg.near_end))

    # 2. 近端盲区平滑过渡(治 <0.6m 近盲区硬壳断崖与超视差飞点)
    if cfg.near_blind_max > cfg.near_blind_min:
        near_factor = _smoothstep(
            (d[valid] - cfg.near_blind_min) / (cfg.near_blind_max - cfg.near_blind_min)
        )
        w[valid] *= near_factor

    is_2d_img = (d.ndim == 2 and d.shape[0] >= 16 and d.shape[1] >= 16)

    # 3. 连通域斑点过滤(Speckle Rejection: 消除散乱孤立斑点浮在 YOLO 上)
    if cfg.speckle_filter and is_2d_img and valid.any():
        v_u8 = valid.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(v_u8, connectivity=8)
        small_mask = np.zeros_like(valid)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < cfg.min_speckle_area:
                small_mask[labels == lbl] = True
        if small_mask.any():
            w[small_mask] = 0.0
            valid = valid & (~small_mask)

    # 4. 有效密度门控: 邻域有效率低的小岛衰减
    if cfg.density_gate and valid.any():
        v32 = valid.astype(np.float32)
        density = cv2.blur(v32, (cfg.density_ksize, cfg.density_ksize))
        w *= _smoothstep((density - cfg.density_lo) / (cfg.density_hi - cfg.density_lo))

    # 5. 深度阶跃边缘羽化(消除物体轮廓与背景交界处的双目拉扯飞点和双重虚影)
    if cfg.depth_jump_feather and is_2d_img and valid.any():
        d_val = np.where(valid, d, 0.0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        d_max = cv2.dilate(d_val, kernel)
        d_min = cv2.erode(np.where(valid, d, 999.0).astype(np.float32), kernel)
        jump = (d_max - d_min) / np.maximum(d_val, 0.1)
        jump_edge = valid & (jump > cfg.depth_jump_thresh) & (d_min < 900.0)
        if jump_edge.any() and cfg.depth_jump_dilate_px > 0:
            k_edge = 2 * cfg.depth_jump_dilate_px + 1
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_edge, k_edge))
            edge_band = cv2.dilate(jump_edge.astype(np.uint8), dil_kernel).astype(bool)
            w[edge_band] *= 0.05

    # 6. 有效区边界空洞羽化
    if cfg.fusion_edge_feather_px > 0 and not valid.all() and valid.any():
        dist_in = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
        w *= _smoothstep(dist_in / cfg.fusion_edge_feather_px)

    if not cfg.fill_near_holes:
        strict = valid & (d <= cfg.near_end) & (d >= cfg.near_blind_max)
        w[strict] = 1.0
    return w.astype(np.float32)


def fuse(
    depth_d455_m: np.ndarray,
    depth_yolo_corrected: np.ndarray,
    cfg: FusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 (融合深度图, D455权重图)。"""
    d = np.asarray(depth_d455_m, np.float32)
    y = np.asarray(depth_yolo_corrected, np.float32)
    if y.shape != d.shape:
        raise ValueError(f"shape mismatch: d455 {d.shape} vs yolo {y.shape}")
    w = zone_weights(d, cfg)
    fused = w * d + (1.0 - w) * y
    if not cfg.fill_near_holes:
        # 严格模式:近场洞保持 0(无效语义),用校准后 YOLO 判断洞的位置
        hole = ~np.isfinite(d) | (d <= 0.05)
        fused[hole & (y <= cfg.near_end)] = 0.0
    return fused.astype(np.float32), w
