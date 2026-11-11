"""可视化:深度彩图 + 五联面板(RGB / D455 / YOLO 原始 / YOLO 校准 / 融合)。"""
from __future__ import annotations

import cv2
import numpy as np

from .alignment import AnchorParams

FONT = cv2.FONT_HERSHEY_SIMPLEX


def colorize_depth(d: np.ndarray, dmax: float = 12.0) -> np.ndarray:
    """米制深度 → TURBO 彩图(BGR);无效(<=0/NaN)为黑。"""
    d = np.asarray(d, np.float32)
    x = np.clip(d / max(dmax, 1e-6), 0.0, 1.0)
    invalid = ~np.isfinite(d) | (d <= 0)
    x[invalid] = 0.0
    cm = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[invalid] = 0
    return cm


def fusion_panel(
    rgb: np.ndarray,
    d455_m: np.ndarray,
    yolo_raw: np.ndarray,
    yolo_corrected: np.ndarray,
    fused: np.ndarray,
    params: AnchorParams,
    stats: dict,
    cfg=None,
) -> np.ndarray:
    dmax = getattr(cfg, "viz_dmax", 12.0)
    h, w = fused.shape[:2]

    rgb_bgr = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, ::-1]), cv2.COLOR_RGB2BGR)

    def panel(img_bgr, label):
        p = img_bgr.copy()
        cv2.rectangle(p, (0, 0), (w, 26), (30, 30, 30), -1)
        cv2.putText(p, label, (8, 19), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return p

    panels = [
        panel(rgb_bgr, "RGB (yellow=anchor mask 0.5-4m)"),
        panel(colorize_depth(d455_m, dmax), "D455 STEREO (black=invalid)"),
        panel(colorize_depth(yolo_raw, dmax), "YOLO26-DEPTH RAW"),
        panel(colorize_depth(yolo_corrected, dmax), "YOLO26 ANCHORED (calibrated)"),
        panel(colorize_depth(fused, dmax), "FUSED (0-4m D455 | 4-7m blend | 7m+ YOLO)"),
        panel(np.zeros((h, w, 3), np.uint8), ""),
    ]

    # 锚定掩码轮廓叠到 RGB 面板(黄)
    mask = stats.get("mask")
    if mask is not None and mask.shape == (h, w):
        m8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panels[0], cnts, -1, (0, 230, 255), 1)

    top = np.hstack(panels[:3])
    bottom = np.hstack(panels[3:])

    hud_h = 54
    hud = np.zeros((hud_h, top.shape[1], 3), np.uint8)
    line1 = (
        f"mode={params.mode} a/S={params.p1:.3f} b/t={params.p2:+.3f}  "
        f"inlier={stats.get('inlier_ratio', 0):.0%}  anchor={stats.get('anchor_pixels', 0)}px  "
        f"upd={stats.get('updated', False)}  model={stats.get('model_ms', 0):.0f}ms  "
        f"align={stats.get('align_ms', 0):.1f}ms"
    )
    line2 = f"implied depth @model-2m: {stats.get('implied_at_ref', 2.0):.2f} m"
    cv2.putText(hud, line1, (8, 22), FONT, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(hud, line2, (8, 45), FONT, 0.55, (160, 220, 255), 1, cv2.LINE_AA)

    return np.vstack([top, bottom, hud])
