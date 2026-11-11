"""动态尺度锚定核心:D455 高置信掩码 → RANSAC 校准 → 时间平滑。

思想(参照 Gemini 方案):单目深度的绝对尺度会漂移,D455 双目在近场(0.5-4m)
物理精度可靠;用近场重叠区逐帧拟合 yolo→物理 的两参数变换并平滑,再把变换
应用到全图,远场即被"锚定"在当前帧的物理尺度上。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import FusionConfig

REF_Y = 2.0  # 跳变守卫参考点:校准后模型在"模型读数 2m"处应输出的物理深度


@dataclass
class AnchorParams:
    """两参数校准:log 模式 d = e^b·y^a;linear 模式 d = S·y + t。"""

    mode: str
    p1: float  # log: a(指数);linear: S(尺度)
    p2: float  # log: b(对数偏移);linear: t(米制偏移)
    inlier_ratio: float = 0.0
    n_anchor: int = 0
    identity: bool = False

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float32)
        if self.identity:
            return y.copy()
        if self.mode == "log":
            out = np.exp(self.p2) * np.power(np.clip(y, 1e-4, None), self.p1)
        else:
            out = self.p1 * y + self.p2
        return np.clip(out, 0.01, None).astype(np.float32)

    def implied_depth_at(self, y_ref: float = REF_Y) -> float:
        """校准后模型读数 y_ref 对应的物理深度(平滑/守卫的统一指标)。"""
        if self.identity:
            return float(y_ref)
        return float(self.apply(np.array([[y_ref]], np.float32))[0, 0])


IDENTITY = AnchorParams("log", 1.0, 0.0, 0.0, 0, identity=True)


def d455_high_confidence_mask(depth_m: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    """高置信掩码:D455 距离窗内 + 空间梯度平滑(相对梯度限斜率)+ 腐蚀。"""
    d = np.asarray(depth_m, np.float32)
    valid = np.isfinite(d) & (d >= cfg.anchor_dmin) & (d <= cfg.anchor_dmax)
    if not valid.any():
        return valid

    df = np.where(valid, d, 0.0)
    gy, gx = np.gradient(df)
    grad = np.hypot(gx, gy)
    rel = np.divide(grad, d, out=np.full_like(d, np.inf), where=d > 0.01)
    mask = valid & (rel < cfg.max_rel_gradient)

    if cfg.mask_erode_px > 0 and mask.any():
        k = 2 * cfg.mask_erode_px + 1
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return mask


def fit_anchor_params(
    yolo_d: np.ndarray,
    d455_d: np.ndarray,
    mask: np.ndarray,
    cfg: FusionConfig,
    rng: np.random.Generator | None = None,
) -> AnchorParams | None:
    """在掩码重叠区用 RANSAC 拟合两参数校准,失败返回 None(沿用上一帧)。"""
    rng = rng or np.random.default_rng(cfg.rng_seed)
    idx = np.flatnonzero(mask)
    if idx.size < cfg.min_anchor_pixels:
        return None

    y = np.asarray(yolo_d, np.float32).reshape(-1)[idx].astype(np.float64)
    z = np.asarray(d455_d, np.float32).reshape(-1)[idx].astype(np.float64)
    ok = np.isfinite(y) & np.isfinite(z) & (y > 1e-4) & (z > 1e-4)
    y, z = y[ok], z[ok]
    if y.size < cfg.min_anchor_pixels:
        return None

    if cfg.ransac_subsample and y.size > cfg.ransac_subsample:
        sel = rng.choice(y.size, cfg.ransac_subsample, replace=False)
        y_s, z_s = y[sel], z[sel]
    else:
        y_s, z_s = y, z

    log_mode = cfg.fit_mode == "log"
    ly_s, lz_s = (np.log(y_s), np.log(z_s)) if log_mode else (y_s, z_s)
    thresh = np.log1p(cfg.ransac_rel_thresh) if log_mode else cfg.ransac_abs_thresh
    min_span = np.log(1.1) if log_mode else 0.3 * max(z_s.std(), 0.1)
    n = ly_s.size

    # 向量化 RANSAC:一次性生成全部最小样本对并批量评分(float32 压内存带宽)
    lo, hi = (cfg.min_exp, cfg.max_exp) if log_mode else (cfg.min_scale, cfg.max_scale)
    ii = rng.integers(0, n, size=(cfg.ransac_iters, 2))
    i, j = ii[:, 0], ii[:, 1]
    dy = ly_s[j] - ly_s[i]
    p1 = (lz_s[j] - lz_s[i]) / np.where(np.abs(dy) >= min_span, dy, 1.0)
    ok_pair = (np.abs(dy) >= min_span) & (p1 >= lo) & (p1 <= hi)
    p2 = lz_s[i] - p1 * ly_s[i]
    ly32 = ly_s.astype(np.float32)
    lz32 = lz_s.astype(np.float32)
    resid = np.abs(
        p1[:, None].astype(np.float32) * ly32[None, :]
        + p2[:, None].astype(np.float32) - lz32[None, :]
    )
    counts = (resid < np.float32(thresh)).sum(axis=1)
    counts[~ok_pair] = -1
    best = int(np.argmax(counts))
    if counts[best] < max(100, 0.05 * n):  # 共识簇消失 → 锚定失效,沿用上一帧
        return None
    best_inl = (np.abs(p1[best] * ly_s + p2[best] - lz_s) < thresh)
    best_p1, best_p2 = float(p1[best]), float(p2[best])

    # 用全部掩码像素中的内点做最小二乘精修(一元线性闭式解,polyfit 在大数组上太慢)
    ly, lz = (np.log(y), np.log(z)) if log_mode else (y, z)
    pred = best_p1 * ly + best_p2
    inl_full = np.abs(pred - lz) < thresh
    if inl_full.sum() < max(100, 0.05 * y.size):
        inl_full = np.ones_like(pred, bool)
    x_inl, t_inl = ly[inl_full], lz[inl_full]
    n_i = x_inl.size
    sx, sy = x_inl.sum(), t_inl.sum()
    sxx, sxy = float(np.dot(x_inl, x_inl)), float(np.dot(x_inl, t_inl))
    det = n_i * sxx - sx * sx
    if abs(det) > 1e-9:
        cand_p1 = (n_i * sxy - sx * sy) / det
        cand_p2 = (sy - cand_p1 * sx) / n_i
        if lo <= cand_p1 <= hi:
            p1, p2 = cand_p1, cand_p2
        else:
            p1, p2 = best_p1, best_p2
    else:
        p1, p2 = best_p1, best_p2

    return AnchorParams(
        mode=cfg.fit_mode,
        p1=float(p1),
        p2=float(p2),
        inlier_ratio=float(inl_full.mean()),
        n_anchor=int(idx.size),
    )


class AnchorSmoother:
    """EMA 平滑 + 跳变守卫;锚定失效/异常帧沿用上次参数,连续异常才接受新参数。"""

    def __init__(self, cfg: FusionConfig):
        self.cfg = cfg
        self.cur: AnchorParams = IDENTITY
        self._reject_streak = 0

    def update(self, params: AnchorParams | None) -> tuple[AnchorParams, bool]:
        """返回 (平滑后参数, 本帧是否更新)。params=None 表示本帧拟合失败。"""
        if params is None:
            return self.cur, False
        if self.cur.identity:
            self.cur = params
            return params, True

        d_new = params.implied_depth_at()
        d_cur = self.cur.implied_depth_at()
        jumped = abs(d_new - d_cur) / max(abs(d_cur), 1e-6) > self.cfg.jump_guard
        if jumped:
            self._reject_streak += 1
            if self._reject_streak < self.cfg.jump_guard_consecutive:
                return self.cur, False  # 单帧异常:忽略,防一帧坏数据拖垮全图尺度
        else:
            self._reject_streak = 0

        a = self.cfg.ema_alpha
        self.cur = AnchorParams(
            mode=params.mode,
            p1=self.cur.p1 * (1 - a) + params.p1 * a,
            p2=self.cur.p2 * (1 - a) + params.p2 * a,
            inlier_ratio=params.inlier_ratio,
            n_anchor=params.n_anchor,
        )
        return self.cur, True
