"""端到端管线:RGB + D455 物理深度 → 锚定校准 → 三区融合。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .alignment import AnchorParams, AnchorSmoother, d455_high_confidence_mask, fit_anchor_params
from .backends import Yolo26DepthBackend
from .config import FusionConfig
from .fusion import fuse


@dataclass
class FrameResult:
    fused: np.ndarray
    yolo_raw: np.ndarray
    yolo_corrected: np.ndarray
    weight: np.ndarray
    mask: np.ndarray
    params: AnchorParams
    updated: bool
    stats: dict = field(default_factory=dict)


class DepthFusionPipeline:
    def __init__(
        self,
        cfg: FusionConfig | None = None,
        backend: Yolo26DepthBackend | None = None,
    ):
        self.cfg = cfg or FusionConfig()
        self.backend = backend or Yolo26DepthBackend(
            self.cfg.weights, self.cfg.imgsz, self.cfg.device
        )
        self.smoother = AnchorSmoother(self.cfg)
        self._rng = np.random.default_rng(self.cfg.rng_seed)

    def process(self, rgb: np.ndarray, d455_m: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        yolo_raw = self.backend(rgb)
        if yolo_raw.shape != np.asarray(d455_m).shape:
            yolo_raw = cv2.resize(
                yolo_raw, (d455_m.shape[1], d455_m.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        t_model = time.perf_counter() - t0

        t1 = time.perf_counter()
        mask = d455_high_confidence_mask(d455_m, self.cfg)
        raw_params = fit_anchor_params(yolo_raw, d455_m, mask, self.cfg, self._rng)
        params, updated = self.smoother.update(raw_params)
        yolo_corrected = params.apply(yolo_raw)
        fused, w = fuse(d455_m, yolo_corrected, self.cfg)
        t_align = time.perf_counter() - t1

        return FrameResult(
            fused=fused,
            yolo_raw=yolo_raw,
            yolo_corrected=yolo_corrected,
            weight=w,
            mask=mask,
            params=params,
            updated=updated,
            stats={
                "model_ms": t_model * 1e3,
                "align_ms": t_align * 1e3,
                "anchor_pixels": int(mask.sum()),
                "inlier_ratio": float(params.inlier_ratio),
                "n_anchor": int(params.n_anchor),
                "p1": float(params.p1),
                "p2": float(params.p2),
                "implied_at_ref": params.implied_depth_at(),
                "updated": updated,
            },
        )
