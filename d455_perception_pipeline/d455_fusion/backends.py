"""YOLO26-Depth 推理后端:单 RGB → 每像素米制深度图。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class Yolo26DepthBackend:
    def __init__(self, weights, imgsz: int = 768, device: str = ""):
        self.model = YOLO(str(weights))
        if self.model.task != "depth":
            raise ValueError(f"{weights} task={self.model.task!r}, 需要 depth 模型")
        self.imgsz = imgsz
        self.device = device or (0 if torch.cuda.is_available() else "cpu")
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((480, 848, 3), np.uint8)
        self(dummy)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: HxWx3 uint8(RGB 序)→ float32 HxW 米。"""
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        res = self.model.predict(
            bgr, imgsz=self.imgsz, device=self.device, verbose=False
        )[0]
        dep = getattr(res, "depth", None)
        if dep is None:
            raise RuntimeError("结果对象无 depth 字段,检查 ultralytics 版本(需支持 depth 任务)")
        data = getattr(dep, "data", dep)
        if hasattr(data, "detach"):
            data = data.detach().float().cpu().numpy()
        d = np.asarray(data, np.float32)
        if d.ndim == 3:
            d = d.squeeze()
        if d.shape != rgb.shape[:2]:
            d = cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        return d
