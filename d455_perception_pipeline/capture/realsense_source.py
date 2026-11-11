"""帧源:RealSense 实况/回放 + 合成 Mock(自测/离线验证用,不依赖相机)。"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np


class FrameSource:
    def read(self) -> tuple[np.ndarray, np.ndarray, float]:
        """返回 (rgb uint8 HxWx3, depth_m float32 HxW 米(0=无效), ts)。"""
        raise NotImplementedError

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RealSenseSource(FrameSource):
    """D455 对齐帧源:RGB 与深度时间同步+像素对齐;color 分辨率下输出。

    官方后处理链(Intel 白皮书建议):视差域边缘保持滤波
    dispaty_transform → spatial → temporal → 反变换 —— 治重影/边缘晕圈/噪声。
    近端盲区官方调法:① 深度 X 分辨率减半 → minZ 线性减半;② 高级模式
    Disparity Shift(远端量程换近端,本架构远场由 YOLO 负责,代价可承受)。
    """

    def __init__(
        self,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
        temporal: bool = True,
        serial: str | None = None,
        bag: str | None = None,
        depth_width: int | None = None,   # None=与 color 同分辨率;424=盲区减半
        disparity_shift: int = 0,         # 高级模式 Disparity Shift(0=关闭)
        spatial: bool = True,
    ):
        import pyrealsense2 as rs

        self.rs = rs
        if bag is None:
            # 无条件显式设置深度表(STDepthTable 持久化在设备上,必须显式归零,
            # 否则上次会话的 shift 会泄漏到本次乃至其他程序)
            self._apply_disparity_shift(rs.context().query_devices()[0], disparity_shift)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if bag:
            cfg.enable_device_from_file(bag, repeat_playback=False)
        else:
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            dw, dh = (depth_width, depth_width * height // width) if depth_width else (width, height)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
        profile = self.pipe.start(cfg)
        self.depth_scale = (
            profile.get_device().first_depth_sensor().get_depth_scale()
        )  # 米/LSB, D455 默认 0.001
        self.align = rs.align(rs.stream.color)

        # 官方权威滤波链: 截断滤波 → 进视差域(1/d) → 空间边缘保持 → 时域滤波 → 回深度域
        # 1. 阈值截断: 消除近端视差饱和溢出产生的漂移伪影与飞点
        self.thresh_f = rs.threshold_filter()
        self.thresh_f.set_option(rs.option.min_distance, 0.30)
        self.thresh_f.set_option(rs.option.max_distance, 8.0)

        # 2. 视差域转换
        self.to_disp = rs.disparity_transform(True)
        self.from_disp = rs.disparity_transform(False)
        self.spatial_f = rs.spatial_filter()
        if spatial:
            self.spatial_f.set_option(rs.option.filter_magnitude, 2)
            self.spatial_f.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_f.set_option(rs.option.filter_smooth_delta, 20)

        # 3. 时域滤波(严禁使用默认 holes_fill=3, 否则手移开后过去4帧里有2帧有效会强行画残影)
        self.temporal = rs.temporal_filter() if temporal else None
        if self.temporal is not None:
            self.temporal.set_option(rs.option.holes_fill, 1)  # 仅持续有效才平滑, 杜绝幽灵拖尾
            self.temporal.set_option(rs.option.filter_smooth_alpha, 0.65)  # 提高动态响应度
            self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        self.is_live = bag is None

    def _apply_disparity_shift(self, device, shift: int):
        """官方 Advanced Mode 深度表:增大 disparityShift 降低 minZ,远端量程同步收窄。"""
        try:
            adv = self.rs.rs400_advanced_mode(device)
            if not adv.is_enabled():
                adv.toggle_advanced_mode(True)
                time.sleep(0.3)
            table = adv.get_depth_table()
            old = int(table.disparityShift)
            table.disparityShift = int(shift)
            adv.set_depth_table(table)
            print(f"[realsense] Disparity Shift {old} -> {shift}(minZ 降低,远端量程同步缩短)")
        except Exception as e:
            print(f"[realsense] Disparity Shift 应用失败(忽略): {e}")

    def read(self):
        for _ in range(20):  # 高级模式切换/流重配置期会出空帧,重试
            frames = self.pipe.wait_for_frames()
            aligned = self.align.process(frames)
            dframe = aligned.get_depth_frame()
            if not dframe:
                continue
            # 官方标准链: 截断 → 视差域 → 空间滤波 → 时域滤波 → 回深度域
            dframe = self.thresh_f.process(dframe)
            dframe = self.to_disp.process(dframe)
            dframe = self.spatial_f.process(dframe)
            if self.temporal is not None:
                dframe = self.temporal.process(dframe)
            dframe = self.from_disp.process(dframe)
            if not dframe:
                continue
            depth_m = np.asanyarray(dframe.get_data()).astype(np.float32) * self.depth_scale
            rgb = np.asanyarray(aligned.get_color_frame().get_data())[:, :, ::-1]  # BGR→RGB
            return rgb, np.ascontiguousarray(depth_m), time.time()
        raise RuntimeError("连续 20 帧未取到有效深度帧(检查相机连接/USB 供电)")

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass

    def __del__(self):
        self.close()


class MockSource(FrameSource):
    """合成走廊场景(透视正确纹理,NYU 风格,深度网络可识别其结构):

    地面(相机高 0.9m)+ 左右墙(走廊半宽 1.2m)+ 尽头墙 6.5m
    + 障碍箱 2.0m / 3.5m + 9m 立柱(堵走廊,远场目标)。
    D455 仿真:仅 0.4-4.2m 有效(高斯噪声 + 1% 反光离群斑),其余置 0。
    true_depth 记录每帧无噪声真值,供测试对照。
    """

    W, H = 848, 480
    FX = FY = 424.0
    CX, CY = 424.0, 240.0
    CAM_H = 0.9      # 相机离地高度
    HALF_W = 1.2     # 走廊半宽
    END_Z = 6.5      # 尽头墙

    def __init__(self, noise_sigma: float = 0.01, outlier_ratio: float = 0.01, seed: int = 7):
        self.noise_sigma = noise_sigma
        self.outlier_ratio = outlier_ratio
        self.rng = np.random.default_rng(seed)
        self.i = 0
        self.true_depth: np.ndarray | None = None

    def _scene_depth(self, frame_i: int) -> np.ndarray:
        dx = int(3 * np.sin(0.15 * frame_i))  # 相机横摆
        v = np.arange(self.H, dtype=np.float32)[:, None]
        u = np.arange(self.W, dtype=np.float32)[None, :] - dx

        z_floor = np.where(
            v > self.CY, self.CAM_H * self.FY / np.maximum(v - self.CY, 1e-3), np.inf
        )
        z_left = np.where(
            u < self.CX, self.HALF_W * self.FX / np.maximum(self.CX - u, 1e-3), np.inf
        )
        z_right = np.where(
            u > self.CX, self.HALF_W * self.FX / np.maximum(u - self.CX, 1e-3), np.inf
        )
        z = np.minimum(np.minimum(z_floor, z_left), np.minimum(z_right, self.END_Z))

        def put(z0, v0, v1, c0, c1):
            c0, c1 = max(c0 + dx, 0), min(c1 + dx, self.W)
            if c1 > c0:
                z[v0:v1, c0:c1] = z0

        put(9.0, 60, 420, 354, 494)     # 立柱(远场 9m,堵走廊)
        put(2.0, 300, 430, 334, 514)    # 障碍箱 A(近场)
        put(3.5, 350, 445, 470, 620)    # 障碍箱 B(中景)
        return z

    def _scene_rgb(self, depth: np.ndarray, frame_i: int) -> np.ndarray:
        rng = np.random.default_rng(1000 + frame_i)
        dx = int(3 * np.sin(0.15 * frame_i))
        v = np.arange(self.H, dtype=np.float32)[:, None]
        u = np.arange(self.W, dtype=np.float32)[None, :] - dx
        z = np.maximum(depth, 1e-3)
        x_w = (u - self.CX) * z / self.FX   # 世界坐标
        y_w = (v - self.CY) * z / self.FY

        is_floor = np.abs(y_w - self.CAM_H) < 0.08
        is_lwall = np.abs(x_w + self.HALF_W) < 0.08
        is_rwall = np.abs(x_w - self.HALF_W) < 0.08

        tile_floor = ((np.floor(x_w / 0.6) + np.floor(z / 0.6)) % 2).astype(np.float32)
        tile_wall = ((np.floor(y_w / 0.5) + np.floor(z / 0.5)) % 2).astype(np.float32)

        img = np.full((self.H, self.W, 3), 0.48, np.float32)          # 尽头墙:平灰
        wall_col = 0.42 + 0.28 * tile_wall[..., None]
        img[is_lwall] = np.tile(wall_col[is_lwall], (1, 3)) * np.array([0.9, 0.95, 1.1])
        img[is_rwall] = np.tile(wall_col[is_rwall], (1, 3)) * np.array([1.1, 0.95, 0.9])
        floor_col = 0.40 + 0.35 * tile_floor[..., None]
        img[is_floor] = np.tile(floor_col[is_floor], (1, 3)) * np.array([1.05, 1.0, 0.85])

        # 障碍着色(按深度带)
        img[(depth > 1.9) & (depth < 2.1)] = [0.75, 0.35, 0.25]
        img[(depth > 3.4) & (depth < 3.6)] = [0.25, 0.55, 0.75]
        img[(depth > 8.8) & (depth < 9.2)] = [0.8, 0.65, 0.2]
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        return (img * 255).astype(np.uint8)

    def read(self):
        true_z = self._scene_depth(self.i)
        valid = (true_z >= 0.4) & (true_z <= 4.2)
        d455 = np.where(valid, true_z, 0.0).astype(np.float32)
        noise = self.rng.normal(0, self.noise_sigma, d455.shape).astype(np.float32)
        d455 = np.where(valid, d455 + noise, d455)
        out_mask = valid & (self.rng.random(d455.shape) < self.outlier_ratio)
        d455[out_mask] *= 0.4  # 模拟反光斑:读数偏近
        rgb = self._scene_rgb(true_z, self.i)
        self.true_depth = true_z
        self.i += 1
        return rgb, np.ascontiguousarray(d455), time.time()
