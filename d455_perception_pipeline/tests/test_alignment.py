"""B1 对齐模块单测:掩码 / RANSAC 恢复已知变换 / 平滑守卫。纯 numpy,无模型。"""
import numpy as np

from d455_fusion.alignment import (
    AnchorParams,
    AnchorSmoother,
    d455_high_confidence_mask,
    fit_anchor_params,
)
from d455_fusion.config import FusionConfig

H, W = 480, 848


def make_scene() -> np.ndarray:
    """合成 D455 深度:平坦地面 1.2-4.0m + 近台 0.8m + 近距断崖(应剔除)+ 无效区。"""
    d = np.zeros((H, W), np.float32)
    rows = np.arange(300, 461)
    d[rows, :] = np.linspace(1.2, 4.0, rows.size).astype(np.float32)[:, None]
    d[461:, :] = 0.8
    d[100:150, :] = 8.0   # 超出锚定窗(>4m)
    d[350:420, 400:440] = 0.6  # 2m 平面上的近距突跳(相对梯度大 → 剔除)
    return d


def test_mask():
    cfg = FusionConfig()
    d = make_scene()
    m = d455_high_confidence_mask(d, cfg)
    assert m[310:340, 100:300].mean() > 0.9, "平坦地面应保留"
    # 断崖(0.6m 块):过渡带剔除,但块内部是平坦真实深度,按设计保留
    assert not m[346:354, 405:435].any(), "断崖上沿过渡带应被梯度+腐蚀剔除"
    assert not m[416:424, 405:435].any(), "断崖下沿过渡带应被剔除"
    assert m[356:414, 405:435].mean() > 0.9, "断崖内部平坦区是高置信深度,应保留"
    assert not m[:300].any(), "无效区(0)不应进掩码"
    assert not m[100:150].any(), "超窗(8m)不应进掩码"
    print("[PASS] mask: 平面保留/过渡带剔除/块内平坦保留/无效与超窗排除")


def test_fit_log():
    rng = np.random.default_rng(3)
    d455 = make_scene()
    a_true, scale_true = 1.1, 0.9
    yolo = (scale_true * np.power(np.clip(d455, 1e-4, None), a_true)).astype(np.float32)
    valid = d455 > 0
    yolo[valid] *= (1 + rng.normal(0, 0.01, int(valid.sum()))).astype(np.float32)
    out = valid & (rng.random(d455.shape) < 0.05)
    yolo[out] *= 0.5  # 镜面斑离群
    cfg = FusionConfig(fit_mode="log")
    mask = d455_high_confidence_mask(d455, cfg)
    p = fit_anchor_params(yolo, d455, mask, cfg)
    assert p is not None and p.mode == "log"
    # 拟合方向是 y→z:y = s·z^a 的逆映射为 z = s^(-1/a)·y^(1/a)
    a_inv = 1.0 / a_true
    s_inv = scale_true ** (-1.0 / a_true)
    assert abs(p.p1 - a_inv) < 0.03, f"指数偏差过大: {p} (期望 {a_inv:.3f})"
    assert abs(np.exp(p.p2) - s_inv) / s_inv < 0.03, f"尺度偏差过大: {p} (期望 {s_inv:.3f})"
    assert p.inlier_ratio > 0.8, f"内点率过低: {p.inlier_ratio:.2f}"
    # 校准应把 yolo 拉回真值:y=2.0 对应的真实深度
    z_ref_true = (2.0 / scale_true) ** (1.0 / a_true)
    z2 = p.apply(np.array([[2.0]], np.float32))[0, 0]
    assert abs(z2 - z_ref_true) / z_ref_true < 0.03, (z2, z_ref_true)
    print(f"[PASS] fit(log): a={p.p1:.3f}(期望{a_inv:.3f}) exp(b)={np.exp(p.p2):.3f}(期望{s_inv:.3f}) 内点率={p.inlier_ratio:.0%}")


def test_fit_linear():
    rng = np.random.default_rng(4)
    d455 = make_scene()
    S_true, t_true = 1.15, -0.2
    yolo = (S_true * d455 + t_true).astype(np.float32)
    valid = d455 > 0
    yolo[valid] += rng.normal(0, 0.01, int(valid.sum())).astype(np.float32)
    out = valid & (rng.random(d455.shape) < 0.05)
    yolo[out] *= 0.5
    cfg = FusionConfig(fit_mode="linear")
    mask = d455_high_confidence_mask(d455, cfg)
    p = fit_anchor_params(yolo, d455, mask, cfg)
    assert p is not None and p.mode == "linear"
    # 拟合方向是 y→z:y = S·z + t 的逆映射为 z = y/S - t/S
    assert abs(p.p1 - 1.0 / S_true) < 0.04, f"S 偏差过大: {p} (期望 {1/S_true:.3f})"
    assert abs(p.p2 - (-t_true / S_true)) < 0.04, f"t 偏差过大: {p} (期望 {-t_true/S_true:.3f})"
    assert p.inlier_ratio > 0.8
    print(f"[PASS] fit(linear): S={p.p1:.3f}(期望{1/S_true:.3f}) t={p.p2:+.3f}(期望{-t_true/S_true:+.3f}) 内点率={p.inlier_ratio:.0%}")


def test_fit_insufficient():
    d455 = make_scene()
    cfg = FusionConfig(min_anchor_pixels=10_000_000)  # 故意设不可能的阈值
    mask = d455_high_confidence_mask(d455, FusionConfig())
    p = fit_anchor_params(np.clip(d455, 1e-3, None), d455, mask, cfg)
    assert p is None
    print("[PASS] fit: 像素不足返回 None(沿用上一帧)")


def test_smoother():
    cfg = FusionConfig(ema_alpha=0.2)
    s = AnchorSmoother(cfg)
    p_a = AnchorParams("log", 1.0, 0.0, 0.9, 5000)
    got, upd = s.update(p_a)
    assert upd and got is p_a, "首帧应直接采纳"
    got, upd = s.update(AnchorParams("log", 1.0, 0.0, 0.9, 5000))
    assert upd and abs(got.p1 - 1.0) < 1e-9

    p_jump = AnchorParams("log", 1.6, 0.0, 0.9, 5000)  # 参考点 2m→3.03m,+51%
    got, upd = s.update(p_jump)
    assert not upd and abs(got.p1 - 1.0) < 1e-9, "单帧跳变应被守卫拒绝"
    got, upd = s.update(p_jump)
    assert not upd, "第 2 次跳变仍应拒绝"
    got, upd = s.update(p_jump)
    assert upd and 1.0 < got.p1 < 1.6, "连续 3 次跳变应接受(EMA 快速跟上)"
    cur_before = s.cur
    got, upd = s.update(None)
    assert (not upd) and got is cur_before, "拟合失败应沿用上次参数"
    print("[PASS] smoother: 首帧采纳/单帧跳变拒绝/连续跳变接受/失效沿用")


def run():
    test_mask()
    test_fit_log()
    test_fit_linear()
    test_fit_insufficient()
    test_smoother()


if __name__ == "__main__":
    run()
