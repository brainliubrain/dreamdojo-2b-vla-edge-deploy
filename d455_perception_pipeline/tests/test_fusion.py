"""B2 三区融合模块单测:权重剖面 / 近远场归属 / 补洞 / 边界连续性。纯 numpy。"""
import numpy as np

from d455_fusion.config import FusionConfig
from d455_fusion.fusion import fuse, zone_weights


def test_weight_profile():
    cfg = FusionConfig(density_gate=False, fusion_edge_feather_px=0)  # 纯剖面,防重影特性另测
    d = np.array([[1.0, 4.0, 5.5, 7.0, 8.0, 0.0, np.nan]], np.float32)
    w = zone_weights(d, cfg)
    assert w[0, 0] == 1.0, "近场应全信 D455"
    assert abs(w[0, 1] - 1.0) < 1e-6, "4.0m 处权重应为 1(与近场连续)"
    assert abs(w[0, 2] - 0.5) < 1e-6, "5.5m 为过渡区中点,权重 0.5"
    assert w[0, 3] == 0.0, "7.0m 处权重应为 0(与远场连续)"
    assert w[0, 4] == 0.0, "远场应为 0"
    assert w[0, 5] == 0.0 and w[0, 6] == 0.0, "无效像素权重 0"
    print("[PASS] 权重剖面: 近场1 / 中点0.5 / 边界连续 / 无效0")


def test_zone_ownership():
    cfg = FusionConfig(density_gate=False, fusion_edge_feather_px=0)
    d455 = np.array([[1.0, 5.5, 8.0]], np.float32)
    yolo = np.array([[9.9, 6.6, 7.7]], np.float32)
    fused, w = fuse(d455, yolo, cfg)
    assert abs(fused[0, 0] - 1.0) < 1e-6, "近场输出=D455"
    assert abs(fused[0, 1] - 0.5 * (5.5 + 6.6)) < 1e-5, "过渡区=对半混合"
    assert abs(fused[0, 2] - 7.7) < 1e-6, "远场输出=校准后YOLO"


def test_near_hole_fill():
    cfg = FusionConfig(fill_near_holes=True)
    d455 = np.array([[0.0, 2.0]], np.float32)  # 近场有洞
    yolo = np.array([[1.9, 2.1]], np.float32)
    fused, w = fuse(d455, yolo, cfg)
    assert abs(fused[0, 0] - 1.9) < 1e-6, "近场洞应由校准后 YOLO 填补"
    cfg_off = FusionConfig(fill_near_holes=False)
    fused2, _ = fuse(d455, yolo, cfg_off)
    assert fused2[0, 0] == 0.0, "严格模式下洞保持 0(安全语义)"


def test_boundary_continuity():
    """yolo_corrected 与 d455 完全一致时,融合输出应处处等于该值(无接缝)。"""
    cfg = FusionConfig()
    v = np.linspace(0.5, 10.0, 200, dtype=np.float32)
    fused, _ = fuse(v[None, :], v[None, :], cfg)
    assert np.allclose(fused, v[None, :], atol=1e-5), "边界处不应出现台阶"
    print("[PASS] 边界连续性: 4m/7m 处无台阶")


def test_density_gate_and_feather():
    """防重影:孤立噪声岛让位 YOLO,大块连续区不受影响,边界羽化平滑。"""
    cfg = FusionConfig()
    d = np.zeros((60, 80), np.float32)
    d[10:14, 10:14] = 1.0    # 孤立小岛(4x4,邻域有效率极低)
    d[30:58, 30:70] = 1.0    # 大块连续区
    yolo = np.full_like(d, 1.8)
    w = zone_weights(d, cfg)
    fused, _ = fuse(d, yolo, cfg)
    assert w[11, 11] < 0.2, f"孤立岛应被密度门控压制, w={w[11, 11]:.2f}"
    assert abs(fused[11, 11] - 1.8) < 0.15, "孤立岛输出应接近 YOLO"
    assert w[44, 50] > 0.9, f"大块连续区中心应保持满权重, w={w[44, 50]:.2f}"
    assert abs(fused[44, 50] - 1.0) < 1e-5, "大块区中心输出=D455"
    # 边界羽化:大块区边缘权重低于中心
    assert w[31, 50] < w[44, 50], "距失效区近的边缘权重应衰减(羽化)"
    # 关闭防重影后行为与旧版一致
    cfg_off = FusionConfig(
        density_gate=False,
        fusion_edge_feather_px=0,
        speckle_filter=False,
        depth_jump_feather=False,
    )
    w_off = zone_weights(d, cfg_off)
    assert w_off[11, 11] == 1.0 and w_off[31, 50] == 1.0
    print("[PASS] 防重影: 孤立岛压制 / 大块保留 / 边界羽化 / 可关闭回退")


def test_near_blind_and_depth_jump():
    """测试近盲区平滑过渡与深度跳跃阶跃羽化。"""
    cfg = FusionConfig()
    # 1. 近盲区平滑测试
    d_near = np.array([[0.2, 0.32, 0.41, 0.50, 0.80]], np.float32)
    w_near = zone_weights(d_near, cfg)
    assert w_near[0, 0] == 0.0, "极近场(<0.32m)彻底让位 YOLO"
    assert w_near[0, 1] == 0.0, "0.32m 起点处权重为 0"
    assert 0.3 < w_near[0, 2] < 0.7, "0.41m 过渡中点处权重平滑在 0.5 附近"
    assert abs(w_near[0, 3] - 1.0) < 1e-4, "0.50m 终点处恢复满权重"
    assert w_near[0, 4] == 1.0, "正常近场(0.8m)全信 D455"

    # 2. 深度跳跃阶跃羽化测试(前景/背景交界处)
    d2 = np.full((40, 40), 2.5, dtype=np.float32)  # 背景 2.5m
    d2[15:25, 15:25] = 0.8                         # 前景手/物体 0.8m
    w2 = zone_weights(d2, cfg)
    # 前景中心保持满权重
    assert w2[20, 20] > 0.8, f"前景中心应保持高权重, w={w2[20, 20]:.2f}"
    # 前景与背景交界处(15, 20)因深度突变产生阶跃羽化，让位给 YOLO
    assert w2[15, 20] < 0.5, f"阶跃边缘处权重应衰减, w={w2[15, 20]:.2f}"
    print("[PASS] 近盲区平滑与深度阶跃边缘羽化")


def run():
    test_weight_profile()
    test_zone_ownership()
    test_near_hole_fill()
    test_boundary_continuity()
    test_density_gate_and_feather()
    test_near_blind_and_depth_jump()


if __name__ == "__main__":
    run()
