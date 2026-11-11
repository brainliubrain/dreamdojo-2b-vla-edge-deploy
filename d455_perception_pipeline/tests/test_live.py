"""B4 实时管线自测(无相机):MockSource 连续 6 帧全管线,验证可连续运行。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import MockSource  # noqa: E402
from d455_fusion import DepthFusionPipeline, FusionConfig  # noqa: E402
from d455_fusion.viz import fusion_panel  # noqa: E402

OUT = ROOT / "outputs"


def run(out_dir: Path = OUT, n_frames: int = 6):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = FusionConfig()
    src = MockSource()
    pipe = DepthFusionPipeline(cfg)

    last = None
    model_ms = []
    align_ms = []
    for _ in range(n_frames):
        rgb, d455, _ = src.read()
        res = pipe.process(rgb, d455)
        assert res.fused.shape == d455.shape
        assert res.fused.min() > 0 and res.fused.max() < 1e4
        model_ms.append(res.stats["model_ms"])
        align_ms.append(res.stats["align_ms"])
        last = (rgb, d455, res)

    rgb, d455, res = last
    panel = fusion_panel(
        rgb, d455, res.yolo_raw, res.yolo_corrected, res.fused,
        res.params, {**res.stats, "mask": res.mask}, cfg,
    )
    cv2.imwrite(str(out_dir / "live_selftest_viz.png"), panel)
    summary = {
        "frames": n_frames,
        "model_ms_mean": sum(model_ms) / len(model_ms),
        "align_ms_mean": sum(align_ms) / len(align_ms),
        "fps_est": 1000.0 / ((sum(model_ms) + sum(align_ms)) / len(model_ms)),
        "params_final": {"p1": res.params.p1, "p2": res.params.p2,
                         "implied_at_ref": res.params.implied_depth_at()},
        "inlier_ratio": res.stats["inlier_ratio"],
    }
    (out_dir / "live_selftest_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert summary["model_ms_mean"] < 500, "单帧推理异常缓慢"
    print(f"[PASS] live selftest: {n_frames} 帧, model={summary['model_ms_mean']:.0f}ms, "
          f"align={summary['align_ms_mean']:.1f}ms, 估算 {summary['fps_est']:.1f} FPS")
    return summary


if __name__ == "__main__":
    run()
