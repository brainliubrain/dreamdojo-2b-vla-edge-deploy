"""离线批处理:--bag 回放 或 --images 目录对(rgb_*.png + depth_*.npy 米制)。

输出到 outputs/offline/:每帧融合 npy + 面板 png(可抽样)+ 汇总 json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import RealSenseSource  # noqa: E402
from d455_fusion import DepthFusionPipeline, FusionConfig  # noqa: E402
from d455_fusion.viz import fusion_panel  # noqa: E402


def iter_bag(bag: str, step: int):
    src = RealSenseSource(bag=bag)
    try:
        i = 0
        while True:
            try:
                rgb, d455, _ = src.read()
            except RuntimeError:
                break  # 回放结束
            if i % step == 0:
                yield i, rgb, d455
            i += 1
    finally:
        src.close()


def iter_images(folder: Path, step: int):
    rgb_files = sorted(folder.glob("rgb_*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"{folder} 下无 rgb_*.png")
    for i, f in enumerate(rgb_files):
        if i % step:
            continue
        dpath = f.with_name(f.name.replace("rgb_", "depth_")).with_suffix(".npy")
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        d455 = np.load(dpath).astype(np.float32)
        yield i, rgb, d455


def main():
    ap = argparse.ArgumentParser(description="离线融合批处理")
    ap.add_argument("--bag", default=None)
    ap.add_argument("--images", default=None, help="含 rgb_*.png + depth_*.npy 的目录")
    ap.add_argument("--weights", default=None, help="默认取 FusionConfig().weights")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--step", type=int, default=1, help="每 N 帧处理一次")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "offline"))
    args = ap.parse_args()
    if not args.bag and not args.images:
        ap.error("需要 --bag 或 --images")

    cfg = FusionConfig(weights=args.weights or FusionConfig().weights, imgsz=args.imgsz)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = DepthFusionPipeline(cfg)

    it = iter_bag(args.bag, args.step) if args.bag else iter_images(Path(args.images), args.step)
    rows = []
    for i, rgb, d455 in it:
        res = pipe.process(rgb, d455)
        np.save(out_dir / f"frame_{i:06d}_fused.npy", res.fused)
        if i % max(args.step, 10) == 0:  # 面板抽样存
            panel = fusion_panel(
                rgb, d455, res.yolo_raw, res.yolo_corrected, res.fused,
                res.params, {**res.stats, "mask": res.mask}, cfg)
            cv2.imwrite(str(out_dir / f"frame_{i:06d}_panel.png"), panel)
        rows.append({"frame": i, "anchor_px": res.stats["anchor_pixels"],
                     "inlier": res.stats["inlier_ratio"],
                     "implied_at_ref": res.stats["implied_at_ref"]})
        print(f"\rframe {i}", end="")
    (out_dir / "summary.json").write_text(
        json.dumps({"config": str(cfg), "frames": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\ndone → {out_dir}")


if __name__ == "__main__":
    main()
