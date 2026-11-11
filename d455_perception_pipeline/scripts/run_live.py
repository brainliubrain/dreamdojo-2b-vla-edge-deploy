"""实时运行:D455 实况(或 --selftest 合成自测)→ 融合深度 → 可视化窗口。

用法:
  python scripts/run_live.py                # D455 实况(需插相机,USB3)
  python scripts/run_live.py --selftest     # 无相机自测(MockSource)
  python scripts/run_live.py --bag x.bag    # 回放录制的 bag
按键:q 退出 / s 保存当前面板到 outputs/live/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from capture import MockSource, RealSenseSource  # noqa: E402
from d455_fusion import DepthFusionPipeline, FusionConfig  # noqa: E402
from d455_fusion.viz import fusion_panel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="D455 × YOLO26-Depth 融合实时管线")
    ap.add_argument("--selftest", action="store_true", help="无相机自测(MockSource)")
    ap.add_argument("--bag", default=None, help="回放 .bag 文件")
    ap.add_argument("--weights", default=None, help="默认取 FusionConfig().weights")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fit-mode", choices=["log", "linear"], default=None,
                    help="默认取 FusionConfig().fit_mode (linear)")
    ap.add_argument("--near-end", type=float, default=4.0)
    ap.add_argument("--far-start", type=float, default=7.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0=不限(自测模式默认 8)")
    ap.add_argument("--depth-width", type=int, default=None,
                    help="深度分辨率宽(None=随 color)。官方:降为 424 可使 minZ(近端盲区)线性减半")
    ap.add_argument("--disparity-shift", type=int, default=0,
                    help="高级模式 Disparity Shift(0=关)。增大可进一步降 minZ,代价远端量程收窄")
    ap.add_argument("--no-spatial", action="store_true", help="关闭视差域边缘保持滤波")
    ap.add_argument("--save-dir", default=str(ROOT / "outputs" / "live"))
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    _base = FusionConfig()
    cfg = FusionConfig(
        weights=args.weights or _base.weights, imgsz=args.imgsz,
        fit_mode=args.fit_mode or _base.fit_mode,
        near_end=args.near_end, far_start=args.far_start,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        source = MockSource()
        args.max_frames = args.max_frames or 8
        args.no_show = True
    elif args.bag:
        source = RealSenseSource(bag=args.bag)
    else:
        source = RealSenseSource(
            args.width, args.height, args.fps,
            depth_width=args.depth_width,
            disparity_shift=args.disparity_shift,
            spatial=not args.no_spatial,
        )

    print(f"[pipeline] weights={cfg.weights} imgsz={cfg.imgsz} mode={cfg.fit_mode} "
          f"zones=0-{cfg.near_end}-{cfg.far_start}m")
    pipe = DepthFusionPipeline(cfg)

    n, last_stats = 0, None
    t_start = time.perf_counter()
    try:
        while True:
            rgb, d455, _ = source.read()
            res = pipe.process(rgb, d455)
            last_stats = {**res.stats, "mask": res.mask}
            panel = fusion_panel(
                rgb, d455, res.yolo_raw, res.yolo_corrected, res.fused,
                res.params, last_stats, cfg,
            )
            n += 1
            if args.no_show:
                fps = n / (time.perf_counter() - t_start)
                print(f"\rframe {n}  {fps:.1f} fps  "
                      f"model={res.stats['model_ms']:.0f}ms  "
                      f"inlier={res.stats['inlier_ratio']:.0%}  "
                      f"implied@2m={res.stats['implied_at_ref']:.2f}m   ", end="")
            else:
                cv2.imshow("D455 x YOLO26-Depth fusion  [q]uit [s]ave", panel)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    p = save_dir / f"panel_{int(time.time())}.png"
                    cv2.imwrite(str(p), panel)
                    print(f"saved {p}")
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()

    if last_stats is not None:
        summary = {k: v for k, v in last_stats.items() if k != "mask"}
        summary["frames"] = n
        (save_dir / "last_frame_stats.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.no_show:
        cv2.imwrite(str(save_dir / "last_panel.png"), panel)
        print(f"\nsaved {save_dir/'last_panel.png'}")


if __name__ == "__main__":
    main()
