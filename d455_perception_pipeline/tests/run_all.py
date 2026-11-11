"""闸门统一测试入口:B1 对齐 → B2 融合 → B3 端到端(真模型) → B4 实时自测。"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_alignment  # noqa: E402
import test_e2e  # noqa: E402
import test_fusion  # noqa: E402
import test_live  # noqa: E402

SUITE = [
    ("B1 alignment", test_alignment),
    ("B2 fusion", test_fusion),
    ("B3 e2e(model)", test_e2e),
    ("B4 live selftest(model)", test_live),
]


def main() -> int:
    results = []
    t_all = time.perf_counter()
    for name, mod in SUITE:
        t0 = time.perf_counter()
        try:
            mod.run()
            results.append((name, "PASS", time.perf_counter() - t0, ""))
        except Exception:
            results.append((name, "FAIL", time.perf_counter() - t0, traceback.format_exc()))
    print("\n===== GATE SUMMARY =====")
    failed = 0
    for name, status, dt, tb in results:
        print(f"[{status}] {name} ({dt:.1f}s)")
        if status == "FAIL":
            failed += 1
            print(tb)
    print(f"total {time.perf_counter() - t_all:.1f}s, {len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
