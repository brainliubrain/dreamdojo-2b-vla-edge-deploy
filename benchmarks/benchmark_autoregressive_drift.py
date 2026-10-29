import os
import sys
import time
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from dreamdojo_standalone_engine import DreamDojoDiT2B, LatentDepthHead, load_weights_into_model
from latent_stabilizer import LatentStabilizer

# High performance settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["MUJOCO_GL"] = "osmesa"

def run_autoregressive_benchmark(steps=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  DREAMDOJO 2B AUTOREGRESSIVE DRIFT & STABILITY EVALUATION")
    print(f"  Steps: {steps} | Hardware: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Load Model in BF16
    print("\n[1/4] Initializing DiT 2B Model & Components...")
    model = DreamDojoDiT2B(num_blocks=28, model_channels=2048, action_dim=384, num_action_chunk=4, use_int8=False)
    model.to(device)
    depth_head = LatentDepthHead(in_channels=16).to(device)

    transformer_dir = "/mnt/d/DreamDojo_2B/transformer"
    load_weights_into_model(model, transformer_dir, device=device)
    model.eval()

    stabilizer = LatentStabilizer(
        vae_config_path="/mnt/d/DreamDojo_2B/vae/config.json",
        device=device,
        dtype=torch.bfloat16,
        proj_strength=0.35,
        clamp_k=3.5
    ).to(device)

    output_dir = "/mnt/d/DreamDojo_2B/outputs/drift_eval"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Setup Canonical Initial State and Inputs
    print("\n[2/4] Initializing Canonical Physical State...")
    B, C, T, H, W = 1, 18, 1, 30, 30
    
    # Deterministic initial seed for exact 1:1 fairness across all 4 groups
    torch.manual_seed(42)
    init_latent_state = torch.randn(B, 16, T, H, W, dtype=torch.bfloat16, device=device)
    init_cond = torch.randn(B, 2, T, H, W, dtype=torch.bfloat16, device=device)
    base_input = torch.cat([init_latent_state, init_cond], dim=1)
    
    text_emb = torch.randn(B, 77, 1024, dtype=torch.bfloat16, device=device)
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=device)

    # Simulated robot trajectory: end-effector oscillating gently in workspace
    actions_seq = []
    for s in range(steps):
        act = torch.randn(B, 4, 384, dtype=torch.bfloat16, device=device) * 0.1
        # Set realistic moving end-effector position
        act[0, 0, 0] = 0.1 * math.sin(s * 0.2)
        act[0, 0, 1] = 0.1 * math.cos(s * 0.2)
        act[0, 0, 2] = 0.45
        actions_seq.append(act)

    # Save initial canonical depth
    with torch.inference_mode():
        init_depth = depth_head(init_latent_state)
        init_d_norm = ((init_depth - init_depth.min()) / (init_depth.max() - init_depth.min() + 1e-6) * 255.0).clamp(0, 255).byte().cpu().numpy()
        Image.fromarray(init_d_norm, mode="L").save(os.path.join(output_dir, "depth_step00_canonical.png"))

    groups = [
        {"name": "Group_A_Baseline", "s2": False, "s3": False, "desc": "Baseline (No Stabilizer)"},
        {"name": "Group_B_Scheme2_Only", "s2": True, "s3": False, "desc": "Scheme 2 Only (Background Freeze)"},
        {"name": "Group_C_Scheme3_Only", "s2": False, "s3": True, "desc": "Scheme 3 Only (Manifold Clamping)"},
        {"name": "Group_D_Full_Stabilized", "s2": True, "s3": True, "desc": "Full Stabilized (Scheme 2 + Scheme 3)"},
    ]

    results_summary = {}

    # 3. Execute 4 Comparison Groups
    print(f"\n[3/4] Running 4 Comparison Experiments Across {steps} Closed-Loop Steps...")
    
    for grp in groups:
        g_name = grp["name"]
        g_desc = grp["desc"]
        s2 = grp["s2"]
        s3 = grp["s3"]
        print(f"\n---> Testing {g_name} ({g_desc})...")

        current_input = base_input.clone()
        canonical_z = init_latent_state.clone()

        telemetry = []
        saved_depth_imgs = {}
        t_start_group = time.perf_counter()
        latencies = []

        with torch.inference_mode():
            for s in range(1, steps + 1):
                t0 = time.perf_counter()
                
                # Forward pass
                delta = model(current_input, timestep, actions_seq[s-1], text_emb)
                raw_next = current_input[:, :16, :, :, :] + delta

                # Apply stabilizer
                stabilized_next = stabilizer(
                    raw_next,
                    canonical_z,
                    enable_scheme2=s2,
                    enable_scheme3=s3
                )
                current_input[:, :16, :, :, :].copy_(stabilized_next)

                torch.cuda.synchronize()
                step_ms = (time.perf_counter() - t0) * 1000
                latencies.append(step_ms)

                # Collect metrics
                z_curr = current_input[:, :16, :, :, :]
                mean_drift = torch.abs(z_curr.mean() - canonical_z.mean()).item()
                std_curr = z_curr.std().item()
                std_ratio = std_curr / canonical_z.std().item()
                max_mag = z_curr.abs().max().item()
                corner_drift = torch.abs(z_curr[:, :, :, :5, :5] - canonical_z[:, :, :, :5, :5]).mean().item()
                center_drift = torch.abs(z_curr[:, :, :, 10:20, 10:20] - canonical_z[:, :, :, 10:20, 10:20]).mean().item()

                telemetry.append({
                    "step": s,
                    "mean_drift": round(mean_drift, 4),
                    "std_ratio": round(std_ratio, 4),
                    "max_magnitude": round(max_mag, 4),
                    "corner_drift": round(corner_drift, 4),
                    "center_drift": round(center_drift, 4),
                    "latency_ms": round(step_ms, 2)
                })

                # Checkpoint depth map
                if s in [10, 20, steps]:
                    d_map = depth_head(z_curr)
                    d_norm = ((d_map - d_map.min()) / (d_map.max() - d_map.min() + 1e-6) * 255.0).clamp(0, 255).byte().cpu().numpy()
                    img_path = os.path.join(output_dir, f"{g_name}_depth_step{s:02d}.png")
                    img = Image.fromarray(d_norm, mode="L")
                    img.save(img_path)
                    saved_depth_imgs[f"step_{s}"] = img_path

        total_time_ms = (time.perf_counter() - t_start_group) * 1000
        avg_step_ms = sum(latencies) / len(latencies)
        fps = 1000.0 / avg_step_ms
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)

        last_telem = telemetry[-1]
        print(f"     [FINISHED] Avg Latency: {avg_step_ms:.2f} ms ({fps:.2f} FPS)")
        print(f"     [TELEMETRY Step {steps}] Mean Drift: {last_telem['mean_drift']:.4f} | Std Ratio: {last_telem['std_ratio']:.4f} | Max Mag: {last_telem['max_magnitude']:.2f}")
        print(f"     [TELEMETRY Step {steps}] Corner Drift: {last_telem['corner_drift']:.4f} (Background stability)")

        results_summary[g_name] = {
            "description": g_desc,
            "scheme2_enabled": s2,
            "scheme3_enabled": s3,
            "avg_latency_ms": round(avg_step_ms, 2),
            "effective_fps": round(fps, 2),
            "step30_mean_drift": last_telem["mean_drift"],
            "step30_std_ratio": last_telem["std_ratio"],
            "step30_max_magnitude": last_telem["max_magnitude"],
            "step30_corner_drift": last_telem["corner_drift"],
            "step30_center_drift": last_telem["center_drift"],
            "saved_depth_images": saved_depth_imgs,
            "telemetry_history": telemetry
        }

    # 4. Create Side-by-Side Comparison Collage
    print("\n[4/4] Generating Side-by-Side Visual Comparison Collage...")
    tile_size = 480
    collage = Image.new("RGB", (tile_size * 4, tile_size + 40), color=(30, 30, 30))
    draw = ImageDraw.Draw(collage)

    for idx, grp in enumerate(groups):
        g_name = grp["name"]
        img_path = results_summary[g_name]["saved_depth_images"][f"step_{steps}"]
        tile = Image.open(img_path).convert("RGB")
        x_offset = idx * tile_size
        collage.paste(tile, (x_offset, 40))

        # Title banner
        title_text = f"{grp['desc']}\nStdRatio: {results_summary[g_name]['step30_std_ratio']:.2f} | Corner: {results_summary[g_name]['step30_corner_drift']:.3f}"
        draw.text((x_offset + 10, 8), title_text, fill=(255, 255, 255))

    collage_path = os.path.join(output_dir, "autoregressive_step30_comparison.png")
    collage.save(collage_path)
    print(f"[ARTIFACT] Comparison collage saved to: {collage_path}")

    # Save comprehensive report
    report_file = os.path.join(output_dir, "stabilization_drift_report.json")
    with open(report_file, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[ARTIFACT] JSON report saved to: {report_file}")

    print("\n" + "=" * 80)
    print("  AUTOREGRESSIVE STABILIZATION EVALUATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    run_autoregressive_benchmark(steps=30)
