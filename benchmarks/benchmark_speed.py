import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamdojo_standalone_engine import DreamDojoDiT2B, LatentDepthHead, load_weights_into_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[BENCHMARK] Device: {device} ({torch.cuda.get_device_name(0)})")

# Test 1: Native BF16 (use_int8=False)
print("\n--- TEST: Native BF16 (Zero Dequantization Overhead) ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Initialize with use_int8=False (Pure BF16)
model_bf16 = DreamDojoDiT2B(num_blocks=28, model_channels=2048, action_dim=384, num_action_chunk=4, use_int8=False)
model_bf16.to(device)

vram_alloc_init = torch.cuda.memory_allocated() / (1024**3)
print(f"Allocated memory for uninitialized BF16 model: {vram_alloc_init:.2f} GB")

# Load weights
transformer_dir = "/mnt/d/DreamDojo_2B/transformer"
t0 = time.time()
load_weights_into_model(model_bf16, transformer_dir, device=device)
load_time = time.time() - t0
vram_post_load = torch.cuda.memory_allocated() / (1024**3)
print(f"Post-load allocated VRAM: {vram_post_load:.2f} GB (Loaded in {load_time:.2f}s)")

# Prepare input tensors
B, C, T, H, W = 1, 18, 4, 30, 30
model_input = torch.randn(B, C, T, H, W, dtype=torch.bfloat16, device=device)
robot_actions = torch.randn(B, 4, 384, dtype=torch.bfloat16, device=device)
text_emb = torch.randn(B, 77, 1024, dtype=torch.bfloat16, device=device)
timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=device)

# Warmup
with torch.inference_mode():
    _ = model_bf16(model_input, timestep, robot_actions, text_emb)
torch.cuda.synchronize()

# Benchmark 5 runs
latencies = []
torch.cuda.reset_peak_memory_stats()
for i in range(5):
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.inference_mode():
        out = model_bf16(model_input, timestep, robot_actions, text_emb)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t_start) * 1000)

peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
avg_lat = sum(latencies) / len(latencies)
fps = 1000.0 / avg_lat

print(f"\n[RESULTS - Native BF16]")
print(f"  Average Latency: {avg_lat:.2f} ms")
print(f"  Effective Frame Rate: {fps:.2f} FPS")
print(f"  Peak VRAM: {peak_vram:.2f} GB / 6.00 GB")
print(f"  Headroom: {6.00 - peak_vram:.2f} GB")
