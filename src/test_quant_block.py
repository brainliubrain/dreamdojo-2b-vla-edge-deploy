import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from einops import rearrange, repeat

# -------------------------------------------------------------
# Precision & Norm Layer
# -------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# -------------------------------------------------------------
# Int8 Linear Layer for Selective Quantization
# -------------------------------------------------------------
class Int8WeightLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scales", torch.ones((out_features, 1), dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

    def set_weight(self, weight_bf16):
        # Calculate per-channel scale
        max_abs = weight_bf16.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
        scale = max_abs / 127.0
        q_weight = (weight_bf16 / scale).round().clamp(-128, 127).to(torch.int8)
        self.weight_int8.copy_(q_weight)
        self.scales.copy_(scale.to(torch.bfloat16))

    def forward(self, x):
        # On-the-fly dequantization in GPU registers/cache
        w_bf16 = self.weight_int8.to(x.dtype) * self.scales
        return F.linear(x, w_bf16, self.bias)

# -------------------------------------------------------------
# RoPE
# -------------------------------------------------------------
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, rope_emb):
    # x: [B, S, H, D], rope_emb: [S, 1, 1, D]
    cos = torch.cos(rope_emb).type_as(x)
    sin = torch.sin(rope_emb).type_as(x)
    return (x * cos) + (rotate_half(x) * sin)

# -------------------------------------------------------------
# Self & Cross Attention
# -------------------------------------------------------------
class DiTAttention(nn.Module):
    def __init__(self, q_dim, k_dim, n_heads=16, head_dim=128, is_self=True, use_int8=True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.is_self = is_self
        inner_dim = n_heads * head_dim

        LinearClass = Int8WeightLinear if use_int8 else nn.Linear
        self.q_proj = LinearClass(q_dim, inner_dim, bias=False)
        self.k_proj = LinearClass(k_dim, inner_dim, bias=False)
        self.v_proj = LinearClass(k_dim, inner_dim, bias=False)
        self.output_proj = LinearClass(inner_dim, q_dim, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, context=None, rope_emb=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        
        ctx = x if context is None else context
        _, S_k, _ = ctx.shape
        k = self.k_proj(ctx).view(B, S_k, self.n_heads, self.head_dim)
        v = self.v_proj(ctx).view(B, S_k, self.n_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.is_self and rope_emb is not None:
            q = apply_rotary_pos_emb(q, rope_emb)
            k = apply_rotary_pos_emb(k, rope_emb)

        # PyTorch FlashAttention / SDPA
        q = q.transpose(1, 2) # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.output_proj(out)

# -------------------------------------------------------------
# DiT Transformer Block
# -------------------------------------------------------------
class DiTBlock(nn.Module):
    def __init__(self, dim=2048, context_dim=1024, n_heads=16, head_dim=128, adaln_lora_dim=256, use_int8=True):
        super().__init__()
        self.dim = dim
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = DiTAttention(dim, dim, n_heads, head_dim, is_self=True, use_int8=use_int8)

        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = DiTAttention(dim, context_dim, n_heads, head_dim, is_self=False, use_int8=use_int8)

        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        LinearClass = Int8WeightLinear if use_int8 else nn.Linear
        self.mlp = nn.Module()
        self.mlp.layer1 = LinearClass(dim, dim * 4, bias=False)
        self.mlp.layer2 = LinearClass(dim * 4, dim, bias=False)
        self.mlp.act = nn.GELU()

        # adaLN modulation layers (Keep High Precision FP16/BF16!)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * dim, bias=False)
        )
        self.adaln_modulation_cross_attn = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * dim, bias=False)
        )
        self.adaln_modulation_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, adaln_lora_dim, bias=False),
            nn.Linear(adaln_lora_dim, 3 * dim, bias=False)
        )

    def forward(self, x, emb_t_d, adaln_lora, crossattn_emb, rope_emb=None):
        # 1. Self Attention with adaLN
        shift_s, scale_s, gate_s = (self.adaln_modulation_self_attn(emb_t_d) + adaln_lora).chunk(3, dim=-1)
        norm_x = self.layer_norm_self_attn(x) * (1 + scale_s) + shift_s
        x = x + gate_s * self.self_attn(norm_x, rope_emb=rope_emb)

        # 2. Cross Attention with adaLN
        shift_c, scale_c, gate_c = (self.adaln_modulation_cross_attn(emb_t_d) + adaln_lora).chunk(3, dim=-1)
        norm_x = self.layer_norm_cross_attn(x) * (1 + scale_c) + shift_c
        x = x + gate_c * self.cross_attn(norm_x, context=crossattn_emb)

        # 3. MLP with adaLN
        shift_m, scale_m, gate_m = (self.adaln_modulation_mlp(emb_t_d) + adaln_lora).chunk(3, dim=-1)
        norm_x = self.layer_norm_mlp(x) * (1 + scale_m) + shift_m
        h = self.mlp.layer2(self.mlp.act(self.mlp.layer1(norm_x)))
        x = x + gate_m * h
        return x

print("DiT Block and Quantized Linear classes defined successfully!")
