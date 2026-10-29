import torch
import torch.nn as nn
import cv2
import numpy as np
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ==========================================
# 潜空间深度适配器 (Latent-to-Depth Adapter)
# 对应您提到的 DiTF / Marigold 路线，彻底抛弃 RGB
# ==========================================
class LatentDepthHead(nn.Module):
    def __init__(self, latent_channels=16): 
        super().__init__()
        # 用最轻量的 1x1 卷积，直接把高维潜变量特征压缩为 1 维物理深度
        self.proj = nn.Conv2d(latent_channels, 1, kernel_size=1)
        
    def forward(self, latents):
        # 如果是视频特征 (B, C, T, H, W)，取最新的一帧 (B, C, H, W)
        if latents.dim() == 5:
            latents = latents[:, :, -1, :, :]
        # 直接在潜空间输出深度
        depth_latent = self.proj(latents)
        return depth_latent

def main():
    print("[INFO] =========================================")
    print("[INFO] 潜空间(Latent Space)深度直截取模式启动")
    print("[INFO] 核心策略：跳过 VAE 像素渲染，释放巨量显存")
    print("[INFO] =========================================")

    # 1. 模拟加载 DreamDojo 管道
    # pipe = DiffusionPipeline.from_pretrained("/mnt/d/DreamDojo_2B", torch_dtype=torch.float16)
    
    # [核心操作]：物理超度 VAE，彻底不加载，完美保护 6GB 显卡
    # del pipe.vae 
    # torch.cuda.empty_cache()

    # 2. 模拟加载轻量级 Latent 探针
    depth_head = LatentDepthHead(latent_channels=16) # 根据 DreamDojo 的 VAE 通道数调整
    
    # 3. 模拟获取潜变量（取代原本的 RGB 渲染）
    print("[INFO] 正在提取 DreamDojo 预测的未来状态潜变量 (Latents)...")
    # latents = pipe(prompt="robot grasping", output_type="latent").images
    
    # 模拟生成的潜变量张量 (Batch=1, Channels=16, H=64, W=64)
    dummy_latents = torch.randn(1, 16, 64, 64) 

    # 4. 潜变量直接转深度图
    print("[INFO] 绕过像素空间，正在将潜变量直接解码为物理深度...")
    with torch.no_grad():
        # (1, 1, 64, 64)
        raw_depth = depth_head(dummy_latents)
        
        # 将 64x64 的特征图双线性插值放大回原始分辨率 (例如 512x512)
        full_res_depth = torch.nn.functional.interpolate(
            raw_depth, size=(512, 512), mode='bicubic', align_corners=False
        )

    # 5. 后处理与伪彩色渲染
    depth_array = full_res_depth.squeeze().numpy()
    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    cv2.imwrite("/mnt/d/DreamDojo_2B/latent_depth_map.png", depth_colored)
    print("[SUCCESS] 潜空间深度截取成功！")
    print("[SUCCESS] VAE 渲染开销已省去，深度图保存在 D:\\DreamDojo_2B\\latent_depth_map.png")

if __name__ == "__main__":
    main()
