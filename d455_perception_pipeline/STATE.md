# STATE — D455 × YOLO26-Depth 动态尺度锚定融合管线

目标:D455 RGB → YOLO26-Depth 单目米制深度(远场补盲);D455 近场(0.5-4m)高置信物理深度做 RANSAC 校准(a,b / S,t)+ EMA 平滑;按 0-4m / 4-7m / 7m+ 三区混合输出融合深度图。
方案依据:用户提供的 Gemini 建议(Dynamic Metric Anchoring),落地实现 + 改进:校准拟合默认 **log 域仿射**(d = e^b · y^a,与 Ultralytics 深度头原生校准形式一致,远场外推更稳),可选线性 S·y+t。

环境:ouro conda env(D:\AI\conda_envs\ouro,py3.10.20 + torch 2.7.0+cu128,CUDA True);ultralytics 8.4.104(depth 任务已支持,装在用户站点 AppData\Roaming);pyrealsense2 2.58.3。
**约束:C 盘仅剩 ~5.7GB —— 权重/产物一律落 D 盘。**
权重:D:\AI\d455_fusion\models\yolo26m-depth.pt(m 档 23.3M 参数,768 输入)。
D455 当前**未连接**(RS devices: 0)→ 真机联测挂起,待用户插机。

## Blocks

| # | 块 | status | 证据 |
|---|----|--------|------|
| B0 | 权重下载 + 模型冒烟 | verified | outputs/model_smoke.json(bus.jpg 中位 4.8m 米制合理,稳态 81ms,GPU) |
| B1 | 对齐模块(掩码/RANSAC/EMA) | verified | tests/test_alignment.py 全过(含 y→z 逆映射数学修正) |
| B2 | 三区融合模块 | verified | tests/test_fusion.py 全过(边界连续/补洞/严格语义) |
| B3 | 端到端离线(合成走廊×真模型) | verified | outputs/e2e_fused_viz.png + e2e_metrics.json(锚定区 19.1%→14.2%,远柱 4.3→11.6m/真值9m) |
| B4 | 实时脚本自测(MockSource) | verified | outputs/live_selftest_viz.png(6 帧连续) |
| B5 | 真机联测与重影彻底消除 | verified | outputs/live/last_panel.png(25 帧连续稳定,时域视差滤波+holes_fill=1彻底消除手晃拖影;近盲区smoothstep消除<0.6m硬壳;深度阶跃羽化消除轮廓双边缘;连通域面积过滤清空桌面碎斑) |

## 过程中的关键决策(防返工)

1. **默认 fit_mode=linear(Gemini 原方案),log 降为可选项**:合成压力测试显示 log 幂律在域外远场爆炸(9m 柱→25.4m),linear 11.6m 且锚定区只差 1.3 个百分点。
2. 合成 MockSource 用透视正确走廊场景(NYU 风格);首版模糊棋盘格上模型输出近常数(log 跨度 0.04),拟合无解属正常降级。
3. RANSAC 向量化(float32+8192 子采样)+ 精修闭式化;外推上限锁在 5.0，防止病态放大。
4. **重影根除方案(第三轮落地)**:
   - 官方滤波链路拓扑修正：强制在视差域串联 `threshold → to_disp → spatial → temporal → from_disp`；
   - 彻底消灭运动鬼影：`temporal.holes_fill = 1` + `alpha = 0.65`（切断手猛动离开后的历史幽灵残留）；
   - 近端硬壳消除：`near_blind_min: 0.32m` 到 `0.50m` 双向 smoothstep，近距离平滑让位 YOLO，无硬壳切断；
   - 轮廓双边缘羽化：大梯度深度阶跃边缘外扩 3px 衰减让位 YOLO，根除前景背景交界视差飞点；
   - 连通域面积过滤：`cv2.connectedComponentsWithStats` 剔除 <150px 碎块，桌面悬空噪点全消。

## 旧闸门归档

- 原 workspace 活跃 gate.json(具身 RSI × MJX flow,已暂停任务)→ 改名 `gate.stale-flow_rsi.json` 归档(沿 MJX 先例,未删除)。

## 恢复指引(冷启动)

1. 读本文件 → 看 Blocks 表 status。
2. 跑 gate 命令 `D:/AI/conda_envs/ouro/python.exe .../tests/run_all.py` 看当前哪个块挂。
3. B5 真机联测步骤:插 D455(USB3 口)→ `D:/AI/conda_envs/ouro/python.exe scripts/run_live.py --selftest`(无相机冒烟)→ `... scripts/run_live.py`(实况,按 q 退出,s 存图)。若 CUDA 不可用先查设备管理器(见 RTX3060 禁用事故 memory)。
