# D455 × YOLO26-Depth 动态尺度锚定融合管线

单 RGB 上的 YOLO26-Depth(Ultralytics 官方米制单目深度)负责**远场补盲**;D455 双目在近场(0.5-4m)物理精度可靠,负责**逐帧锚定校准**单目深度的尺度漂移;两路按距离分区混合输出一张稠密米制深度图。

```
RGB(848x480) ──→ YOLO26-Depth(768) ─→ y 米制深度 ──→ [RANSAC 校准 S,t] ─→ y 校准 ─┐
                                                                 ↑                  ├→ 三区融合 → fused
D455 depth ─────→ 高置信掩码(0.5-4m+梯度平滑+腐蚀)───────────────┘(EMA+跳变守卫)   ┘
```

## 三区融合规则(Zone-based Multiplexing)

| 区间 | 权重 | 说明 |
|------|------|------|
| 0–4m | D455 = 100% | 近场物理深度毫米级可信;失效像素默认由校准 YOLO 补洞(`fill_near_holes`,可关→洞保持 0 安全语义) |
| 4–7m | smoothstep 线性过渡 | 无断崖,4m/7m 边界处数学连续 |
| 7m+ | 校准 YOLO = 100% | 远场规划,D455 数据弃用 |

## 校准模型(两参数,RANSAC + 最小二乘精修)

- **linear(默认)**:`d = S·y + t`。外推有界,域外远场安全。
- **log(可选)**:`log d = a·log y + b`,与深度头原生校准形式一致,域内锚定区略优,**但域外远场会指数爆炸**(合成压力实测 9m 立柱被校到 25.4m,linear 仅 11.6m),故不作默认。
- 时间平滑:EMA(α=0.2)+ 跳变守卫(参考点读数跳变 >25% 拒收,连续 3 帧才接受=场景真实变化);锚定像素不足或 RANSAC 共识退化 → 沿用上一帧参数。

## 验证结果(2026-10-07,3060 Laptop 6GB,ultralytics 8.4.104,yolo26m-depth.pt)

**合成走廊场景(真值已知,含对抗性域差距)**:`tests/run_all.py` 4/4 绿,`outputs/e2e_metrics.json`:
- 锚定区中位相对误差:raw 19.1% → linear 14.2%(log 12.9%)
- 远柱 9m(D455 全盲):原始读数 4.3m(误差 52%)→ linear 锚定后 **11.6m(误差 29%)**,远场误差砍半

**真机(D455,桌底暗场景)**:30 帧连续运行,内点率 48–57%,校准参数帧间波动 <3%;单目在暗场景低估 ~45%(implied@2m 3.4-3.6m),被锚定逐帧拉回。面板:`outputs/live/last_panel.png`。

**性能(768 输入,m 档)**:模型 44–88ms/帧 + 对齐拟合 ~80ms,实测 4.7–7 FPS。提速路径:`--imgsz 512`(近似线性提速)、换 `yolo26s-depth.pt`、对齐拟合进一步向量化。

## 用法

```bash
PY=D:/AI/conda_envs/ouro/python.exe   # 依赖全在此 env(torch 2.7.0+cu128, 用户站点 ultralytics 8.4.104)
cd C:/Users/HHH/.zcode/workspace/default/d455_yolo_depth_fusion

$PY scripts/run_live.py                # D455 实况窗口(q 退出,s 存图)
$PY scripts/run_live.py --selftest     # 无相机自测(MockSource 合成走廊)
$PY scripts/run_live.py --fit-mode log --imgsz 512   # 可选项
$PY scripts/run_offline.py --bag x.bag --step 5      # bag 回放批处理
$PY tests/run_all.py                   # 全部验收(gate.json 指向这里)
```

配置全在 `d455_fusion/config.py`(锚定窗口/区界/EMA/RANSAC 阈值);权重路径环境变量 `D455_FUSION_WEIGHTS`,默认 `D:/AI/d455_fusion/models/yolo26m-depth.pt`(C 盘紧张,权重一律放 D 盘)。

## 已知限制

1. **log 模式外推爆炸**:见上,选 linear。
2. **对齐拟合 ~80ms**:RANSAC 评分矩阵已 float32+8192 子采样,精修已闭式化;进一步优化可批内点选择或降锚定分辨率。
3. 单目深度对 OOD 场景(合成图、极端光照)形状畸变大,锚定只能纠尺度不能纠形状;远场精度依赖模型域内程度。

## 防重影与近端盲区(2026-10-07 第二轮,官方算法落地)

**重影三来源与对策**(对照前后面板 `outputs/ghost_ab/`):
- 传感器侧:官方白皮书建议的**视差域边缘保持滤波链** `disparity_transform → spatial(α.5,δ20,mag2) → temporal → 反变换`(治边缘晕圈/双轮廓/噪声斑点,洞保持 0 不造假数据);
- 融合侧:**有效密度门控**(31px 窗,有效率<0.1 的孤立噪声岛权重归零,让位 YOLO;治浮游斑块)+ **边界羽化**(距失效区 4px 内权重距离变换递减;治斑块硬边)。均可配置关闭,新增单测覆盖。

**近端盲区 minZ(官方两旋钮,实测)**:
- `--disparity-shift N`(Advanced Mode 深度表):**shift8 推荐**(848 分辨率 minZ 0.33→0.31m,2.5-4.2m 锚定带 136k px 不降反增);**shift16/32 否决**(锚定带坍缩到 20k/0 px,远端量程被吃掉);**STDepthTable 持久化在设备上**,管线每次启动无条件显式写入(默认 0=出厂),防参数泄漏。
- `--depth-width 424`(minZ 与 X 分辨率线性):实测锚定带覆盖掉到 33k px,得不偿失,不推荐。
- <0.3m 极近场本来就该由 YOLO 补(锚定窗从 0.5m 起)。

**D455 USB 假死救援(不再需要拔插)**:`pnputil /restart-device 'USB\VID_8086&PID_0B5C&MI_00\...'`(Depth+RGB 两个节点,需管理员/UAC),实测可复活 hardware_reset 救不回的假死;诱因是进程异常退出(尤其高级模式切换在开流后进行)。已修:高级模式在 pipeline.start **之前**切换 + read() 空帧重试 + __del__ 兜底关闭。幽灵节点(Status Unknown)无害可留。

## 动态拖影与近处硬壳收官(2026-10-08 第三轮,全面消除边沿残迹)

针对 Zcode 前期遗留的“手猛动时 FUSED 边沿残迹”以及“近处(<0.6m)硬壳断崖”深度攻坚落地:
1. **滤波拓扑校正与视差时域解耦 (Motion Ghosting)**:
   - 彻底修复 `RealSenseSource` 滤波链路：强制在 **Disparity 视差域** 中执行 `threshold → to_disp → spatial → temporal → from_disp`。
   - 解决手快速晃动残影的核心：设置 `rs.option.holes_fill = 1`（仅连续有效才平滑，彻底消除手移走后旧帧在原地残留 2~3 帧的幽灵鬼影）+ `filter_smooth_alpha = 0.65`（大幅提高动态瞬态响应）。
   - 传感器级 `threshold_filter` 截断 0.30m~8.0m，剔除近端超视差产生的漂移非零飞点。
2. **近端盲区平滑过渡 (<0.6m Hard-Shell Elimination)**:
   - 新增 `near_blind_min: 0.32m` 到 `near_blind_max: 0.50m` 双向 smoothstep 权重门控。
   - 物体靠近相机逼近 0.5m~0.32m 时，D455 权重平滑渐变递减，100% 由 YOLO 单目无缝接管，彻底抹平硬壳边缘断崖与碎块撕裂。
3. **遮挡边缘深度阶跃羽化 (Depth Jump Discontinuity Feathering)**:
   - 针对手部、头部轮廓与背景交界处（前景背景交界视差拉扯区），检测局部深度跳变（$|\Delta d|/d > 0.20$），外扩 3px 边缘带将 D455 权重衰减让位给与 RGB 像素级对齐的 YOLO 单目，彻底根治轮廓双边缘虚影。
4. **连通域面积斑点过滤 (Speckle Rejection)**:
   - `cv2.connectedComponentsWithStats` 自动剔除面积小于 150px 的孤立噪点碎块，彻底清除散落在桌面上/半空中的浮游斑块。
5. **RANSAC 外推安全边界**:
   - 约束线性拟合尺度 $S \in [0.1, 5.0]$ 并严格进行精修边界截断，防止单目在特定局部区域病态拉伸导致与双目数值脱节。
- 验收结果: `tests/run_all.py` 4/4 绿(含新单测 `test_near_blind_and_depth_jump`)，真机 25 帧连续实测证实人物边缘双轮廓与桌面悬空斑块完全消除，面板落盘 `outputs/live/last_panel.png`。

## 结构

```
d455_fusion/  config.py alignment.py fusion.py backends.py pipeline.py viz.py
capture/      realsense_source.py(RealSenseSource + MockSource 走廊场景)
scripts/      run_live.py run_offline.py
tests/        test_alignment test_fusion test_e2e test_live run_all.py
outputs/      证据产物(面板 png + 指标 json)
```
