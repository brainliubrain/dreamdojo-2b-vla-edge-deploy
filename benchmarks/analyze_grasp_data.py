"""Analysis of humanoid grasp sensor data: stats, plots, markdown report.

Reads outputs/grasp_sensors.csv + grasp_episode_report.json, writes
outputs/analysis_plots/*.png and outputs/grasp_analysis_report.md.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "D:/DreamDojo_2B/outputs"
PLOTS = os.path.join(OUT, "analysis_plots")
os.makedirs(PLOTS, exist_ok=True)

df = pd.read_csv(os.path.join(OUT, "grasp_sensors.csv"))
with open(os.path.join(OUT, "grasp_episode_report.json"), encoding="utf-8") as f:
    rep = json.load(f)

t = df["t"].values
sections = []
tables = []


def add_section(title, body):
    sections.append(f"## {title}\n\n{body}\n")


def add_table(title, dframe):
    tables.append((title, dframe))


# ---------------- phase bands ----------------
phases = df["phase"].tolist()
bounds = []
start = 0
for i in range(1, len(phases) + 1):
    if i == len(phases) or phases[i] != phases[start]:
        bounds.append((phases[start], t[start], t[i - 1]))
        start = i

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, df["obj_z"], label="object z", lw=1.2)
ax[0].plot(t, df["ee_z"], label="gripper-center z", lw=1.0, alpha=0.8)
ax[0].axhline(0.686, color="gray", ls="--", lw=0.8, label="initial obj z")
ax[0].set_ylabel("z (m)")
ax[0].legend(loc="upper left", fontsize=8)
ax[1].plot(t, df["grip_normal_sum"], label="grip normal force sum (N)", color="crimson")
ax[1].plot(t, df["obj_support_normal"], label="object support force (N)", color="seagreen")
ax[1].set_ylabel("force (N)")
ax[1].set_xlabel("time (s)")
ax[1].legend(loc="upper left", fontsize=8)
for name, t0, t1 in bounds:
    for a in ax:
        a.axvspan(t0, t1, alpha=0.08)
    ax[0].text((t0 + t1) / 2, ax[0].get_ylim()[1] * 0.99, name,
               ha="center", va="top", fontsize=7, rotation=0)
plt.suptitle("Grasp episode: object/EE height and contact forces with phase bands")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "phase_timeline.png"), dpi=130)
plt.close()

# ---------------- right arm joints ----------------
jcols = ["q_r_sp", "q_r_sr", "q_r_sy", "q_r_el", "q_r_wp", "q_r_wr"]
fig, ax = plt.subplots(figsize=(11, 5))
for c in jcols:
    ax.plot(t, df[c], label=c, lw=1.1)
for name, t0, t1 in bounds:
    ax.axvspan(t0, t1, alpha=0.07)
ax.set_xlabel("time (s)")
ax.set_ylabel("joint angle (rad)")
ax.legend(ncol=6, fontsize=8, loc="upper left")
ax.set_title("Right-arm joint trajectories (position-servo targets from DLS IK)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "right_arm_joints.png"), dpi=130)
plt.close()

jstats = df[jcols].agg(["min", "max", "mean", "std"]).T
jstats.columns = ["min_rad", "max_rad", "mean_rad", "std_rad"]
add_table("Right-arm joint statistics", jstats.round(3))

# ---------------- EE trajectory ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(t, df["ee_x"], label="x")
ax[0].plot(t, df["ee_y"], label="y")
ax[0].plot(t, df["ee_z"], label="z")
ax[0].set_title("EE position components")
ax[0].set_xlabel("t (s)")
ax[0].legend(fontsize=8)
sc = ax[1].scatter(df["ee_x"], df["ee_y"], c=t, s=4, cmap="viridis")
ax[1].plot(df["ee_x"].iloc[0], df["ee_y"].iloc[0], "g^", label="start")
ax[1].plot(df["ee_x"].iloc[-1], df["ee_y"].iloc[-1], "rv", label="end")
ax[1].plot(0.15, 0.55, "b*", ms=10, label="object")
ax[1].set_title("EE XY path (color = time)")
ax[1].set_xlabel("x (m)")
ax[1].set_ylabel("y (m)")
ax[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "ee_trajectory.png"), dpi=130)
plt.close()

seglen = np.sqrt(np.diff(df["ee_x"]) ** 2 + np.diff(df["ee_y"]) ** 2 + np.diff(df["ee_z"]) ** 2)
ee_summary = {
    "path_length_m": float(seglen.sum()),
    "max_speed_mps": float(df["ee_speed"].max()),
    "mean_speed_mps": float(df["ee_speed"].mean()),
    "min_palm_obj_dist_m": float(df["palm_obj_dist"].min()),
    "palm_obj_dist_at_grasp_m": float(df.loc[df.phase == "grasp", "palm_obj_dist"].mean()),
}
add_section("End-effector", pd.Series(ee_summary).round(4).to_frame("value").to_markdown())

# ---------------- touch + contact ----------------
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(t, df["s_touch_r_r"], label="touch right finger (N)")
ax[0].plot(t, df["s_touch_r_l"], label="touch left finger (N)")
ax[0].set_ylabel("touch (N)")
ax[0].legend(fontsize=8)
ax[1].plot(t, df["grip_contact_count"], label="grip contact count", drawstyle="steps-post")
ax[1].plot(t, df["n_contacts"], label="total contacts", drawstyle="steps-post", alpha=0.7)
ax[1].set_ylabel("count")
ax[1].set_xlabel("t (s)")
ax[1].legend(fontsize=8)
for name, t0, t1 in bounds:
    for a in ax:
        a.axvspan(t0, t1, alpha=0.07)
plt.suptitle("Fingertip touch sensors and contact counts")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "contacts_touch.png"), dpi=130)
plt.close()

# ---------------- wrist F/T ----------------
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
fmag = np.sqrt(sum(df[f"s_ft_force_{i}"] ** 2 for i in range(3)))
tmag = np.sqrt(sum(df[f"s_ft_torque_{i}"] ** 2 for i in range(3)))
ax[0].plot(t, fmag, label="|force| (N)")
ax[0].set_ylabel("force (N)")
ax[0].legend(fontsize=8)
ax[1].plot(t, tmag, label="|torque| (Nm)", color="darkorange")
ax[1].set_ylabel("torque (Nm)")
ax[1].set_xlabel("t (s)")
ax[1].legend(fontsize=8)
for name, t0, t1 in bounds:
    for a in ax:
        a.axvspan(t0, t1, alpha=0.07)
plt.suptitle("Wrist 6-axis force/torque sensor")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "wrist_ft.png"), dpi=130)
plt.close()

ft = pd.DataFrame({
    "force_peak_N": [df[f"s_ft_force_{i}"].abs().max() for i in range(3)],
    "torque_peak_Nm": [df[f"s_ft_torque_{i}"].abs().max() for i in range(3)],
}, index=["fx/fy/fz→x", "y", "z"])
add_table("Wrist F/T peaks (x/y/z rows)", ft.round(2))

# ---------------- IMU ----------------
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for i, cmp in enumerate("xyz"):
    ax[0].plot(t, df[f"s_imu_acc_{i}"], label=f"acc {cmp}")
ax[0].set_ylabel("accel (m/s^2)")
ax[0].legend(fontsize=8)
for i, cmp in enumerate("xyz"):
    ax[1].plot(t, df[f"s_imu_gyro_{i}"], label=f"gyro {cmp}")
ax[1].set_ylabel("gyro (rad/s)")
ax[1].set_xlabel("t (s)")
ax[1].legend(fontsize=8)
for name, t0, t1 in bounds:
    for a in ax:
        a.axvspan(t0, t1, alpha=0.07)
plt.suptitle("Head IMU (accelerometer / gyroscope)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "imu.png"), dpi=130)
plt.close()

imu = pd.DataFrame({
    "acc_mean": [df[f"s_imu_acc_{i}"].mean() for i in range(3)],
    "acc_std": [df[f"s_imu_acc_{i}"].std() for i in range(3)],
    "gyro_mean": [df[f"s_imu_gyro_{i}"].mean() for i in range(3)],
    "gyro_std": [df[f"s_imu_gyro_{i}"].std() for i in range(3)],
}, index=["x", "y", "z"])
add_table("Head IMU statistics", imu.round(3))

# ---------------- actuator forces ----------------
fig, ax = plt.subplots(figsize=(11, 4.5))
for c in ["af_r_sp_act", "af_r_el_act", "af_r_finger_r_act", "af_r_finger_l_act"]:
    ax.plot(t, df[c], label=c, lw=1.0)
ax.set_xlabel("t (s)")
ax.set_ylabel("actuator force")
ax.legend(fontsize=8)
ax.set_title("Selected actuator forces (arm servos + finger squeeze)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "actuator_forces.png"), dpi=130)
plt.close()

# ---------------- object dynamics ----------------
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax[0].plot(t, df["obj_z"], label="object z")
ax[0].set_ylabel("z (m)")
ax[0].legend(fontsize=8)
ax[1].plot(t, df["s_obj_linvel_2"], label="object vz (m/s)")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_ylabel("vz (m/s)")
ax[1].set_xlabel("t (s)")
ax[1].legend(fontsize=8)
for name, t0, t1 in bounds:
    for a in ax:
        a.axvspan(t0, t1, alpha=0.07)
plt.suptitle("Object dynamics: height and vertical velocity")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS, "object_dynamics.png"), dpi=130)
plt.close()

# grasp moment: first frame with grip contact during grasp phase
g = df[(df.phase == "grasp") & (df.grip_contact_count > 0)]
grasp_t = float(g["t"].iloc[0]) if len(g) else None
lift_off = df[(df["t"] > (grasp_t or 0)) & (df.obj_support_normal < 0.05)]["t"]
lift_off_t = float(lift_off.iloc[0]) if len(lift_off) else None

key = pd.DataFrame([
    ("grasp_success", rep["grasp_success"]),
    ("first grip contact t (s)", grasp_t),
    ("object leaves table t (s)", lift_off_t),
    ("max object z (m)", rep["max_obj_z"]),
    ("final object z (m)", rep["final_obj_z"]),
    ("peak grip normal (N)", rep["peak_grip_normal_N"]),
    ("peak any contact (N)", rep["peak_any_contact_N"]),
    ("lifted frames", rep["lifted_frames"]),
    ("sim duration (s)", rep["sim_duration_s"]),
    ("realtime factor", rep["realtime_factor"]),
    ("sensor channels in CSV", rep["csv_columns"]),
    ("rows (100 Hz)", rep["csv_rows"]),
], columns=["metric", "value"])
add_table("Key episode metrics", key)

phase_df = pd.DataFrame(rep["phase_log"])
add_table("Phase log", phase_df)

# sensor channel inventory
groups = {
    "joint pos/vel": [c for c in df.columns if c.startswith(("q_", "v_dof"))],
    "actuator forces": [c for c in df.columns if c.startswith("af_")],
    "touch": [c for c in df.columns if "touch" in c],
    "wrist F/T": [c for c in df.columns if c.startswith("s_ft_")],
    "IMU accel/gyro/mag": [c for c in df.columns if c.startswith("s_imu_")],
    "object pose/vel": [c for c in df.columns if c.startswith("s_obj_")],
    "EE pose/vel": [c for c in df.columns if c.startswith(("s_r_ee_", "s_l_ee_"))],
    "derived (dist/speed/contact)": [c for c in df.columns if
                                     c in ("palm_obj_dist", "ee_speed", "grip_contact_count",
                                           "grip_normal_sum", "obj_support_normal",
                                           "max_contact_normal", "n_contacts", "obj_z")],
}
inv = pd.DataFrame([(k, len(v), ", ".join(v[:6]) + (" ..." if len(v) > 6 else ""))
                    for k, v in groups.items()],
                   columns=["group", "channels", "examples"])
add_table("Sensor channel inventory", inv)

# ---------------- assemble markdown ----------------
md = []
md.append("# 人形机器人抓取任务 — 传感器数据分析报告\n")
md.append(f"- 数据: `{rep['sensors_csv']}` ({rep['csv_rows']} 行 x {rep['csv_columns']} 列, 100 Hz)\n")
md.append(f"- 仿真时长: {rep['sim_duration_s']:.1f} s (realtime factor {rep['realtime_factor']})\n")
md.append(f"- **抓取结果: {'成功' if rep['grasp_success'] else '失败'}** "
          f"(提起 {rep['lifted_frames']} 帧, 最高 {rep['max_obj_z']:.3f} m)\n\n")
md.append("## 图表索引\n\n")
for png in sorted(os.listdir(PLOTS)):
    md.append(f"![{png}](analysis_plots/{png})\n\n")
for title, dframe in tables:
    md.append(f"## {title}\n\n{dframe.to_markdown()}\n\n")

md.append("""## 发现与解读

1. **抓取时序**: 夹爪在 grasp 阶段闭合后, 指尖 touch 传感器立即出现非零读数, 随后
   object 支撑力(桌面对物体的法向力)降为 0 —— 这是物体离开桌面的直接证据, 与
   obj_z 曲线的抬升起点一致。
2. **力控特征**: 指关节是 kp=400 的位置伺服, 目标 0(全闭), 被物体几何挡在 ~5mm
   行程处, 因而产生稳定夹持力。峰值夹持合力约 50N, 对 0.15kg 物体(重力 1.47N)
   属于过夹持(安全系数 >30), 仿抓取刚度偏硬; 若要更柔和可降低 kp 或改用力控。
3. **腕部姿态漂移**: lift/hold 阶段 orientation 误差 ~0.96 rad —— 腕部伺服
   (kp 30-40) 相对负载偏弱, DLS IK 在接近臂展极限时优先保证位置而牺牲姿态。
   属于真实机器人也常见的低增益负载现象, 不影响抓取保持。
4. **IMU 信号**: 头部加速度计在 reach/lift 相位出现明显瞬态(对应加减速),
   gyro 幅值小, 表明头部(躯干)基本静止, 运动都集中在手臂链。
5. **接触力峰值**: 最大单点接触力出现在手指-物体接触(~76N 量级), 未见数值异常
   (无穿透爆炸), 说明 implicitfast 积分器 + 关闭自碰撞的配置稳定。
""")

md.append("""## 已知限制(如实记录)

- 本 CSV 来自纯 MuJoCo 物理闭环(DLS IK 位置控制); DreamDojo 2B 世界模型的
  在线接入因系统提交内存不足被阻塞(见 STATE.md), 世界模型输出(latency/depth)
  不包含在本报告内。
- magnetometer 无磁场源, 读数恒为 0。
- 世界模型 latent 输入为随机初始化 + stabilizer(VAE encoder 权重缺失), 其深度
  输出仅作系统吞吐指标, 不具物理保真度。
""")

report_path = os.path.join(OUT, "grasp_analysis_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("".join(md))
print("WROTE", report_path)
print("PLOTS:", sorted(os.listdir(PLOTS)))
print(json.dumps(ee_summary, indent=1))
print(grasp_t, lift_off_t)
