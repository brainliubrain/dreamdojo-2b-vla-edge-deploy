"""Humanoid grasping simulation with full sensor extraction and optional
DreamDojo 2B world-model integration (6GB VRAM, INT8).

Usage:
  python humanoid_grasp_sim.py --smoke                 # quick validation run
  python humanoid_grasp_sim.py --episode 900           # full grasp episode, sensors -> CSV
  python humanoid_grasp_sim.py --episode 900 --with-dreamdojo   # + world model in the loop
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import mujoco

XML_PATH = "D:/DreamDojo_2B/humanoid_grasp.xml"
OUT_DIR = "D:/DreamDojo_2B/outputs"
ENGINE_DIR = "D:/DreamDojo_2B"

ARM_JOINTS = ["r_sp", "r_sr", "r_sy", "r_el", "r_wp", "r_wr"]
ARM_ACTS = ["r_sp_act", "r_sr_act", "r_sy_act", "r_el_act", "r_wp_act", "r_wr_act"]
GRIP_OPEN = 0.032
GRIP_CLOSE = 0.0
# ready pose: arm lifted, elbow bent, so the EE starts near the workspace
READY = {"r_sp": 0.35, "r_sr": -0.05, "r_sy": 0.0, "r_el": 0.35, "r_wp": 0.0, "r_wr": 0.0}
LEFT_REST = {"l_sp": -0.45, "l_sr": 0.10, "l_sy": 0.0, "l_el": 0.55, "l_wp": 0.0, "l_wr": 0.0}

# desired EE orientation: gripper pointing straight down (rotation pi about x)
R_DES = np.diag([1.0, -1.0, -1.0])

PHASES = ["lift_ready", "over_table", "descend", "grasp", "lift", "hold"]
PHASE_MAX_STEPS = {"lift_ready": 220, "over_table": 180, "descend": 220,
                   "grasp": 200, "lift": 170, "hold": 150}
VEL_LIMIT = 0.45  # m/s task-space speed clamp
OMEGA_LIMIT = 2.0


def rot_err_axisangle(R_cur):
    """World-frame rotation vector taking R_cur to R_DES."""
    R_err = R_DES @ R_cur.T
    w = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(w)
    if ang < 1e-6:
        return np.zeros(3)
    v = np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
    axis = v / (2.0 * np.sin(ang) + 1e-12)
    n = np.linalg.norm(axis)
    return axis / (n + 1e-12) * ang if n > 1e-9 else np.zeros(3)


class GraspSim:
    def __init__(self, with_dreamdojo=False, seed=0):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.m, self.d = self.model, self.data
        self.rng = np.random.default_rng(seed)
        self.with_dreamdojo = with_dreamdojo

        # name lookups
        self.arm_qposadr = np.array([self.m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dofadr = np.array([self.m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_actadr = np.array([self.m.actuator(a).id for a in ARM_ACTS])
        self.grip_actadr = [self.m.actuator(a).id for a in ["r_finger_r_act", "r_finger_l_act"]]
        self.finger_qposadr = [self.m.joint(j).qposadr[0] for j in ["r_f1", "r_f2"]]
        self.obj_qposadr = self.m.joint("object_free").qposadr[0]

        self.site_gripper = self.m.site("r_gripper_center").id
        self.geom_obj = self.m.geom("object_geom").id
        self.geoms_fingers = [self.m.geom(n).id for n in
                              ["r_finger_r_geom", "r_finger_l_geom", "r_palm"]]
        self.geoms_support = [self.m.geom(n).id for n in ["table", "floor"]]

        # sensor name -> slice
        self.sensor_slices = {}
        for i in range(self.m.nsensor):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i)
            adr, dim = self.m.sensor_adr[i], self.m.sensor_dim[i]
            self.sensor_slices[name] = slice(adr, adr + dim)

        # sensor data column list
        self.sensor_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_SENSOR, i)
                             for i in range(self.m.nsensor)]
        self.joint_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(self.m.njnt)]
        self.act_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                          for i in range(self.m.nu)]

        # initial state: robot in ready pose, fingers open
        mujoco.mj_resetData(self.m, self.d)
        for j, q in {**READY, **LEFT_REST,
                     "r_f1": GRIP_OPEN, "r_f2": GRIP_OPEN,
                     "l_f1": GRIP_OPEN, "l_f2": GRIP_OPEN}.items():
            self.d.qpos[self.m.joint(j).qposadr[0]] = q
        for j, q in {**READY, **LEFT_REST}.items():
            self.d.ctrl[self.m.actuator(j + "_act").id] = q
        self.q_ctrl = self.d.qpos[self.arm_qposadr].copy()
        mujoco.mj_forward(self.m, self.d)

        # contact force buffer
        self._cfbuf = np.zeros(6)

    # ---------------- control ----------------
    def ik_step(self, target_pos, dt_ctrl, k_pos=6.0, k_rot=6.0):
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site_gripper)
        J = np.vstack([jacp[:, self.arm_dofadr], jacr[:, self.arm_dofadr]])

        pos = self.d.site_xpos[self.site_gripper].copy()
        R_cur = self.d.site_xmat[self.site_gripper].reshape(3, 3)
        p_err = target_pos - pos
        v = p_err * k_pos
        nv = np.linalg.norm(v)
        if nv > VEL_LIMIT:
            v *= VEL_LIMIT / nv
        w = rot_err_axisangle(R_cur) * k_rot
        nw = np.linalg.norm(w)
        if nw > OMEGA_LIMIT:
            w *= OMEGA_LIMIT / nw
        twist = np.concatenate([v, w])

        damping = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)
        self.q_ctrl = self.q_ctrl + dq * dt_ctrl
        for i, j in enumerate(ARM_JOINTS):
            lo_, hi_ = self.m.jnt_range[self.m.joint(j).id]
            self.q_ctrl[i] = np.clip(self.q_ctrl[i], lo_ + 0.02, hi_ - 0.02)
        self.d.ctrl[self.arm_actadr] = self.q_ctrl

    def ee_error(self, target_pos):
        pos = self.d.site_xpos[self.site_gripper]
        R_cur = self.d.site_xmat[self.site_gripper].reshape(3, 3)
        return np.linalg.norm(target_pos - pos), np.linalg.norm(rot_err_axisangle(R_cur))

    def contacts_summary(self):
        grip_n = 0.0
        grip_cnt = 0
        sup_n = 0.0
        max_n = 0.0
        n_total = self.d.ncon
        for i in range(n_total):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            mujoco.mj_contactForce(self.m, self.d, i, self._cfbuf)
            fn = abs(float(self._cfbuf[0]))
            max_n = max(max_n, fn)
            pair = {g1, g2}
            if self.geom_obj in pair and any(g in pair for g in self.geoms_fingers):
                grip_n += fn
                grip_cnt += 1
            elif self.geom_obj in pair and any(g in pair for g in self.geoms_support):
                sup_n += fn
        return grip_cnt, grip_n, sup_n, max_n, n_total

    # ---------------- DreamDojo world model ----------------
    def load_world_model(self):
        import torch
        import sys
        sys.path.insert(0, ENGINE_DIR)
        import dreamdojo_standalone_engine_win as dd

        self.torch = torch
        print("[WM] loading DreamDojo 2B (INT8, pre-quantized shards) ...", flush=True)
        # meta-device build + to_empty(cuda): no 2.1GB CPU allocation, which the
        # system cannot afford while WSL holds most of the commit charge
        with torch.device("meta"):
            model = dd.DreamDojoDiT2B(num_blocks=28, model_channels=2048, action_dim=384,
                                      num_action_chunk=4, use_int8=True)
        model.to_empty(device="cuda")
        model.eval()

        int8_dir = os.path.join(ENGINE_DIR, "transformer_int8")
        with open(os.path.join(int8_dir, "model_int8_index.json")) as f:
            wm_index = json.load(f)["weight_map"]
        shards = sorted(set(wm_index.values()))
        from safetensors import safe_open
        for shard in shards:
            sd = {}
            with safe_open(os.path.join(int8_dir, shard), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    sd[k] = sf.get_tensor(k)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            # every key must land; scales buffers are part of the module state dict
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
            assert all(("scales" in k or "int8" in k) for k in missing), \
                f"missing non-quant keys: {missing[:5]}"
            del sd
            print(f"[WM]   loaded {shard}", flush=True)
        model.eval()
        self.wm_model = model
        self.wm_depth = dd.LatentDepthHead(in_channels=16).to("cuda")
        self.wm_stab = dd.LatentStabilizer(
            vae_config_path=os.path.join(ENGINE_DIR, "vae", "config.json"),
            device="cuda", dtype=torch.bfloat16)

        g = torch.Generator().manual_seed(0)
        self.wm_latent = (torch.randn(1, 18, 1, 30, 30, generator=g) * 0.5).to("cuda", torch.bfloat16)
        self.wm_text = (torch.randn(1, 77, 1024, generator=g) * 0.5).to("cuda", torch.bfloat16)
        self.wm_t = torch.tensor([500.0], dtype=torch.bfloat16, device="cuda")
        self.wm_canonical = self.wm_latent[:, :16].clone()
        self.wm_calls = 0
        self.wm_log = []
        print("[WM] loaded, VRAM GB:", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

    def world_model_step(self, action_history):
        """action_history: list of last 4 action vectors (7-dim each)."""
        torch = self.torch
        act = np.array(action_history[-4:], dtype=np.float32)  # (4,7)
        if act.shape[0] < 4:
            act = np.vstack([np.repeat(act[:1], 4 - act.shape[0], axis=0), act])
        act = act / np.array([0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 3.0], dtype=np.float32)
        act384 = np.tile(act.reshape(-1), 384 // act.size + 1)[:384]
        actions = torch.from_numpy(act384).view(1, 4, 384).to("cuda", torch.bfloat16)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = self.wm_model(self.wm_latent, self.wm_t, actions, self.wm_text)
            raw_next = self.wm_latent[:, :16] + pred
            stab = self.wm_stab(raw_next, self.wm_canonical, enable_scheme2=True, enable_scheme3=True)
            self.wm_latent[:, :16].copy_(stab)
            depth = self.wm_depth(self.wm_latent[:, :16])
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        dmin, dmean, dmax, dstd = (float(depth.min()), float(depth.mean()),
                                   float(depth.max()), float(depth.std()))
        vram = torch.cuda.max_memory_allocated() / 1e9
        self.wm_calls += 1
        if self.wm_calls in (1, 6, 12):
            arr = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).clamp(0, 255).byte().cpu().numpy()
            from PIL import Image
            Image.fromarray(arr, mode="L").save(
                os.path.join(OUT_DIR, f"worldmodel_depth_{self.wm_calls:02d}.png"))
        row = {"wm_call": self.wm_calls, "sim_time": float(self.d.time),
               "latency_ms": latency_ms, "vram_peak_gb": vram,
               "depth_min": dmin, "depth_mean": dmean, "depth_max": dmax, "depth_std": dstd}
        self.wm_log.append(row)
        return row

    # ---------------- main loop ----------------
    def run(self, total_steps, wm_every=50, physics_per_ctrl=5):
        dt_ctrl = self.m.opt.timestep * physics_per_ctrl
        rows = []
        phase_idx = 0
        phase_step = 0
        phase_log = []
        target = None
        action_hist = deque(maxlen=4)
        prev_ee = self.d.site_xpos[self.site_gripper].copy()
        prev_t = 0.0
        t_start = time.perf_counter()

        for step in range(total_steps):
            obj0 = np.array([0.15, 0.55, 0.686])
            phase = PHASES[phase_idx]

            if phase == "lift_ready":
                target = np.array([0.20, 0.30, 0.90])
            elif phase == "over_table":
                target = np.array([0.15, 0.55, 0.90])
            elif phase == "descend":
                target = obj0 + np.array([0.0, 0.0, 0.03])
            elif phase == "grasp":
                target = None  # hold position
            elif phase in ("lift", "hold"):
                target = obj0 + np.array([0.0, 0.0, 0.15])

            if target is not None:
                self.ik_step(target, dt_ctrl)
            else:
                pass  # grasp phase: freeze arm targets, only close fingers
            # gripper
            grip_cmd = GRIP_CLOSE if phase in ("grasp", "lift", "hold") else GRIP_OPEN
            self.d.ctrl[self.grip_actadr] = grip_cmd

            # left arm holds rest
            for j, q in LEFT_REST.items():
                self.d.ctrl[self.m.actuator(j + "_act").id] = q

            # step physics
            for _ in range(physics_per_ctrl):
                mujoco.mj_step(self.m, self.d)

            # sensors after physics
            mujoco.mj_forward(self.m, self.d)  # refresh site/sensor values consistently

            ee = self.d.site_xpos[self.site_gripper].copy()
            ee_vel = (ee - prev_ee) / dt_ctrl
            action_hist.append(np.concatenate([ee_vel * 2.0, [grip_cmd * 100.0]]).astype(np.float32))
            prev_ee = ee

            grip_cnt, grip_n, sup_n, max_n, ncon = self.contacts_summary()
            obj_pos = self.d.sensordata[self.sensor_slices["obj_pos"]].copy()
            row = {
                "t": float(self.d.time), "ctrl_step": step, "phase": phase,
                "obj_z": float(obj_pos[2]),
                "ee_x": float(ee[0]), "ee_y": float(ee[1]), "ee_z": float(ee[2]),
                "ee_speed": float(np.linalg.norm(ee_vel)),
                "palm_obj_dist": float(np.linalg.norm(ee - obj_pos)),
                "grip_contact_count": grip_cnt, "grip_normal_sum": grip_n,
                "obj_support_normal": sup_n, "max_contact_normal": max_n,
                "n_contacts": ncon,
                "grip_cmd": grip_cmd,
            }
            for i, jn in enumerate(self.joint_names):
                row["q_" + jn] = float(self.d.qpos[i])
            for i in range(self.m.nv):
                row["v_dof%d" % i] = float(self.d.qvel[i])
            for i, an in enumerate(self.act_names):
                row["af_" + an] = float(self.d.actuator_force[i])
            for sn in self.sensor_names:
                sl = self.sensor_slices[sn]
                vals = self.d.sensordata[sl]
                if len(vals) == 1:
                    row["s_" + sn] = float(vals[0])
                else:
                    for k in range(len(vals)):
                        row["s_%s_%d" % (sn, k)] = float(vals[k])
            rows.append(row)

            # world model call cadence
            if self.with_dreamdojo and step % wm_every == 0 and step > 0:
                wm = self.world_model_step(list(action_hist))
                print("[WM] call %d t=%.2f lat=%.0fms depth_mean=%.3f vram=%.2fGB" %
                      (wm["wm_call"], wm["sim_time"], wm["latency_ms"], wm["depth_mean"],
                       wm["vram_peak_gb"]), flush=True)

            # phase transition
            phase_step += 1
            pe, oe = (self.ee_error(target) if target is not None else (0, 0))
            converged = pe < 0.015 and oe < 0.06
            if (converged and phase_step >= 40) or phase_step >= PHASE_MAX_STEPS[phase]:
                phase_log.append({"phase": phase, "steps": phase_step,
                                  "t_enter": round(phase_log[-1]["t_exit"] if phase_log else 0.0, 3),
                                  "t_exit": round(float(self.d.time), 3),
                                  "pos_err": round(float(pe), 4), "ori_err": round(float(oe), 4),
                                  "converged": bool(converged)})
                phase_idx = min(phase_idx + 1, len(PHASES) - 1)
                phase_step = 0
                print("[PHASE] %s done at t=%.2f err=%.3f%s" %
                      (phase, self.d.time, pe, "" if converged else " (TIMEOUT)"), flush=True)

            if phase == "hold" and phase_step >= PHASE_MAX_STEPS["hold"]:
                break

        wall = time.perf_counter() - t_start

        # success criteria: object lifted above table+0.06 while gripped
        hold_rows = [r for r in rows if r["phase"] in ("lift", "hold")]
        lifted = [r for r in hold_rows if r["obj_z"] > 0.72 and r["grip_normal_sum"] > 0.1]
        success = len(lifted) > 30
        report = {
            "episode_steps": len(rows),
            "sim_duration_s": float(self.d.time),
            "wall_time_s": round(wall, 2),
            "realtime_factor": round(float(self.d.time) / wall, 3),
            "phase_log": phase_log,
            "grasp_success": bool(success),
            "lifted_frames": len(lifted),
            "max_obj_z": max(r["obj_z"] for r in rows),
            "final_obj_z": rows[-1]["obj_z"],
            "peak_grip_normal_N": max(r["grip_normal_sum"] for r in rows),
            "peak_any_contact_N": max(r["max_contact_normal"] for r in rows),
            "sensor_channels": len(rows[0]) - 4,
        }
        if self.with_dreamdojo:
            lat = [r["latency_ms"] for r in self.wm_log]
            report["world_model"] = {
                "calls": self.wm_calls,
                "mean_latency_ms": round(float(np.mean(lat)), 1),
                "peak_vram_gb": round(float(max(r["vram_peak_gb"] for r in self.wm_log)), 2),
                "mean_fps": round(1000.0 / float(np.mean(lat)), 2),
            }
        return rows, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--episode", type=int, default=900)
    ap.add_argument("--with-dreamdojo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 140 if args.smoke else args.episode
    sim = GraspSim(with_dreamdojo=args.with_dreamdojo, seed=args.seed)
    if args.with_dreamdojo:
        sim.load_world_model()

    rows, report = sim.run(total)
    report["config"] = {"smoke": args.smoke, "episode": args.episode,
                        "with_dreamdojo": args.with_dreamdojo, "seed": args.seed}

    import pandas as pd
    df = pd.DataFrame(rows)
    suffix = "_smoke" if args.smoke else ""
    csv_path = os.path.join(OUT_DIR, f"grasp_sensors{suffix}.csv")
    df.to_csv(csv_path, index=False)
    report["sensors_csv"] = csv_path
    report["csv_rows"] = len(df)
    report["csv_columns"] = len(df.columns)
    print("[CSV] %d rows x %d cols -> %s" % (len(df), len(df.columns), csv_path), flush=True)

    if args.with_dreamdojo:
        wm_csv = os.path.join(OUT_DIR, "world_model_log.csv")
        pd.DataFrame(sim.wm_log).to_csv(wm_csv, index=False)
        report["world_model_csv"] = wm_csv

    rep_path = os.path.join(OUT_DIR, ("grasp_report_smoke.json" if args.smoke
                                      else "grasp_episode_report.json"))
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2)[:2000], flush=True)
    print("[REPORT]", rep_path, flush=True)


if __name__ == "__main__":
    main()
