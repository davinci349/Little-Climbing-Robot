import mujoco
import mujoco.viewer
import numpy as np
import time
import tkinter as tk
from threading import Thread

XML_PATH = "models/XS-Robot(Alex).xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# =========================================================
# actuator config
# =========================================================
limb_cfg = {
    "L_arm": {
        "label": "Left Arm",
        "act_names": ["act_shoulder_L", "act_elbow_L", "act_wrist_L"],
        "extra_name": "act_finger_L",
    },
    "R_arm": {
        "label": "Right Arm",
        "act_names": ["act_shoulder_R", "act_elbow_R", "act_wrist_R"],
        "extra_name": "act_finger_R",
    },
    "L_leg": {
        "label": "Left Leg",
        "act_names": ["act_hiproll_L", "act_hipyaw_L", "act_thigh_L", "act_knee_L", "act_ankle_L"],
        "extra_name": None,
    },
    "R_leg": {
        "label": "Right Leg",
        "act_names": ["act_hiproll_R", "act_hipyaw_R", "act_thigh_R", "act_knee_R", "act_ankle_R"],
        "extra_name": None,
    },
}

for limb in limb_cfg:
    cfg = limb_cfg[limb]
    cfg["act_ids"] = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in cfg["act_names"]
    ]
    if cfg["extra_name"] is not None:
        cfg["extra_aid"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg["extra_name"]
        )
    else:
        cfg["extra_aid"] = None

# =========================================================
# pose data
# =========================================================
def deg(vals):
    return np.deg2rad(np.array(vals, dtype=float))

pose_data = {
    "L_arm": {
        "start": deg([0.0, 0.0, 0.0]),
        "above": deg([20.0, 22.0, 0.0]),
        "approach": deg([-3.0, 22.0, 0.0]),
        "hook": deg([-3.0, 22.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "R_arm": {
        "start": deg([0.0, 0.0, 0.0]),
        "above": deg([20.0, -22.0, 0.0]),
        "approach": deg([-3.0, -22.0, 0.0]),
        "hook": deg([-3.0, -22.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "L_leg": {
        "start": deg([0.0, 0.0, 0.0, 0.0, 0.0]),
        "above": deg([0.0, -70.0, -65.0, -65.0, 0.0]),
        "approach": deg([0.0, -70.0, -90.0, -90.0, 0.0]),
        "hook": deg([0.0, -20.0, -52.0, -52.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "R_leg": {
        "start": deg([0.0, 0.0, 0.0, 0.0, 0.0]),
        "above": deg([0.0, 80.0, 65.0, -65.0, 0.0]),
        "approach": deg([0.0, 80.0, 90.0, -90.0, 0.0]),
        "hook": deg([0.0, 10.0, 50.0, -50.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
}

# =========================================================
# grouped sync offsets
# =========================================================
shoulder_sync_offset = 0.0
wrist_sync_offset = 0.0
thigh_sync_offset = 0.0
knee_sync_offset = 0.0

current_display_state = None

# =========================================================
# helpers
# =========================================================
def get_full_state(which_pose="start"):
    state = {}
    for limb in limb_cfg:
        state[limb] = {
            "joints": pose_data[limb][which_pose].copy(),
            "extra": pose_data[limb][f"extra_{which_pose}"],
        }
    return state

def copy_state(state):
    out = {}
    for limb in state:
        out[limb] = {
            "joints": np.array(state[limb]["joints"], dtype=float).copy(),
            "extra": float(state[limb]["extra"]),
        }
    return out

def clamp_to_ctrlrange(aid, val):
    low, high = model.actuator_ctrlrange[aid]
    return float(np.clip(val, low, high))

def apply_full_state(state):
    global shoulder_sync_offset, wrist_sync_offset, thigh_sync_offset, knee_sync_offset

    for limb, cfg in limb_cfg.items():
        joint_vals = np.array(state[limb]["joints"], dtype=float).copy()

        if limb == "L_arm":
            # [shoulder, elbow, wrist]
            joint_vals[0] += shoulder_sync_offset   # shoulder 保持原本方向
            joint_vals[2] += wrist_sync_offset      # wrist 正向

        elif limb == "R_arm":
            # [shoulder, elbow, wrist]
            joint_vals[0] += shoulder_sync_offset   # shoulder 保持原本方向
            joint_vals[2] -= wrist_sync_offset      # wrist 正向

        elif limb == "L_leg":
            # [hiproll, hipyaw, thigh, knee, ankle]
            joint_vals[2] += thigh_sync_offset      # left thigh
            joint_vals[3] += knee_sync_offset       # left knee

        elif limb == "R_leg":
            # [hiproll, hipyaw, thigh, knee, ankle]
            joint_vals[2] -= thigh_sync_offset      # right thigh
            joint_vals[3] += knee_sync_offset       # right knee

        for aid, val in zip(cfg["act_ids"], joint_vals):
            data.ctrl[aid] = clamp_to_ctrlrange(aid, val)

        if cfg["extra_aid"] is not None:
            data.ctrl[cfg["extra_aid"]] = clamp_to_ctrlrange(
                cfg["extra_aid"], state[limb]["extra"]
            )

def interpolate_states(state_a, state_b, alpha):
    out = {}
    for limb in limb_cfg:
        joints = (1 - alpha) * state_a[limb]["joints"] + alpha * state_b[limb]["joints"]
        extra = (1 - alpha) * state_a[limb]["extra"] + alpha * state_b[limb]["extra"]
        out[limb] = {
            "joints": joints,
            "extra": extra,
        }
    return out

def move_full_state(viewer, state_a, state_b, steps=100, sleep=0.01):
    global current_display_state
    for i in range(steps):
        alpha = (i + 1) / steps
        state = interpolate_states(state_a, state_b, alpha)
        current_display_state = copy_state(state)
        apply_full_state(state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(sleep)

def hold(viewer, steps=80, sleep=0.01):
    global current_display_state
    for _ in range(steps):
        if current_display_state is not None:
            apply_full_state(current_display_state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(sleep)

def print_pose_summary():
    print("\n=== Pose Summary ===")
    for limb, pdata in pose_data.items():
        print(f"\n{limb}:")
        for key in ["start", "above", "approach", "hook"]:
            print(f"  {key:8s}: {np.round(np.rad2deg(pdata[key]), 1).tolist()}")
    print("====================\n")

# =========================================================
# slider UI
# =========================================================
def launch_group_sliders():
    def on_shoulder(val):
        global shoulder_sync_offset
        shoulder_sync_offset = np.deg2rad(float(val))

    def on_wrist(val):
        global wrist_sync_offset
        wrist_sync_offset = np.deg2rad(float(val))

    def on_thigh(val):
        global thigh_sync_offset
        thigh_sync_offset = np.deg2rad(float(val))

    def on_knee(val):
        global knee_sync_offset
        knee_sync_offset = np.deg2rad(float(val))

    root = tk.Tk()
    root.title("Grouped Joint Sync Control")
    root.geometry("420x420")

    tk.Label(root, text="Grouped Joint Sync Control", font=("Arial", 13, "bold")).pack(pady=8)
    tk.Label(root, text="Drag sliders to rotate left/right joints together").pack(pady=2)

    tk.Label(root, text="Both Shoulders (deg)", font=("Arial", 10)).pack(pady=(12, 0))
    shoulder_slider = tk.Scale(
        root, from_=-90, to=90, orient="horizontal",
        length=320, resolution=1, command=on_shoulder
    )
    shoulder_slider.set(0)
    shoulder_slider.pack()

    tk.Label(root, text="Both Wrists (deg)", font=("Arial", 10)).pack(pady=(10, 0))
    wrist_slider = tk.Scale(
        root, from_=-90, to=90, orient="horizontal",
        length=320, resolution=1, command=on_wrist
    )
    wrist_slider.set(0)
    wrist_slider.pack()

    tk.Label(root, text="Both Thighs (deg)", font=("Arial", 10)).pack(pady=(10, 0))
    thigh_slider = tk.Scale(
        root, from_=-90, to=90, orient="horizontal",
        length=320, resolution=1, command=on_thigh
    )
    thigh_slider.set(0)
    thigh_slider.pack()

    tk.Label(root, text="Both Knees (deg)", font=("Arial", 10)).pack(pady=(10, 0))
    knee_slider = tk.Scale(
        root, from_=-90, to=90, orient="horizontal",
        length=320, resolution=1, command=on_knee
    )
    knee_slider.set(0)
    knee_slider.pack()

    def reset_all():
        shoulder_slider.set(0)
        wrist_slider.set(0)
        thigh_slider.set(0)
        knee_slider.set(0)

    tk.Button(root, text="Reset All Sliders", command=reset_all).pack(pady=12)

    root.mainloop()

# =========================================================
# sequence builder
# =========================================================
def build_sequence_states():
    states = []

    s0 = get_full_state("start")
    states.append(("All limbs at start", s0))

    s1 = copy_state(states[-1][1])
    s1["L_arm"]["joints"] = pose_data["L_arm"]["above"].copy()
    s1["L_arm"]["extra"] = pose_data["L_arm"]["extra_start"]
    states.append(("Left arm above", s1))

    s2 = copy_state(states[-1][1])
    s2["L_arm"]["joints"] = pose_data["L_arm"]["approach"].copy()
    states.append(("Left arm approach", s2))

    s3 = copy_state(states[-1][1])
    s3["L_arm"]["joints"] = pose_data["L_arm"]["hook"].copy()
    s3["L_arm"]["extra"] = pose_data["L_arm"]["extra_hook"]
    states.append(("Left arm hook", s3))

    s4 = copy_state(states[-1][1])
    s4["R_arm"]["joints"] = pose_data["R_arm"]["above"].copy()
    s4["R_arm"]["extra"] = pose_data["R_arm"]["extra_start"]
    states.append(("Right arm above", s4))

    s5 = copy_state(states[-1][1])
    s5["R_arm"]["joints"] = pose_data["R_arm"]["approach"].copy()
    states.append(("Right arm approach", s5))

    s6 = copy_state(states[-1][1])
    s6["R_arm"]["joints"] = pose_data["R_arm"]["hook"].copy()
    s6["R_arm"]["extra"] = pose_data["R_arm"]["extra_hook"]
    states.append(("Right arm hook", s6))

    s7 = copy_state(states[-1][1])
    s7["L_leg"]["joints"] = pose_data["L_leg"]["above"].copy()
    states.append(("Left leg above", s7))

    s8 = copy_state(states[-1][1])
    s8["L_leg"]["joints"] = pose_data["L_leg"]["approach"].copy()
    states.append(("Left leg approach", s8))

    s9 = copy_state(states[-1][1])
    s9["L_leg"]["joints"] = pose_data["L_leg"]["hook"].copy()
    states.append(("Left leg hook", s9))

    s10 = copy_state(states[-1][1])
    s10["R_leg"]["joints"] = pose_data["R_leg"]["above"].copy()
    states.append(("Right leg above", s10))

    s11 = copy_state(states[-1][1])
    s11["R_leg"]["joints"] = pose_data["R_leg"]["approach"].copy()
    states.append(("Right leg approach", s11))

    s12 = copy_state(states[-1][1])
    s12["R_leg"]["joints"] = pose_data["R_leg"]["hook"].copy()
    states.append(("Right leg hook / full standby", s12))

    return states

# =========================================================
# main
# =========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    slider_thread = Thread(target=launch_group_sliders, daemon=True)
    slider_thread.start()

    print_pose_summary()

    sequence = build_sequence_states()

    current_display_state = copy_state(sequence[0][1])
    apply_full_state(current_display_state)

    for _ in range(200):
        apply_full_state(current_display_state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print("Start climbing standby sequence...")
    print("Sequence: L_arm -> R_arm -> L_leg -> R_leg")

    for idx in range(len(sequence) - 1):
        label_a, state_a = sequence[idx]
        label_b, state_b = sequence[idx + 1]

        print(f"{idx:02d}: {label_a}  -->  {label_b}")

        if "above" in label_b:
            steps = 110
        elif "approach" in label_b:
            steps = 120
        else:
            steps = 140

        move_full_state(viewer, state_a, state_b, steps=steps, sleep=0.01)
        hold(viewer, steps=40, sleep=0.01)

    print("\nFull climbing standby pose reached.")
    print("Now use the slider window to tune grouped joints.")

    current_display_state = copy_state(sequence[-1][1])

    while viewer.is_running():
        apply_full_state(current_display_state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)