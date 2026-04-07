import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = "little robot (Alex).xml"

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
        "above": deg([25.0, 22.0, 31.0]),
        "approach": deg([-28.0, 22.0, 31.0]),
        "hook": deg([-70.0, 22.0, 38.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "R_arm": {
        "start": deg([0.0, 0.0, 0.0]),
        "above": deg([25.0, -22.0, -31.0]),
        "approach": deg([-28.0, -22.0, -31.0]),
        "hook": deg([-80.0, -22.0, -25.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "L_leg": {
        "start": deg([0.0, 0.0, 0.0, 0.0, 0.0]),
        "above": deg([0.0, -55.0, -8.0, -8.0, 0.0]),
        "approach": deg([0.0, -55.0, -82.0, -82.0, 0.0]),
        "hook": deg([0.0, -30.0, -53.0, -53.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
    "R_leg": {
        "start": deg([0.0, 0.0, 0.0, 0.0, 0.0]),   # 你圖裡是 None，這裡先用全 0 當 start
        "above": deg([0.0, 54.0, 8.0, -8.0, 0.0]),
        "approach": deg([0.0, 55.0, 82.0, -82.0, 0.0]),
        "hook": deg([0.0, 20, 49.0, -49.0, 0.0]),
        "extra_start": 0.0,
        "extra_hook": 0.0,
    },
}

# =========================================================
# helpers
# =========================================================
def get_full_state(which_pose="start"):
    """
    Build a full-body control dict:
    for every limb -> joints + extra
    """
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

def apply_full_state(state):
    for limb, cfg in limb_cfg.items():
        for aid, val in zip(cfg["act_ids"], state[limb]["joints"]):
            data.ctrl[aid] = float(val)

        if cfg["extra_aid"] is not None:
            data.ctrl[cfg["extra_aid"]] = float(state[limb]["extra"])

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
    for i in range(steps):
        alpha = (i + 1) / steps
        state = interpolate_states(state_a, state_b, alpha)
        apply_full_state(state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(sleep)

def hold(viewer, steps=80, sleep=0.01):
    for _ in range(steps):
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
# sequence builder
# =========================================================
def build_sequence_states():
    """
    Create full-body sequence:
    0. all start
    1. L_arm above
    2. L_arm approach
    3. L_arm hook
    4. R_arm above   (L_arm stays hook)
    5. R_arm approach
    6. R_arm hook
    7. L_leg above   (both arms stay hook)
    8. L_leg approach
    9. L_leg hook
    10. R_leg above  (L_arm, R_arm, L_leg stay hook)
    11. R_leg approach
    12. R_leg hook
    """
    states = []

    # S0: all start
    s0 = get_full_state("start")
    states.append(("All limbs at start", s0))

    # Left arm moves
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

    # Right arm moves, left arm stays hook
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

    # Left leg moves, both arms stay hook
    s7 = copy_state(states[-1][1])
    s7["L_leg"]["joints"] = pose_data["L_leg"]["above"].copy()
    states.append(("Left leg above", s7))

    s8 = copy_state(states[-1][1])
    s8["L_leg"]["joints"] = pose_data["L_leg"]["approach"].copy()
    states.append(("Left leg approach", s8))

    s9 = copy_state(states[-1][1])
    s9["L_leg"]["joints"] = pose_data["L_leg"]["hook"].copy()
    states.append(("Left leg hook", s9))

    # Right leg moves, others stay hook
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
    print_pose_summary()

    sequence = build_sequence_states()

    # initialize at all-start
    apply_full_state(sequence[0][1])

    for _ in range(200):
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    print("Start climbing standby sequence...")
    print("Sequence: L_arm -> R_arm -> L_leg -> R_leg")

    # transition timing
    for idx in range(len(sequence) - 1):
        label_a, state_a = sequence[idx]
        label_b, state_b = sequence[idx + 1]

        print(f"{idx:02d}: {label_a}  -->  {label_b}")

        # tune per stage
        if "above" in label_b:
            steps = 110
        elif "approach" in label_b:
            steps = 120
        else:  # hook
            steps = 140

        move_full_state(viewer, state_a, state_b, steps=steps, sleep=0.01)
        hold(viewer, steps=40, sleep=0.01)

    print("\nFull climbing standby pose reached.")
    print("Press ENTER to exit.")
    input()