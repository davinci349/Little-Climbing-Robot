import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import tkinter as tk

XML_PATH = "models/XS-Robot(Alex).xml"

# =========================================================
# load model
# =========================================================
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# =========================================================
# helper
# =========================================================
def deg(vals):
    return np.deg2rad(np.array(vals, dtype=float))

def rad_to_deg(x):
    return float(np.rad2deg(x))

def deg_to_rad(x):
    return float(np.deg2rad(x))

def clamp_to_ctrlrange(aid, val):
    low, high = model.actuator_ctrlrange[aid]
    return float(np.clip(val, low, high))

# =========================================================
# pose data
# =========================================================
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
# limb config
# =========================================================
limb_cfg = {
    "L_arm": {
        "label": "Left Arm",
        "act_names": ["act_shoulder_L", "act_elbow_L", "act_wrist_L"],
        "extra_name": "act_finger_L",
        "joint_labels": ["shoulder", "elbow", "wrist"],
        "deg_ranges": [(-60, 60), (-60, 60), (-60, 60)],
        "extra_range": (-0.4, 0.4),
    },
    "R_arm": {
        "label": "Right Arm",
        "act_names": ["act_shoulder_R", "act_elbow_R", "act_wrist_R"],
        "extra_name": "act_finger_R",
        "joint_labels": ["shoulder", "elbow", "wrist"],
        "deg_ranges": [(-60, 60), (-60, 60), (-60, 60)],
        "extra_range": (-0.4, 0.4),
    },
    "L_leg": {
        "label": "Left Leg",
        "act_names": ["act_hiproll_L", "act_hipyaw_L", "act_thigh_L", "act_knee_L", "act_ankle_L"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "deg_ranges": [(-60, 60), (-60, 60), (-60, 60), (-60, 60), (-60, 60)],
        "extra_range": None,
    },
    "R_leg": {
        "label": "Right Leg",
        "act_names": ["act_hiproll_R", "act_hipyaw_R", "act_thigh_R", "act_knee_R", "act_ankle_R"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "deg_ranges": [(-60, 60), (-60, 60), (-60, 60), (-60, 60), (-60, 60)],
        "extra_range": None,
    },
}

# actuator ids
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
# runtime state
# =========================================================
running = True
lock = threading.Lock()
active_limb = {"name": "L_arm"}

# climbed base pose after auto-start-climb
base_state = {}
for limb in limb_cfg:
    n = len(limb_cfg[limb]["act_ids"])
    base_state[limb] = {
        "joints": np.zeros(n, dtype=float),
        "extra": 0.0,
    }

# individual offsets
individual_offsets = {}
for limb in limb_cfg:
    n = len(limb_cfg[limb]["act_ids"])
    individual_offsets[limb] = {
        "joints": np.zeros(n, dtype=float),
        "extra": 0.0,
    }

# grouped offsets
group_offsets = {
    "shoulder": 0.0,
    "wrist": 0.0,
    "thigh": 0.0,
    "knee": 0.0,
}

sequence_done = {"value": False}

# saved next climbing steps
step_names = ["step1", "step2", "step3", "step4"]
saved_steps = {name: None for name in step_names}

# play requests
play_request = {
    "play_step": None,
    "play_all": False,
}

# =========================================================
# state helpers
# =========================================================
def copy_full_state(state):
    out = {}
    for limb in state:
        out[limb] = {
            "joints": np.array(state[limb]["joints"], dtype=float).copy(),
            "extra": float(state[limb]["extra"]),
        }
    return out

def get_current_full_state():
    """
    current final posture = base_state + individual_offsets + grouped_offsets
    """
    out = {}
    for limb in limb_cfg:
        out[limb] = {
            "joints": get_final_joint_values(limb).copy(),
            "extra": float(get_final_extra_value(limb)),
        }
    return out

def set_base_state_from_full_state(state):
    """
    save whole-body posture as new base state,
    and reset all offsets to zero
    """
    for limb in limb_cfg:
        base_state[limb]["joints"] = np.array(state[limb]["joints"], dtype=float).copy()
        base_state[limb]["extra"] = float(state[limb]["extra"])

    reset_individual_offsets_all()
    reset_group_offsets()

# =========================================================
# tools
# =========================================================
def get_joint_index(limb, joint_name):
    cfg = limb_cfg[limb]
    for i, name in enumerate(cfg["joint_labels"]):
        if name == joint_name:
            return i
    return None

def get_group_offset_vector(limb):
    cfg = limb_cfg[limb]
    vec = np.zeros(len(cfg["act_ids"]), dtype=float)

    if limb == "L_arm":
        sidx = get_joint_index(limb, "shoulder")
        widx = get_joint_index(limb, "wrist")
        if sidx is not None:
            vec[sidx] += group_offsets["shoulder"]
        if widx is not None:
            vec[widx] += group_offsets["wrist"]

    elif limb == "R_arm":
        sidx = get_joint_index(limb, "shoulder")
        widx = get_joint_index(limb, "wrist")
        if sidx is not None:
            vec[sidx] += group_offsets["shoulder"]
        if widx is not None:
            vec[widx] -= group_offsets["wrist"]   # 方向不對改成 +=

    elif limb == "L_leg":
        tidx = get_joint_index(limb, "thigh")
        kidx = get_joint_index(limb, "knee")
        if tidx is not None:
            vec[tidx] += group_offsets["thigh"]
        if kidx is not None:
            vec[kidx] += group_offsets["knee"]

    elif limb == "R_leg":
        tidx = get_joint_index(limb, "thigh")
        kidx = get_joint_index(limb, "knee")
        if tidx is not None:
            vec[tidx] -= group_offsets["thigh"]   # 方向不對改成 +=
        if kidx is not None:
            vec[kidx] += group_offsets["knee"]

    return vec

def get_final_joint_values(limb):
    return (
        base_state[limb]["joints"]
        + individual_offsets[limb]["joints"]
        + get_group_offset_vector(limb)
    )

def get_final_extra_value(limb):
    return base_state[limb]["extra"] + individual_offsets[limb]["extra"]

def apply_all_ctrl():
    for limb, cfg in limb_cfg.items():
        joint_vals = get_final_joint_values(limb)

        for aid, val in zip(cfg["act_ids"], joint_vals):
            data.ctrl[aid] = clamp_to_ctrlrange(aid, val)

        if cfg["extra_aid"] is not None:
            extra_val = get_final_extra_value(limb)
            data.ctrl[cfg["extra_aid"]] = clamp_to_ctrlrange(cfg["extra_aid"], extra_val)

def apply_raw_state(state):
    for limb, cfg in limb_cfg.items():
        joint_vals = np.array(state[limb]["joints"], dtype=float)
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
        out[limb] = {"joints": joints, "extra": extra}
    return out

def move_state(viewer, state_a, state_b, steps=120, sleep=0.01):
    for i in range(steps):
        alpha = (i + 1) / steps
        s = interpolate_states(state_a, state_b, alpha)
        apply_raw_state(s)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(sleep)

def hold_state(viewer, state, steps=40, sleep=0.01):
    for _ in range(steps):
        apply_raw_state(state)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(sleep)

def reset_individual_offsets_active():
    limb = active_limb["name"]
    cfg = limb_cfg[limb]
    individual_offsets[limb]["joints"] = np.zeros(len(cfg["act_ids"]), dtype=float)
    individual_offsets[limb]["extra"] = 0.0

def reset_individual_offsets_all():
    for limb, cfg in limb_cfg.items():
        individual_offsets[limb]["joints"] = np.zeros(len(cfg["act_ids"]), dtype=float)
        individual_offsets[limb]["extra"] = 0.0

def reset_group_offsets():
    group_offsets["shoulder"] = 0.0
    group_offsets["wrist"] = 0.0
    group_offsets["thigh"] = 0.0
    group_offsets["knee"] = 0.0

# =========================================================
# auto climb sequence
# =========================================================
def build_pose_state(which_pose="start"):
    state = {}
    for limb in limb_cfg:
        extra_key = f"extra_{which_pose}"
        state[limb] = {
            "joints": pose_data[limb][which_pose].copy(),
            "extra": float(pose_data[limb].get(extra_key, 0.0)),
        }
    return state

def build_auto_climb_sequence():
    states = []

    s0 = build_pose_state("start")
    states.append(("All limbs at start", s0))

    s1 = copy_full_state(states[-1][1])
    s1["L_arm"]["joints"] = pose_data["L_arm"]["above"].copy()
    states.append(("Left arm above", s1))

    s2 = copy_full_state(states[-1][1])
    s2["L_arm"]["joints"] = pose_data["L_arm"]["approach"].copy()
    states.append(("Left arm approach", s2))

    s3 = copy_full_state(states[-1][1])
    s3["L_arm"]["joints"] = pose_data["L_arm"]["hook"].copy()
    s3["L_arm"]["extra"] = pose_data["L_arm"]["extra_hook"]
    states.append(("Left arm hook", s3))

    s4 = copy_full_state(states[-1][1])
    s4["R_arm"]["joints"] = pose_data["R_arm"]["above"].copy()
    states.append(("Right arm above", s4))

    s5 = copy_full_state(states[-1][1])
    s5["R_arm"]["joints"] = pose_data["R_arm"]["approach"].copy()
    states.append(("Right arm approach", s5))

    s6 = copy_full_state(states[-1][1])
    s6["R_arm"]["joints"] = pose_data["R_arm"]["hook"].copy()
    s6["R_arm"]["extra"] = pose_data["R_arm"]["extra_hook"]
    states.append(("Right arm hook", s6))

    s7 = copy_full_state(states[-1][1])
    s7["L_leg"]["joints"] = pose_data["L_leg"]["above"].copy()
    states.append(("Left leg above", s7))

    s8 = copy_full_state(states[-1][1])
    s8["L_leg"]["joints"] = pose_data["L_leg"]["approach"].copy()
    states.append(("Left leg approach", s8))

    s9 = copy_full_state(states[-1][1])
    s9["L_leg"]["joints"] = pose_data["L_leg"]["hook"].copy()
    states.append(("Left leg hook", s9))

    s10 = copy_full_state(states[-1][1])
    s10["R_leg"]["joints"] = pose_data["R_leg"]["above"].copy()
    states.append(("Right leg above", s10))

    s11 = copy_full_state(states[-1][1])
    s11["R_leg"]["joints"] = pose_data["R_leg"]["approach"].copy()
    states.append(("Right leg approach", s11))

    s12 = copy_full_state(states[-1][1])
    s12["R_leg"]["joints"] = pose_data["R_leg"]["hook"].copy()
    states.append(("Right leg hook / final standby", s12))

    return states

def run_auto_start_climb(viewer):
    print("Start auto climb-to-wall sequence...")
    sequence = build_auto_climb_sequence()

    apply_raw_state(sequence[0][1])
    for _ in range(120):
        apply_raw_state(sequence[0][1])
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

    for idx in range(len(sequence) - 1):
        label_a, state_a = sequence[idx]
        label_b, state_b = sequence[idx + 1]

        print(f"{idx:02d}: {label_a} -> {label_b}")

        if "above" in label_b:
            steps = 110
        elif "approach" in label_b:
            steps = 120
        else:
            steps = 140

        move_state(viewer, state_a, state_b, steps=steps, sleep=0.01)
        hold_state(viewer, state_b, steps=35, sleep=0.01)

    final_state = copy_full_state(sequence[-1][1])

    for limb in limb_cfg:
        base_state[limb]["joints"] = final_state[limb]["joints"].copy()
        base_state[limb]["extra"] = final_state[limb]["extra"]

    sequence_done["value"] = True
    print("Auto climb sequence finished.")
    print("Now sliders control offsets based on this climbed pose.")

# =========================================================
# playing saved steps
# =========================================================
def play_saved_step(viewer, step_name):
    if saved_steps[step_name] is None:
        print(f"{step_name} is empty.")
        return

    start_state = get_current_full_state()
    target_state = copy_full_state(saved_steps[step_name])

    print(f"Play {step_name} ...")
    move_state(viewer, start_state, target_state, steps=140, sleep=0.01)
    hold_state(viewer, target_state, steps=35, sleep=0.01)

    # 播放後，把這步變成新的 base pose，方便接下一步
    set_base_state_from_full_state(target_state)
    print(f"{step_name} finished. It is now the new base pose.")

def play_all_saved_steps(viewer):
    for step_name in step_names:
        if saved_steps[step_name] is not None:
            play_saved_step(viewer, step_name)

# =========================================================
# UI
# =========================================================
def slider_ui():
    global running

    root = tk.Tk()
    root.title("Auto Climb + Save Next Steps + Play Steps")
    root.geometry("1620x980")
    root.minsize(1320, 780)

    tk.Label(
        root,
        text="Auto Climb + Save Next Steps + Play Steps",
        font=("Arial", 16, "bold")
    ).pack(pady=8)

    tk.Label(
        root,
        text="先自動爬到牆上，再用 sliders 微調，然後把姿勢存成下一步攀爬動作並播放",
        font=("Arial", 10)
    ).pack(pady=2)

    content = tk.Frame(root)
    content.pack(fill="both", expand=True, padx=10, pady=8)

    left_panel = tk.LabelFrame(content, text="Active Limb / Status", padx=8, pady=8)
    left_panel.pack(side="left", fill="y", padx=6, pady=4)

    center_panel = tk.LabelFrame(content, text="Individual Offset Sliders", padx=8, pady=8)
    center_panel.pack(side="left", fill="both", expand=True, padx=6, pady=4)

    right_panel = tk.LabelFrame(content, text="Grouped Offsets / Saved Steps", padx=8, pady=8)
    right_panel.pack(side="right", fill="both", padx=6, pady=4)

    bottom_panel = tk.LabelFrame(root, text="Commands", padx=8, pady=8)
    bottom_panel.pack(fill="x", padx=10, pady=8)

    limb_var = tk.StringVar(value="L_arm")

    active_label = tk.Label(left_panel, text="", font=("Arial", 12, "bold"))
    active_label.pack(pady=6)

    tk.Label(left_panel, text="Choose active limb:", font=("Arial", 10)).pack(anchor="w", pady=(4, 6))

    for limb in ["L_arm", "R_arm", "L_leg", "R_leg"]:
        tk.Radiobutton(
            left_panel,
            text=limb,
            variable=limb_var,
            value=limb,
            command=lambda: rebuild_individual_sliders()
        ).pack(anchor="w", pady=3)

    status_box = tk.LabelFrame(left_panel, text="Status", padx=8, pady=8)
    status_box.pack(fill="x", pady=14)

    status_label = tk.Label(status_box, text="", font=("Consolas", 10), justify="left")
    status_label.pack(anchor="w")

    # center scroll
    center_canvas = tk.Canvas(center_panel, highlightthickness=0)
    center_scrollbar = tk.Scrollbar(center_panel, orient="vertical", command=center_canvas.yview)
    center_inner = tk.Frame(center_canvas)

    center_inner.bind(
        "<Configure>",
        lambda e: center_canvas.configure(scrollregion=center_canvas.bbox("all"))
    )

    center_canvas.create_window((0, 0), window=center_inner, anchor="nw")
    center_canvas.configure(yscrollcommand=center_scrollbar.set)
    center_canvas.pack(side="left", fill="both", expand=True)
    center_scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        try:
            center_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    center_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    current_values_box = tk.LabelFrame(center_inner, text="Current Values", padx=8, pady=8)
    current_values_box.pack(fill="x", padx=6, pady=6)

    value_label = tk.Label(current_values_box, text="", font=("Consolas", 11), justify="left")
    value_label.pack(anchor="w")

    slider_box = tk.LabelFrame(center_inner, text="Individual Offset Sliders", padx=8, pady=8)
    slider_box.pack(fill="both", expand=True, padx=6, pady=6)

    individual_sliders = []
    extra_slider = {"widget": None}

    # right panel
    group_rule_box = tk.LabelFrame(right_panel, text="Rule", padx=8, pady=8)
    group_rule_box.pack(fill="x", padx=6, pady=6)

    tk.Label(
        group_rule_box,
        text=(
            "final = base_pose + individual_offset + grouped_offset\n\n"
            "shoulder : L/R together\n"
            "wrist    : L += , R -=\n"
            "thigh    : L += , R -=\n"
            "knee     : L/R together\n\n"
            "播放某一步後，該姿勢會變成新的 base_pose"
        ),
        font=("Consolas", 10),
        justify="left"
    ).pack(anchor="w")

    group_slider_box = tk.LabelFrame(right_panel, text="Grouped Offset Sliders", padx=8, pady=8)
    group_slider_box.pack(fill="x", padx=6, pady=6)

    shoulder_group_slider = tk.Scale(group_slider_box, from_=-60, to=60, orient="horizontal", resolution=1, length=360, label="Both Shoulders Offset (deg)")
    shoulder_group_slider.pack(fill="x", pady=6)

    wrist_group_slider = tk.Scale(group_slider_box, from_=-60, to=60, orient="horizontal", resolution=1, length=360, label="Both Wrists Offset (deg)")
    wrist_group_slider.pack(fill="x", pady=6)

    thigh_group_slider = tk.Scale(group_slider_box, from_=-60, to=60, orient="horizontal", resolution=1, length=360, label="Both Thighs Offset (deg)")
    thigh_group_slider.pack(fill="x", pady=6)

    knee_group_slider = tk.Scale(group_slider_box, from_=-60, to=60, orient="horizontal", resolution=1, length=360, label="Both Knees Offset (deg)")
    knee_group_slider.pack(fill="x", pady=6)

    group_value_box = tk.LabelFrame(right_panel, text="Grouped Offset Values", padx=8, pady=8)
    group_value_box.pack(fill="x", padx=6, pady=6)

    group_value_label = tk.Label(group_value_box, text="", font=("Consolas", 10), justify="left")
    group_value_label.pack(anchor="w")

    saved_step_box = tk.LabelFrame(right_panel, text="Saved Step Summary", padx=8, pady=8)
    saved_step_box.pack(fill="both", expand=True, padx=6, pady=6)

    step_text = tk.Text(saved_step_box, width=48, height=22, font=("Consolas", 10))
    step_text.pack(side="left", fill="both", expand=True)

    step_scroll = tk.Scrollbar(saved_step_box, orient="vertical", command=step_text.yview)
    step_scroll.pack(side="right", fill="y")
    step_text.configure(yscrollcommand=step_scroll.set, state="disabled")

    # callbacks
    def on_individual_joint_slider(i, val):
        limb = active_limb["name"]
        individual_offsets[limb]["joints"][i] = deg_to_rad(float(val))

    def on_extra_slider(val):
        limb = active_limb["name"]
        individual_offsets[limb]["extra"] = float(val)

    def on_group_shoulder(val):
        group_offsets["shoulder"] = deg_to_rad(float(val))

    def on_group_wrist(val):
        group_offsets["wrist"] = deg_to_rad(float(val))

    def on_group_thigh(val):
        group_offsets["thigh"] = deg_to_rad(float(val))

    def on_group_knee(val):
        group_offsets["knee"] = deg_to_rad(float(val))

    shoulder_group_slider.config(command=on_group_shoulder)
    wrist_group_slider.config(command=on_group_wrist)
    thigh_group_slider.config(command=on_group_thigh)
    knee_group_slider.config(command=on_group_knee)

    def load_active_offsets_to_sliders():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        vals = individual_offsets[limb]["joints"].copy()
        ex = individual_offsets[limb]["extra"]

        for s, v in zip(individual_sliders, vals):
            s.set(rad_to_deg(v))

        if cfg["extra_aid"] is not None and extra_slider["widget"] is not None:
            extra_slider["widget"].set(ex)

    def rebuild_individual_sliders():
        for w in slider_box.winfo_children():
            w.destroy()

        individual_sliders.clear()
        extra_slider["widget"] = None

        limb = limb_var.get()
        active_limb["name"] = limb
        cfg = limb_cfg[limb]

        for i, (label_name, rg) in enumerate(zip(cfg["joint_labels"], cfg["deg_ranges"])):
            wrap = tk.Frame(slider_box)
            wrap.pack(fill="x", pady=4)

            s = tk.Scale(
                wrap,
                from_=rg[0], to=rg[1],
                resolution=1,
                orient="horizontal",
                length=700,
                label=f"{label_name} offset (deg)",
                command=lambda val, idx=i: on_individual_joint_slider(idx, val)
            )
            s.pack(fill="x")
            individual_sliders.append(s)

        if cfg["extra_aid"] is not None:
            wrap = tk.Frame(slider_box)
            wrap.pack(fill="x", pady=4)

            s = tk.Scale(
                wrap,
                from_=cfg["extra_range"][0], to=cfg["extra_range"][1],
                resolution=0.01,
                orient="horizontal",
                length=700,
                label="extra offset",
                command=on_extra_slider
            )
            s.pack(fill="x")
            extra_slider["widget"] = s

        load_active_offsets_to_sliders()

    def save_step(step_name):
        state = get_current_full_state()
        saved_steps[step_name] = copy_full_state(state)
        print(f"Saved {step_name}")

    def request_play_step(step_name):
        play_request["play_step"] = step_name

    def request_play_all():
        play_request["play_all"] = True

    def refresh_saved_step_text():
        lines = []
        for step_name in step_names:
            st = saved_steps[step_name]
            lines.append(f"{step_name}:")
            if st is None:
                lines.append("  None")
            else:
                for limb, cfg in limb_cfg.items():
                    degs = [round(rad_to_deg(v), 1) for v in st[limb]["joints"]]
                    lines.append(f"  {limb:6s}: {degs}")
            lines.append("")

        step_text.configure(state="normal")
        step_text.delete("1.0", tk.END)
        step_text.insert(tk.END, "\n".join(lines))
        step_text.configure(state="disabled")

    def refresh_labels():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        base_vals = base_state[limb]["joints"]
        ind_vals = individual_offsets[limb]["joints"]
        grp_vals = get_group_offset_vector(limb)
        final_vals = get_final_joint_values(limb)

        active_label.config(text=f"Active limb: {cfg['label']}")

        lines = []
        lines.append("[Base Pose]")
        for name, v in zip(cfg["joint_labels"], base_vals):
            lines.append(f"{name:10s} base  = {rad_to_deg(v):7.2f} deg")

        lines.append("")
        lines.append("[Individual Offset]")
        for name, v in zip(cfg["joint_labels"], ind_vals):
            lines.append(f"{name:10s} off   = {rad_to_deg(v):7.2f} deg")

        lines.append("")
        lines.append("[Grouped Offset]")
        for name, v in zip(cfg["joint_labels"], grp_vals):
            lines.append(f"{name:10s} grp   = {rad_to_deg(v):7.2f} deg")

        lines.append("")
        lines.append("[Final Control]")
        for name, v in zip(cfg["joint_labels"], final_vals):
            lines.append(f"{name:10s} final = {rad_to_deg(v):7.2f} deg")

        value_label.config(text="\n".join(lines))

        status_label.config(
            text=(
                f"sequence_done = {sequence_done['value']}\n"
                f"active_limb   = {limb}\n"
                f"base_source   = climbed pose / last played step"
            )
        )

        group_value_label.config(
            text=(
                f"shoulder = {rad_to_deg(group_offsets['shoulder']):7.2f} deg\n"
                f"wrist    = {rad_to_deg(group_offsets['wrist']):7.2f} deg\n"
                f"thigh    = {rad_to_deg(group_offsets['thigh']):7.2f} deg\n"
                f"knee     = {rad_to_deg(group_offsets['knee']):7.2f} deg"
            )
        )

        refresh_saved_step_text()

        if running:
            root.after(120, refresh_labels)

    def reset_active_offset_cmd():
        reset_individual_offsets_active()
        load_active_offsets_to_sliders()

    def reset_all_offset_cmd():
        reset_individual_offsets_all()
        load_active_offsets_to_sliders()

    def reset_group_cmd():
        reset_group_offsets()
        shoulder_group_slider.set(0)
        wrist_group_slider.set(0)
        thigh_group_slider.set(0)
        knee_group_slider.set(0)

    def reset_everything_cmd():
        reset_individual_offsets_all()
        reset_group_offsets()
        load_active_offsets_to_sliders()
        shoulder_group_slider.set(0)
        wrist_group_slider.set(0)
        thigh_group_slider.set(0)
        knee_group_slider.set(0)

    # bottom buttons
    row1 = tk.Frame(bottom_panel)
    row1.pack(fill="x", pady=4)

    tk.Button(row1, text="Reset Active Offset", command=reset_active_offset_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset All Offset", command=reset_all_offset_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset Group Offset", command=reset_group_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset Everything", command=reset_everything_cmd, width=18).pack(side="left", padx=6, pady=4)

    row2 = tk.Frame(bottom_panel)
    row2.pack(fill="x", pady=4)

    tk.Button(row2, text="Save Step 1", command=lambda: save_step("step1"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row2, text="Save Step 2", command=lambda: save_step("step2"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row2, text="Save Step 3", command=lambda: save_step("step3"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row2, text="Save Step 4", command=lambda: save_step("step4"), width=16).pack(side="left", padx=6, pady=4)

    row3 = tk.Frame(bottom_panel)
    row3.pack(fill="x", pady=4)

    tk.Button(row3, text="Play Step 1", command=lambda: request_play_step("step1"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row3, text="Play Step 2", command=lambda: request_play_step("step2"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row3, text="Play Step 3", command=lambda: request_play_step("step3"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row3, text="Play Step 4", command=lambda: request_play_step("step4"), width=16).pack(side="left", padx=6, pady=4)
    tk.Button(row3, text="Play All Saved Steps", command=request_play_all, width=18).pack(side="left", padx=6, pady=4)

    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    rebuild_individual_sliders()
    refresh_labels()
    root.mainloop()

# =========================================================
# start UI
# =========================================================
ui_thread = threading.Thread(target=slider_ui, daemon=True)
ui_thread.start()

# =========================================================
# main viewer loop
# =========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started.")
    print("Robot will first auto-complete the climb-to-wall start/standby pose.")
    print("After that, you can tune joints, save next steps, and play them.\n")

    run_auto_start_climb(viewer)

    while running and viewer.is_running():
        step_to_play = None
        do_play_all = False

        with lock:
            step_to_play = play_request["play_step"]
            do_play_all = play_request["play_all"]
            play_request["play_step"] = None
            play_request["play_all"] = False

        if step_to_play is not None:
            play_saved_step(viewer, step_to_play)

        elif do_play_all:
            play_all_saved_steps(viewer)

        apply_all_ctrl()
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("Exit.")
