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
# limb config
# =========================================================
limb_cfg = {
    "L_arm": {
        "label": "Left Arm",
        "act_names": ["act_shoulder_L", "act_elbow_L", "act_wrist_L"],
        "extra_name": "act_finger_L",
        "joint_labels": ["shoulder", "elbow", "wrist"],
        "deg_ranges": [(-120, 120), (-120, 120), (-120, 120)],
        "extra_range": (0.0, 0.4),
    },
    "R_arm": {
        "label": "Right Arm",
        "act_names": ["act_shoulder_R", "act_elbow_R", "act_wrist_R"],
        "extra_name": "act_finger_R",
        "joint_labels": ["shoulder", "elbow", "wrist"],
        "deg_ranges": [(-120, 120), (-120, 120), (-120, 120)],
        "extra_range": (0.0, 0.4),
    },
    "L_leg": {
        "label": "Left Leg",
        "act_names": ["act_hiproll_L", "act_hipyaw_L", "act_thigh_L", "act_knee_L", "act_ankle_L"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "deg_ranges": [
            (-90, 90),
            (-90, 90),
            (-120, 120),
            (-120, 120),
            (-90, 90),
        ],
        "extra_range": None,
    },
    "R_leg": {
        "label": "Right Leg",
        "act_names": ["act_hiproll_R", "act_hipyaw_R", "act_thigh_R", "act_knee_R", "act_ankle_R"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "deg_ranges": [
            (-90, 90),
            (-90, 90),
            (-120, 120),
            (-120, 120),
            (-90, 90),
        ],
        "extra_range": None,
    },
}

# =========================================================
# actuator ids
# =========================================================
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
# shared state
# =========================================================
running = True
lock = threading.Lock()

active_limb = {"name": "L_arm"}

# base ctrl values = each limb individual sliders control
ctrl_values = {}
for limb in limb_cfg:
    n = len(limb_cfg[limb]["act_ids"])
    ctrl_values[limb] = {
        "joints": [0.0] * n,
        "extra": 0.0,
    }

# grouped sync offsets (in radians)
group_offsets = {
    "shoulder": 0.0,
    "wrist": 0.0,
    "thigh": 0.0,
    "knee": 0.0,
}

# request flags
reset_group_request = {"go": False}

# =========================================================
# helpers
# =========================================================
def rad_to_deg(x):
    return float(np.rad2deg(x))

def deg_to_rad(x):
    return float(np.deg2rad(x))

def clamp_to_ctrlrange(aid, val):
    low, high = model.actuator_ctrlrange[aid]
    return float(np.clip(val, low, high))

def get_joint_index(limb, joint_name):
    cfg = limb_cfg[limb]
    for i, name in enumerate(cfg["joint_labels"]):
        if name == joint_name:
            return i
    return None

def get_effective_joint_values(limb):
    """
    base individual joint values + grouped offsets
    """
    vals = np.array(ctrl_values[limb]["joints"], dtype=float).copy()

    if limb == "L_arm":
        sidx = get_joint_index(limb, "shoulder")
        widx = get_joint_index(limb, "wrist")
        if sidx is not None:
            vals[sidx] += group_offsets["shoulder"]
        if widx is not None:
            vals[widx] += group_offsets["wrist"]

    elif limb == "R_arm":
        sidx = get_joint_index(limb, "shoulder")
        widx = get_joint_index(limb, "wrist")
        if sidx is not None:
            vals[sidx] += group_offsets["shoulder"]
        if widx is not None:
            vals[widx] -= group_offsets["wrist"]
            # 若你想左右 wrist 同方向，就改成 += group_offsets["wrist"]

    elif limb == "L_leg":
        tidx = get_joint_index(limb, "thigh")
        kidx = get_joint_index(limb, "knee")
        if tidx is not None:
            vals[tidx] += group_offsets["thigh"]
        if kidx is not None:
            vals[kidx] += group_offsets["knee"]

    elif limb == "R_leg":
        tidx = get_joint_index(limb, "thigh")
        kidx = get_joint_index(limb, "knee")
        if tidx is not None:
            vals[tidx] -= group_offsets["thigh"]
            # 若你想左右 thigh 同方向，就改成 += group_offsets["thigh"]
        if kidx is not None:
            vals[kidx] += group_offsets["knee"]

    return vals

def apply_all_ctrl():
    for limb, cfg in limb_cfg.items():
        joint_vals = get_effective_joint_values(limb)

        for aid, val in zip(cfg["act_ids"], joint_vals):
            data.ctrl[aid] = clamp_to_ctrlrange(aid, val)

        if cfg["extra_aid"] is not None:
            data.ctrl[cfg["extra_aid"]] = clamp_to_ctrlrange(
                cfg["extra_aid"], ctrl_values[limb]["extra"]
            )

def reset_active_limb():
    limb = active_limb["name"]
    cfg = limb_cfg[limb]
    with lock:
        ctrl_values[limb]["joints"] = [0.0] * len(cfg["act_ids"])
        ctrl_values[limb]["extra"] = 0.0

def reset_all_limbs():
    with lock:
        for limb, cfg in limb_cfg.items():
            ctrl_values[limb]["joints"] = [0.0] * len(cfg["act_ids"])
            ctrl_values[limb]["extra"] = 0.0

def reset_group_offsets():
    with lock:
        group_offsets["shoulder"] = 0.0
        group_offsets["wrist"] = 0.0
        group_offsets["thigh"] = 0.0
        group_offsets["knee"] = 0.0

# =========================================================
# ui
# =========================================================
def slider_ui():
    global running

    root = tk.Tk()
    root.title("Joint Control + Group Sync Control")
    root.geometry("1480x900")
    root.minsize(1250, 760)

    # -----------------------------
    # top title
    # -----------------------------
    title = tk.Label(
        root,
        text="MuJoCo Individual Joint Control + Group Sync Control",
        font=("Arial", 16, "bold")
    )
    title.pack(pady=8)

    subtitle = tk.Label(
        root,
        text="Left: choose active limb | Center: individual joint sliders | Right: grouped shoulder/wrist/thigh/knee control",
        font=("Arial", 10)
    )
    subtitle.pack(pady=2)

    # -----------------------------
    # main layout
    # -----------------------------
    content = tk.Frame(root)
    content.pack(fill="both", expand=True, padx=10, pady=8)

    left_panel = tk.LabelFrame(content, text="Active Limb", padx=8, pady=8)
    left_panel.pack(side="left", fill="y", padx=6, pady=4)

    center_panel = tk.LabelFrame(content, text="Individual Joint Adjustment", padx=8, pady=8)
    center_panel.pack(side="left", fill="both", expand=True, padx=6, pady=4)

    right_panel = tk.LabelFrame(content, text="Grouped Sync Control", padx=8, pady=8)
    right_panel.pack(side="right", fill="both", padx=6, pady=4)

    bottom_panel = tk.LabelFrame(root, text="Commands", padx=8, pady=8)
    bottom_panel.pack(fill="x", padx=10, pady=8)

    # =====================================================
    # left panel
    # =====================================================
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

    # =====================================================
    # center panel (scrollable sliders)
    # =====================================================
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

    slider_box = tk.LabelFrame(center_inner, text="Individual Joint Sliders", padx=8, pady=8)
    slider_box.pack(fill="both", expand=True, padx=6, pady=6)

    individual_sliders = []
    extra_slider = {"widget": None}

    # =====================================================
    # right panel (group sliders)
    # =====================================================
    group_info_box = tk.LabelFrame(right_panel, text="Grouped Rule", padx=8, pady=8)
    group_info_box.pack(fill="x", padx=6, pady=6)

    group_info = tk.Label(
        group_info_box,
        text=(
            "Shoulder: L/R together\n"
            "Wrist   : L += , R -=\n"
            "Thigh   : L += , R -=\n"
            "Knee    : L/R together\n\n"
            "如果方向不對，可改 apply function 裡的 +/-"
        ),
        font=("Consolas", 10),
        justify="left"
    )
    group_info.pack(anchor="w")

    group_slider_box = tk.LabelFrame(right_panel, text="Grouped Sliders", padx=8, pady=8)
    group_slider_box.pack(fill="x", padx=6, pady=6)

    shoulder_group_slider = tk.Scale(
        group_slider_box,
        from_=-120, to=120,
        orient="horizontal",
        resolution=1,
        length=360,
        label="Both Shoulders (deg)"
    )
    shoulder_group_slider.pack(fill="x", pady=6)

    wrist_group_slider = tk.Scale(
        group_slider_box,
        from_=-120, to=120,
        orient="horizontal",
        resolution=1,
        length=360,
        label="Both Wrists (deg)"
    )
    wrist_group_slider.pack(fill="x", pady=6)

    thigh_group_slider = tk.Scale(
        group_slider_box,
        from_=-120, to=120,
        orient="horizontal",
        resolution=1,
        length=360,
        label="Both Thighs (deg)"
    )
    thigh_group_slider.pack(fill="x", pady=6)

    knee_group_slider = tk.Scale(
        group_slider_box,
        from_=-120, to=120,
        orient="horizontal",
        resolution=1,
        length=360,
        label="Both Knees (deg)"
    )
    knee_group_slider.pack(fill="x", pady=6)

    group_value_box = tk.LabelFrame(right_panel, text="Grouped Offset Values", padx=8, pady=8)
    group_value_box.pack(fill="x", padx=6, pady=6)

    group_value_label = tk.Label(group_value_box, text="", font=("Consolas", 10), justify="left")
    group_value_label.pack(anchor="w")

    # =====================================================
    # callbacks
    # =====================================================
    def on_individual_joint_slider(i, val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["joints"][i] = deg_to_rad(float(val))

    def on_extra_slider(val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["extra"] = float(val)

    def on_group_shoulder(val):
        with lock:
            group_offsets["shoulder"] = deg_to_rad(float(val))

    def on_group_wrist(val):
        with lock:
            group_offsets["wrist"] = deg_to_rad(float(val))

    def on_group_thigh(val):
        with lock:
            group_offsets["thigh"] = deg_to_rad(float(val))

    def on_group_knee(val):
        with lock:
            group_offsets["knee"] = deg_to_rad(float(val))

    shoulder_group_slider.config(command=on_group_shoulder)
    wrist_group_slider.config(command=on_group_wrist)
    thigh_group_slider.config(command=on_group_thigh)
    knee_group_slider.config(command=on_group_knee)

    def load_active_limb_to_sliders():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            vals = ctrl_values[limb]["joints"].copy()
            ex = ctrl_values[limb]["extra"]

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
                length=680,
                label=f"{label_name} (deg)",
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
                length=680,
                label="extra",
                command=on_extra_slider
            )
            s.pack(fill="x")
            extra_slider["widget"] = s

        load_active_limb_to_sliders()

    def refresh_labels():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            base_vals = ctrl_values[limb]["joints"].copy()
            eff_vals = get_effective_joint_values(limb)
            ex = ctrl_values[limb]["extra"]

            g_shoulder = rad_to_deg(group_offsets["shoulder"])
            g_wrist = rad_to_deg(group_offsets["wrist"])
            g_thigh = rad_to_deg(group_offsets["thigh"])
            g_knee = rad_to_deg(group_offsets["knee"])

        active_label.config(text=f"Active limb: {cfg['label']}")

        lines = []
        lines.append("[Base Individual Values]")
        for name, v in zip(cfg["joint_labels"], base_vals):
            lines.append(f"{name:10s} = {rad_to_deg(v):7.2f} deg")

        if cfg["extra_aid"] is not None:
            lines.append(f"{'extra':10s} = {ex:.4f}")

        lines.append("")
        lines.append("[Effective Values After Group Offset]")
        for name, v in zip(cfg["joint_labels"], eff_vals):
            lines.append(f"{name:10s} = {rad_to_deg(v):7.2f} deg")

        value_label.config(text="\n".join(lines))

        group_value_label.config(
            text=(
                f"shoulder = {g_shoulder:7.2f} deg\n"
                f"wrist    = {g_wrist:7.2f} deg\n"
                f"thigh    = {g_thigh:7.2f} deg\n"
                f"knee     = {g_knee:7.2f} deg"
            )
        )

        if running:
            root.after(100, refresh_labels)

    def reset_active_cmd():
        reset_active_limb()
        load_active_limb_to_sliders()

    def reset_all_cmd():
        reset_all_limbs()
        load_active_limb_to_sliders()

    def reset_group_cmd():
        reset_group_offsets()
        shoulder_group_slider.set(0)
        wrist_group_slider.set(0)
        thigh_group_slider.set(0)
        knee_group_slider.set(0)

    def reset_everything_cmd():
        reset_all_limbs()
        reset_group_offsets()
        load_active_limb_to_sliders()
        shoulder_group_slider.set(0)
        wrist_group_slider.set(0)
        thigh_group_slider.set(0)
        knee_group_slider.set(0)

    # =====================================================
    # bottom buttons
    # =====================================================
    row1 = tk.Frame(bottom_panel)
    row1.pack(fill="x", pady=4)

    tk.Button(row1, text="Reset Active Limb", command=reset_active_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset All Limbs", command=reset_all_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset Group Offsets", command=reset_group_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reset Everything", command=reset_everything_cmd, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(row1, text="Reload Active Sliders", command=load_active_limb_to_sliders, width=18).pack(side="left", padx=6, pady=4)

    # =====================================================
    # close
    # =====================================================
    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    rebuild_individual_sliders()
    refresh_labels()
    root.mainloop()

# =========================================================
# start UI thread
# =========================================================
ui_thread = threading.Thread(target=slider_ui, daemon=True)
ui_thread.start()

# =========================================================
# viewer loop
# =========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started.")
    print("You can:")
    print("1. Select one limb and adjust individual joints")
    print("2. Adjust grouped shoulder / wrist / thigh / knee together")
    print("3. Reset active limb / all limbs / grouped offsets")
    print("Close Tk window to exit.\n")

    while running and viewer.is_running():
        with lock:
            apply_all_ctrl()

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)

print("Exit.")
