import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import tkinter as tk

XML_PATH = "models/XS-Robot(Alex).xml"

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
        "site_name": "finger_L_grip",
        "target_name": "hold2_target",
        "deg_ranges": [(-120, 120), (-120, 120), (-120, 120)],
        "extra_range": (0.0, 0.4),
    },
    "R_arm": {
        "label": "Right Arm",
        "act_names": ["act_shoulder_R", "act_elbow_R", "act_wrist_R"],
        "extra_name": "act_finger_R",
        "joint_labels": ["shoulder", "elbow", "wrist"],
        "site_name": "finger_R_grip",
        "target_name": "hold4_target_R",   # 若名稱不同再改
        "deg_ranges": [(-120, 120), (-120, 120), (-120, 120)],
        "extra_range": (0.0, 0.4),
    },
    "L_leg": {
        "label": "Left Leg",
        "act_names": ["act_hiproll_L", "act_hipyaw_L", "act_thigh_L", "act_knee_L", "act_ankle_L"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "site_name": "sole_L_step_site",
        "target_name": "hold4_target",
        "deg_ranges": [
            (0, 90),
            (-90, 0),
            (-90, 0),
            (-90, 0),
            (-90, 90),
        ],
        "extra_range": None,
    },
    "R_leg": {
        "label": "Right Leg",
        "act_names": ["act_hiproll_R", "act_hipyaw_R", "act_thigh_R", "act_knee_R", "act_ankle_R"],
        "extra_name": None,
        "joint_labels": ["hiproll", "hipyaw", "thigh", "knee", "ankle"],
        "site_name": "sole_R_step_site",
        "target_name": "hold3_target",
        "deg_ranges": [
            (0, 90),
            (0, 90),
            (0, 90),
            (-90, 0),
            (-90, 90),
        ],
        "extra_range": None,
    },
}

# =========================================================
# ids
# =========================================================
for limb in limb_cfg:
    cfg = limb_cfg[limb]
    cfg["act_ids"] = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in cfg["act_names"]
    ]

    if cfg["extra_name"] is not None:
        cfg["extra_aid"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg["extra_name"])
    else:
        cfg["extra_aid"] = None

    cfg["site_sid"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["site_name"])

    try:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["target_name"])
        cfg["target_sid"] = sid if sid >= 0 else None
    except:
        cfg["target_sid"] = None

# =========================================================
# shared state
# =========================================================
running = True
lock = threading.Lock()

active_limb = {"name": "L_arm"}

ctrl_values = {}
for limb in limb_cfg:
    n = len(limb_cfg[limb]["act_ids"])
    ctrl_values[limb] = {
        "joints": [0.0] * n,
        "extra": 0.0,
    }

pose_names = ["start", "above", "approach", "hook"]

saved_poses = {
    limb: {pname: None for pname in pose_names}
    for limb in limb_cfg
}

distance_text = {"value": "distance = ---"}
play_active_request = {"go": False}
play_all_request = {"go": False}

# =========================================================
# tools
# =========================================================
def rad_to_deg(x):
    return np.rad2deg(x)

def deg_to_rad(x):
    return np.deg2rad(x)

def site_pos(sid):
    return data.site_xpos[sid].copy()

def apply_all_ctrl():
    for limb, cfg in limb_cfg.items():
        for aid, val in zip(cfg["act_ids"], ctrl_values[limb]["joints"]):
            data.ctrl[aid] = float(val)

        if cfg["extra_aid"] is not None:
            data.ctrl[cfg["extra_aid"]] = float(ctrl_values[limb]["extra"])

def update_distance_text():
    limb = active_limb["name"]
    cfg = limb_cfg[limb]

    mujoco.mj_forward(model, data)
    grip = site_pos(cfg["site_sid"])

    if cfg["target_sid"] is None:
        distance_text["value"] = (
            f"{cfg['label']}\n"
            f"site = [{grip[0]:.4f}, {grip[1]:.4f}, {grip[2]:.4f}]\n"
            f"target = None"
        )
        return

    target = site_pos(cfg["target_sid"])
    dist = np.linalg.norm(target - grip)

    distance_text["value"] = (
        f"{cfg['label']}\n"
        f"distance = {dist:.6f}\n"
        f"site     = [{grip[0]:.4f}, {grip[1]:.4f}, {grip[2]:.4f}]\n"
        f"target   = [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
    )

def current_pose(limb):
    return {
        "joints": ctrl_values[limb]["joints"].copy(),
        "extra": ctrl_values[limb]["extra"],
    }

def move_pose_runtime(viewer, limb, target_pose, steps=100, sleep=0.01):
    cfg = limb_cfg[limb]
    act_ids = cfg["act_ids"]

    start_pose = np.array([data.ctrl[aid] for aid in act_ids], dtype=float)
    target_pose = np.array(target_pose, dtype=float)

    for i in range(steps):
        if not running:
            return
        r = (i + 1) / steps
        pose = start_pose * (1 - r) + target_pose * r

        for aid, val in zip(act_ids, pose):
            data.ctrl[aid] = float(val)

        with lock:
            ctrl_values[limb]["joints"] = pose.tolist()

        mujoco.mj_step(model, data)
        update_distance_text()
        viewer.sync()
        time.sleep(sleep)

def move_extra_runtime(viewer, limb, target_val=0.0, steps=40, sleep=0.02):
    cfg = limb_cfg[limb]
    if cfg["extra_aid"] is None:
        return

    aid = cfg["extra_aid"]
    start_val = float(data.ctrl[aid])

    for i in range(steps):
        if not running:
            return
        r = (i + 1) / steps
        val = start_val * (1 - r) + target_val * r
        data.ctrl[aid] = float(val)

        with lock:
            ctrl_values[limb]["extra"] = float(val)

        mujoco.mj_step(model, data)
        update_distance_text()
        viewer.sync()
        time.sleep(sleep)

def play_limb_path(viewer, limb):
    pstart = saved_poses[limb]["start"]
    pabove = saved_poses[limb]["above"]
    papproach = saved_poses[limb]["approach"]
    phook = saved_poses[limb]["hook"]

    if None in [pstart, pabove, papproach, phook]:
        print(f"{limb} path incomplete. Please save start/above/approach/hook first.")
        return

    print(f"Play {limb_cfg[limb]['label']} path")

    move_pose_runtime(viewer, limb, pstart["joints"], steps=80, sleep=0.01)
    if limb_cfg[limb]["extra_aid"] is not None:
        move_extra_runtime(viewer, limb, pstart["extra"], steps=20, sleep=0.01)

    move_pose_runtime(viewer, limb, pabove["joints"], steps=100, sleep=0.01)
    if limb_cfg[limb]["extra_aid"] is not None:
        move_extra_runtime(viewer, limb, pabove["extra"], steps=20, sleep=0.01)

    move_pose_runtime(viewer, limb, papproach["joints"], steps=100, sleep=0.01)
    if limb_cfg[limb]["extra_aid"] is not None:
        move_extra_runtime(viewer, limb, papproach["extra"], steps=20, sleep=0.01)

    move_pose_runtime(viewer, limb, phook["joints"], steps=120, sleep=0.01)
    if limb_cfg[limb]["extra_aid"] is not None:
        move_extra_runtime(viewer, limb, phook["extra"], steps=30, sleep=0.02)

def play_all_paths(viewer):
    seq = ["L_arm", "R_arm", "L_leg", "R_leg"]
    for limb in seq:
        play_limb_path(viewer, limb)

def print_saved_poses():
    print("\n===== Saved poses (degree) =====")
    for limb, cfg in limb_cfg.items():
        print(f"\n{limb} / {cfg['label']}")
        for pname in pose_names:
            p = saved_poses[limb][pname]
            if p is None:
                print(f"  {pname:8s}: None")
            else:
                degs = [float(rad_to_deg(v)) for v in p["joints"]]
                print(f"  {pname:8s}: {[round(v, 1) for v in degs]}")
                if cfg["extra_aid"] is not None:
                    print(f"             extra={p['extra']:.3f}")
    print("================================\n")

# =========================================================
# ui
# =========================================================
def slider_ui():
    global running

    root = tk.Tk()
    root.title("Advanced 4-Limb Climbing Pose Editor")
    root.geometry("1300x860")
    root.minsize(1100, 760)

    # =====================================================
    # main layout
    # =====================================================
    top_title = tk.Label(
        root,
        text="Advanced 4-Limb Path Pose Editor",
        font=("Arial", 16, "bold")
    )
    top_title.pack(pady=8)

    subtitle = tk.Label(
        root,
        text="Select one limb, adjust sliders, save start / above / approach / hook, then test path",
        font=("Arial", 10)
    )
    subtitle.pack(pady=2)

    content = tk.Frame(root)
    content.pack(fill="both", expand=True, padx=10, pady=8)

    left_panel = tk.LabelFrame(content, text="Limb Selection", padx=8, pady=8)
    left_panel.pack(side="left", fill="y", padx=6, pady=4)

    center_panel = tk.LabelFrame(content, text="Joint Adjustment", padx=8, pady=8)
    center_panel.pack(side="left", fill="both", expand=True, padx=6, pady=4)

    right_panel = tk.LabelFrame(content, text="Status / Saved Poses", padx=8, pady=8)
    right_panel.pack(side="right", fill="both", padx=6, pady=4)

    bottom_panel = tk.LabelFrame(root, text="Controls", padx=8, pady=8)
    bottom_panel.pack(fill="x", padx=10, pady=8)

    # =====================================================
    # left panel - limb selection
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
            command=lambda: rebuild_sliders()
        ).pack(anchor="w", pady=3)

    # =====================================================
    # center panel - scrollable sliders
    # =====================================================
    slider_canvas = tk.Canvas(center_panel, highlightthickness=0)
    slider_scrollbar = tk.Scrollbar(center_panel, orient="vertical", command=slider_canvas.yview)
    slider_inner = tk.Frame(slider_canvas)

    slider_inner.bind(
        "<Configure>",
        lambda e: slider_canvas.configure(scrollregion=slider_canvas.bbox("all"))
    )

    slider_canvas.create_window((0, 0), window=slider_inner, anchor="nw")
    slider_canvas.configure(yscrollcommand=slider_scrollbar.set)

    slider_canvas.pack(side="left", fill="both", expand=True)
    slider_scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(event):
        try:
            slider_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except:
            pass

    slider_canvas.bind_all("<MouseWheel>", _on_mousewheel)

    sliders = []
    extra_slider = {"widget": None}

    current_values_box = tk.LabelFrame(slider_inner, text="Current Joint Values", padx=8, pady=8)
    current_values_box.pack(fill="x", padx=6, pady=6)

    value_label = tk.Label(current_values_box, text="", font=("Consolas", 11), justify="left")
    value_label.pack(anchor="w")

    slider_box = tk.LabelFrame(slider_inner, text="Sliders", padx=8, pady=8)
    slider_box.pack(fill="both", expand=True, padx=6, pady=6)

    # =====================================================
    # right panel
    # =====================================================
    distance_box = tk.LabelFrame(right_panel, text="Distance / Site Info", padx=8, pady=8)
    distance_box.pack(fill="x", padx=6, pady=6)

    distance_label = tk.Label(distance_box, text="distance = ---", font=("Consolas", 10), justify="left")
    distance_label.pack(anchor="w")

    pose_box = tk.LabelFrame(right_panel, text="Saved Poses", padx=8, pady=8)
    pose_box.pack(fill="both", expand=True, padx=6, pady=6)

    pose_text = tk.Text(pose_box, width=48, height=34, font=("Consolas", 10))
    pose_text.pack(side="left", fill="both", expand=True)

    pose_scroll = tk.Scrollbar(pose_box, orient="vertical", command=pose_text.yview)
    pose_scroll.pack(side="right", fill="y")
    pose_text.configure(yscrollcommand=pose_scroll.set, state="disabled")

    # =====================================================
    # callbacks
    # =====================================================
    def on_joint_slider(i, val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["joints"][i] = deg_to_rad(float(val))

    def on_extra_slider(val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["extra"] = float(val)

    def load_current_ctrl_to_sliders():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            vals = ctrl_values[limb]["joints"].copy()
            ex = ctrl_values[limb]["extra"]

        for s, v in zip(sliders, vals):
            s.set(rad_to_deg(v))

        if cfg["extra_aid"] is not None and extra_slider["widget"] is not None:
            extra_slider["widget"].set(ex)

    def rebuild_sliders():
        for w in slider_box.winfo_children():
            w.destroy()

        sliders.clear()
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
                length=620,
                label=f"{label_name} (deg)",
                command=lambda val, idx=i: on_joint_slider(idx, val)
            )
            s.pack(fill="x")
            sliders.append(s)

        if cfg["extra_aid"] is not None:
            wrap = tk.Frame(slider_box)
            wrap.pack(fill="x", pady=4)

            s = tk.Scale(
                wrap,
                from_=cfg["extra_range"][0], to=cfg["extra_range"][1],
                resolution=0.01,
                orient="horizontal",
                length=620,
                label="extra",
                command=on_extra_slider
            )
            s.pack(fill="x")
            extra_slider["widget"] = s

        load_current_ctrl_to_sliders()

    def refresh_saved_pose_text():
        lines = []
        for limb_name, c in limb_cfg.items():
            lines.append(f"{limb_name} / {c['label']}")
            for pname in pose_names:
                p = saved_poses[limb_name][pname]
                if p is None:
                    lines.append(f"  {pname:8s}: None")
                else:
                    degs = [rad_to_deg(v) for v in p["joints"]]
                    joint_txt = ", ".join([f"{d:6.1f}" for d in degs])
                    if c["extra_aid"] is not None:
                        lines.append(f"  {pname:8s}: [{joint_txt}] extra={p['extra']:.2f}")
                    else:
                        lines.append(f"  {pname:8s}: [{joint_txt}]")
            lines.append("")

        pose_text.configure(state="normal")
        pose_text.delete("1.0", tk.END)
        pose_text.insert(tk.END, "\n".join(lines))
        pose_text.configure(state="disabled")

    def refresh_labels():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            vals = ctrl_values[limb]["joints"].copy()
            ex = ctrl_values[limb]["extra"]

        active_label.config(text=f"Active limb: {cfg['label']}")

        lines = []
        for name, v in zip(cfg["joint_labels"], vals):
            lines.append(f"{name:10s} = {rad_to_deg(v):7.2f} deg")

        if cfg["extra_aid"] is not None:
            lines.append(f"{'extra':10s} = {ex:.4f}")

        value_label.config(text="\n".join(lines))
        distance_label.config(text=distance_text["value"])
        refresh_saved_pose_text()

        if running:
            root.after(120, refresh_labels)

    def save_pose(pname):
        limb = active_limb["name"]
        with lock:
            saved_poses[limb][pname] = current_pose(limb)
        print(f"Saved {pname} pose for {limb}")

    def play_active_path():
        with lock:
            play_active_request["go"] = True

    def play_all():
        with lock:
            play_all_request["go"] = True

    def reset_active():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            ctrl_values[limb]["joints"] = [0.0] * len(cfg["act_ids"])
            ctrl_values[limb]["extra"] = 0.0

        load_current_ctrl_to_sliders()

    def reset_all():
        with lock:
            for limb_name, cfg in limb_cfg.items():
                ctrl_values[limb_name]["joints"] = [0.0] * len(cfg["act_ids"])
                ctrl_values[limb_name]["extra"] = 0.0

        load_current_ctrl_to_sliders()

    # =====================================================
    # bottom controls
    # =====================================================
    pose_btn_frame = tk.Frame(bottom_panel)
    pose_btn_frame.pack(fill="x", pady=4)

    tk.Button(pose_btn_frame, text="Set Start", command=lambda: save_pose("start"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Above", command=lambda: save_pose("above"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Approach", command=lambda: save_pose("approach"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Hook", command=lambda: save_pose("hook"), width=14).pack(side="left", padx=6, pady=4)

    cmd_btn_frame = tk.Frame(bottom_panel)
    cmd_btn_frame.pack(fill="x", pady=4)

    tk.Button(cmd_btn_frame, text="Play Active Path", command=play_active_path, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Play All Paths", command=play_all, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Print Poses", command=print_saved_poses, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Reload Sliders", command=load_current_ctrl_to_sliders, width=16).pack(side="left", padx=6, pady=4)

    reset_btn_frame = tk.Frame(bottom_panel)
    reset_btn_frame.pack(fill="x", pady=4)

    tk.Button(reset_btn_frame, text="Reset Active", command=reset_active, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(reset_btn_frame, text="Reset All", command=reset_all, width=16).pack(side="left", padx=6, pady=4)

    # =====================================================
    # close
    # =====================================================
    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    rebuild_sliders()
    refresh_labels()
    root.mainloop()

# =========================================================
# start ui thread
# =========================================================
ui_thread = threading.Thread(target=slider_ui, daemon=True)
ui_thread.start()

with lock:
    apply_all_ctrl()

for _ in range(100):
    mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started.")
    print("Select one limb in UI.")
    print("Adjust sliders, then save start / above / approach / hook.")
    print("You can play the active limb path or all limb paths.")
    print("Press Enter in terminal to exit.\n")

    def wait_for_enter():
        global running
        input()
        running = False

    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    while running:
        do_play_active = False
        do_play_all = False

        with lock:
            apply_all_ctrl()
            do_play_active = play_active_request["go"]
            do_play_all = play_all_request["go"]

        if do_play_active:
            with lock:
                play_active_request["go"] = False
                limb = active_limb["name"]
            play_limb_path(viewer, limb)

        if do_play_all:
            with lock:
                play_all_request["go"] = False
            play_all_paths(viewer)

        mujoco.mj_step(model, data)

        with lock:
            update_distance_text()

        viewer.sync()
        time.sleep(0.01)

print("Exit.")
