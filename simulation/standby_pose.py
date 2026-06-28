import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import tkinter as tk

# =========================================================
# XS-Robot_V2 Standby / Pose Editor
# Put this .py file in the SAME folder as XS-Robot_V2.xml
# Folder example:
# small_robot_V2/
#   XS-Robot_V2.xml
#   meshes/
#   standby_pose_editor_v2.py
# =========================================================

XML_PATH = "./models/XS-Robot_V2.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# =========================================================
# Limb config for XS-Robot_V2
# All slider values are shown in degrees.
# Internally, MuJoCo ctrl uses radians.
# =========================================================
limb_cfg = {
    "L_arm": {
        "label": "Left Arm",
        "act_names": [
            "act_shoulder_pitch_L",
            "act_shoulder_roll_L",
            "act_elbow_L",
            "act_wrist_L",
        ],
        "extra_name": "act_finger_L",
        "joint_labels": [
            "shoulder_pitch",
            "shoulder_roll",
            "elbow",
            "wrist",
        ],
        "site_name": "finger_L_grip",
        "target_name": "hold1_1_target",
        "deg_ranges": [
            (-90, 90),
            (-90, 90),
            (-90, 90),
            (-90, 90),
        ],
        "extra_range": (-60, 60),  # finger in degree
    },

    "R_arm": {
        "label": "Right Arm",
        "act_names": [
            "act_shoulder_pitch_R",
            "act_shoulder_roll_R",
            "act_elbow_R",
            "act_wrist_R",
        ],
        "extra_name": "act_finger_R",
        "joint_labels": [
            "shoulder_pitch",
            "shoulder_roll",
            "elbow",
            "wrist",
        ],
        "site_name": "finger_R_grip",
        "target_name": "hold2_1_target",
        "deg_ranges": [
            (-90, 90),
            (-90, 90),
            (-90, 90),
            (-90, 90),
        ],
        "extra_range": (-60, 60),  # finger in degree
    },

    "L_leg": {
        "label": "Left Leg",
        "act_names": [
            "act_pelvis_roll_L",
            "act_pelvis_yaw_L",
            "act_pelvis_pitch_L",
            "act_knee_L",
            "act_ankle_L",
        ],
        "extra_name": None,
        "joint_labels": [
            "pelvis_roll",
            "pelvis_yaw",
            "pelvis_pitch",
            "knee",
            "ankle",
        ],
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
        "act_names": [
            "act_pelvis_roll_R",
            "act_pelvis_yaw_R",
            "act_pelvis_pitch_R",
            "act_knee_R",
            "act_ankle_R",
        ],
        "extra_name": None,
        "joint_labels": [
            "pelvis_roll",
            "pelvis_yaw",
            "pelvis_pitch",
            "knee",
            "ankle",
        ],
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
# Helper functions
# =========================================================
def rad_to_deg(x):
    return float(np.rad2deg(x))

def deg_to_rad(x):
    return float(np.deg2rad(x))

def check_name(obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Cannot find MuJoCo name: {name}")
    return obj_id

# =========================================================
# Build ids
# =========================================================
for limb in limb_cfg:
    cfg = limb_cfg[limb]

    cfg["act_ids"] = [
        check_name(mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in cfg["act_names"]
    ]

    if cfg["extra_name"] is not None:
        cfg["extra_aid"] = check_name(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            cfg["extra_name"]
        )
    else:
        cfg["extra_aid"] = None

    cfg["site_sid"] = check_name(
        mujoco.mjtObj.mjOBJ_SITE,
        cfg["site_name"]
    )

    target_sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        cfg["target_name"]
    )
    cfg["target_sid"] = target_sid if target_sid >= 0 else None

# =========================================================
# Shared state
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

pose_names = ["standby", "start", "above", "approach", "hook"]

saved_poses = {
    limb: {pname: None for pname in pose_names}
    for limb in limb_cfg
}

distance_text = {"value": "distance = ---"}
play_active_request = {"go": False}
play_all_request = {"go": False}
move_standby_request = {"go": False}

# =========================================================
# MuJoCo tools
# =========================================================
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
        f"site   = [{grip[0]:.4f}, {grip[1]:.4f}, {grip[2]:.4f}]\n"
        f"target = [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}]"
    )

def current_pose(limb):
    return {
        "joints": ctrl_values[limb]["joints"].copy(),
        "extra": ctrl_values[limb]["extra"],
    }

def move_pose_runtime(viewer, limb, target_pose, target_extra=None, steps=100, sleep=0.01):
    cfg = limb_cfg[limb]
    act_ids = cfg["act_ids"]

    start_pose = np.array([data.ctrl[aid] for aid in act_ids], dtype=float)
    target_pose = np.array(target_pose, dtype=float)

    if cfg["extra_aid"] is not None and target_extra is not None:
        extra_aid = cfg["extra_aid"]
        start_extra = float(data.ctrl[extra_aid])
        target_extra = float(target_extra)
    else:
        extra_aid = None
        start_extra = None

    for i in range(steps):
        if not running:
            return

        r = (i + 1) / steps
        pose = start_pose * (1 - r) + target_pose * r

        for aid, val in zip(act_ids, pose):
            data.ctrl[aid] = float(val)

        with lock:
            ctrl_values[limb]["joints"] = pose.tolist()

        if extra_aid is not None:
            ex = start_extra * (1 - r) + target_extra * r
            data.ctrl[extra_aid] = float(ex)
            with lock:
                ctrl_values[limb]["extra"] = float(ex)

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

    for pose in [pstart, pabove, papproach, phook]:
        move_pose_runtime(
            viewer,
            limb,
            pose["joints"],
            target_extra=pose["extra"],
            steps=100,
            sleep=0.01,
        )

def play_all_paths(viewer):
    seq = ["L_arm", "R_arm", "L_leg", "R_leg"]
    for limb in seq:
        play_limb_path(viewer, limb)

def move_all_to_standby(viewer):
    seq = ["L_arm", "R_arm", "L_leg", "R_leg"]
    for limb in seq:
        pose = saved_poses[limb]["standby"]
        if pose is None:
            print(f"{limb} has no standby pose yet.")
            continue

        print(f"Move {limb} to standby")
        move_pose_runtime(
            viewer,
            limb,
            pose["joints"],
            target_extra=pose["extra"],
            steps=100,
            sleep=0.01,
        )

def print_saved_poses():
    print("\n===== Saved poses in degree =====")
    print("You can copy these values back into code later.\n")

    for limb, cfg in limb_cfg.items():
        print(f"{limb} / {cfg['label']}")
        for pname in pose_names:
            p = saved_poses[limb][pname]
            if p is None:
                print(f"  {pname:8s}: None")
            else:
                degs = [round(rad_to_deg(v), 1) for v in p["joints"]]
                if cfg["extra_aid"] is not None:
                    extra_deg = round(rad_to_deg(p["extra"]), 1)
                    print(f"  {pname:8s}: joints={degs}, extra={extra_deg} deg")
                else:
                    print(f"  {pname:8s}: joints={degs}")
        print("")
    print("=================================\n")

# =========================================================
# UI
# =========================================================
def slider_ui():
    global running

    root = tk.Tk()
    root.title("XS-Robot_V2 Standby Pose Editor")
    root.geometry("1300x860")
    root.minsize(1100, 760)

    top_title = tk.Label(
        root,
        text="XS-Robot_V2 Standby / Path Pose Editor",
        font=("Arial", 16, "bold")
    )
    top_title.pack(pady=8)

    subtitle = tk.Label(
        root,
        text="Select limb -> adjust sliders -> save standby/start/above/approach/hook -> test motion",
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

    # Left panel
    limb_var = tk.StringVar(value="L_arm")
    active_label = tk.Label(left_panel, text="", font=("Arial", 12, "bold"))
    active_label.pack(pady=6)

    tk.Label(left_panel, text="Choose active limb:", font=("Arial", 10)).pack(anchor="w", pady=(4, 6))

    # Center panel with scroll
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
        except Exception:
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

    # Right panel
    distance_box = tk.LabelFrame(right_panel, text="Distance / Site Info", padx=8, pady=8)
    distance_box.pack(fill="x", padx=6, pady=6)

    distance_label = tk.Label(distance_box, text="distance = ---", font=("Consolas", 10), justify="left")
    distance_label.pack(anchor="w")

    pose_box = tk.LabelFrame(right_panel, text="Saved Poses", padx=8, pady=8)
    pose_box.pack(fill="both", expand=True, padx=6, pady=6)

    pose_text = tk.Text(pose_box, width=54, height=34, font=("Consolas", 10))
    pose_text.pack(side="left", fill="both", expand=True)

    pose_scroll = tk.Scrollbar(pose_box, orient="vertical", command=pose_text.yview)
    pose_scroll.pack(side="right", fill="y")
    pose_text.configure(yscrollcommand=pose_scroll.set, state="disabled")

    # Callbacks
    def on_joint_slider(i, val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["joints"][i] = deg_to_rad(float(val))

    def on_extra_slider(val):
        limb = active_limb["name"]
        with lock:
            ctrl_values[limb]["extra"] = deg_to_rad(float(val))

    def load_current_ctrl_to_sliders():
        limb = active_limb["name"]
        cfg = limb_cfg[limb]

        with lock:
            vals = ctrl_values[limb]["joints"].copy()
            ex = ctrl_values[limb]["extra"]

        for s, v in zip(sliders, vals):
            s.set(rad_to_deg(v))

        if cfg["extra_aid"] is not None and extra_slider["widget"] is not None:
            extra_slider["widget"].set(rad_to_deg(ex))

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
                from_=rg[0],
                to=rg[1],
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
                from_=cfg["extra_range"][0],
                to=cfg["extra_range"][1],
                resolution=1,
                orient="horizontal",
                length=620,
                label="finger extra (deg)",
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
                        extra_txt = rad_to_deg(p["extra"])
                        lines.append(f"  {pname:8s}: [{joint_txt}] finger={extra_txt:6.1f}")
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
            lines.append(f"{name:15s} = {rad_to_deg(v):7.2f} deg")

        if cfg["extra_aid"] is not None:
            lines.append(f"{'finger':15s} = {rad_to_deg(ex):7.2f} deg")

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

    def move_to_standby():
        with lock:
            move_standby_request["go"] = True

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

    # Limb radio buttons must be created after rebuild_sliders is defined
    for limb in ["L_arm", "R_arm", "L_leg", "R_leg"]:
        tk.Radiobutton(
            left_panel,
            text=limb,
            variable=limb_var,
            value=limb,
            command=rebuild_sliders
        ).pack(anchor="w", pady=3)

    # Bottom controls
    pose_btn_frame = tk.Frame(bottom_panel)
    pose_btn_frame.pack(fill="x", pady=4)

    tk.Button(pose_btn_frame, text="Set Standby", command=lambda: save_pose("standby"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Start", command=lambda: save_pose("start"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Above", command=lambda: save_pose("above"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Approach", command=lambda: save_pose("approach"), width=14).pack(side="left", padx=6, pady=4)
    tk.Button(pose_btn_frame, text="Set Hook", command=lambda: save_pose("hook"), width=14).pack(side="left", padx=6, pady=4)

    cmd_btn_frame = tk.Frame(bottom_panel)
    cmd_btn_frame.pack(fill="x", pady=4)

    tk.Button(cmd_btn_frame, text="Move All To Standby", command=move_to_standby, width=18).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Play Active Path", command=play_active_path, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Play All Paths", command=play_all, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Print Poses", command=print_saved_poses, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(cmd_btn_frame, text="Reload Sliders", command=load_current_ctrl_to_sliders, width=16).pack(side="left", padx=6, pady=4)

    reset_btn_frame = tk.Frame(bottom_panel)
    reset_btn_frame.pack(fill="x", pady=4)

    tk.Button(reset_btn_frame, text="Reset Active", command=reset_active, width=16).pack(side="left", padx=6, pady=4)
    tk.Button(reset_btn_frame, text="Reset All", command=reset_all, width=16).pack(side="left", padx=6, pady=4)

    def on_close():
        global running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    rebuild_sliders()
    refresh_labels()
    root.mainloop()

# =========================================================
# Start UI thread
# =========================================================
ui_thread = threading.Thread(target=slider_ui, daemon=True)
ui_thread.start()

with lock:
    apply_all_ctrl()

for _ in range(100):
    mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Viewer started.")
    print("XS-Robot_V2 standby pose editor is running.")
    print("1. Select one limb in UI.")
    print("2. Adjust sliders.")
    print("3. Click Set Standby / Set Start / Set Above / Set Approach / Set Hook.")
    print("4. Use Move All To Standby or Play Active Path.")
    print("5. Press Enter in terminal to exit.\n")

    def wait_for_enter():
        global running
        input()
        running = False

    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    while running:
        do_play_active = False
        do_play_all = False
        do_move_standby = False

        with lock:
            apply_all_ctrl()
            do_play_active = play_active_request["go"]
            do_play_all = play_all_request["go"]
            do_move_standby = move_standby_request["go"]

        if do_move_standby:
            with lock:
                move_standby_request["go"] = False
            move_all_to_standby(viewer)

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