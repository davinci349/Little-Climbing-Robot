from machine import Pin, PWM, Timer
import time

# ==================================================
# XS-Robot Controller
# Features:
#   1. Move servo by angle
#   2. Keep mode: servo keeps holding torque
#   3. Release all keep joints or one keep joint
#   4. Show angles / limits / keep status
#   5. LED warning:
#        slow blink = normal
#        fast blink = keep mode active
# ==================================================

# ================= LED =================
led = Pin("LED", Pin.OUT)
led_timer = Timer()

def blink(timer):
    led.toggle()

def led_fast():
    # Fast blink means at least one joint is in KEEP mode
    led_timer.init(freq=5, mode=Timer.PERIODIC, callback=blink)

def led_slow():
    # Slow blink means normal mode
    led_timer.init(freq=1, mode=Timer.PERIODIC, callback=blink)

# ================= pins =================
pins = {
    "R_hand": 1, "R_arm": 4, "R_shoulder_roll": 3, "R_shoulder_pitch": 26,
    "R_knee": 2, "R_thigh": 14, "R_pelvis": 15, "R_hip": 13,
    "L_hand": 5, "L_arm": 0, "L_shoulder_roll": 7, "L_shoulder_pitch": 12,
    "L_knee": 6, "L_thigh": 8, "L_pelvis": 9, "L_hip": 11,
      
}

# Servo neutral / standby PWM values in microseconds
base_state = {
    "L_knee": 2050, "L_thigh": 2100, "L_pelvis": 2200, "L_hip": 950,
    "R_knee": 900, "R_thigh": 1050, "R_pelvis": 900, "R_hip": 2000,
    "L_hand": 1500, "L_arm": 1600, "L_shoulder_pitch": 1600, "L_shoulder_roll": 1400,
    "R_hand": 1600, "R_arm": 1500, "R_shoulder_pitch": 1400, "R_shoulder_roll": 1500
}

state = base_state.copy()
pwms = {}
keep_registry = set()

# How many servos can stay in KEEP mode.
# For climbing ready pose, several limbs need to hold position.
MAX_KEEP_JOINTS = 16

# Run climbing ready pose automatically after startup.
AUTO_CLIMB_READY_ON_START = True

# 1 degree = 10 us change
DEG_TO_US = 10

# ================= limits =================
LIMITS = {
    "knee": (0, 90),
    "thigh": (0, 90),
    "pelvis": (0, 90),
    "hip": (0, 90),
    
    "hand": (-90, 90),
    "arm": (-90, 90),
    "shoulder_pitch": (-35, 70),
    "shoulder_roll": (-90, 90)
}

# Real servo direction
# +1 means angle increase makes PWM increase
# -1 means angle increase makes PWM decrease
configs = {
    "knee": {"L": -1, "R": 1},
    "thigh": {"L": -1, "R": 1},
    "pelvis": {"L": -1, "R": 1},
    "hip": {"L": 1, "R": -1},
    "hand": {"L": -1, "R": 1},
    "arm": {"L": -1, "R": 1},
    "shoulder_pitch": {"L": 1, "R": -1},
    "shoulder_roll": {"L": 1, "R": -1},
}

# Group command mapping
mapping = {
    "hand": ["L_hand", "R_hand"],
    "arm": ["L_arm", "R_arm"],
    "shoulder_pitch": ["L_shoulder_pitch", "R_shoulder_pitch"],
    "shoulder_roll": ["L_shoulder_roll", "R_shoulder_roll"],
    "knee": ["L_knee", "R_knee"],
    "hip": ["L_hip", "R_hip"],
    "thigh": ["L_thigh", "R_thigh"],
    "pelvis": ["L_pelvis", "R_pelvis"]
}

# ================= helper =================
def update_led_status():
    if len(keep_registry) > 0:
        led_fast()
    else:
        led_slow()

def show_help():
    print("\n===================================")
    print("      XS-Robot Controller")
    print("===================================")
    print("Move one joint:")
    print("  L_knee 30")
    print("  R_hip 20")
    print("")
    print("Move group joints:")
    print("  knee 30              -> move L_knee and R_knee")
    print("  shoulder_pitch 20    -> move both shoulder pitch")
    print("")
    print("Keep Torque:")
    print("  L_knee 30 keep       -> move and keep L_knee")
    print("  knee 30 keep         -> move and keep both knees")
    print("")
    print("Release:")
    print("  release              -> release ALL keep joints")
    print("  release L_knee       -> release only L_knee")
    print("  release knee         -> release L_knee and R_knee if they are kept")
    print("")
    print("Check:")
    print("  angles               -> show ALL joint angles")
    print("  angles keep          -> show KEEP joints only")
    print("  limits               -> show all limits")
    print("  limits knee          -> show one joint limit")
    print("  save                 -> print current pose")
    print("  ready                -> run climbing ready pose")
    print("  stop                 -> emergency stop all PWM")
    print("")
    print("LED:")
    print("  slow blink = normal")
    print("  fast blink = KEEP ACTIVE")
    print("===================================")

def resolve_names(input_name):
    # Convert group command to servo names.
    # Example: "knee" -> ["L_knee", "R_knee"]
    return mapping.get(input_name, [input_name])

def get_angle(name):
    if name not in state or name not in base_state:
        return None

    diff = state[name] - base_state[name]
    parts = name.split("_")
    side = parts[0]
    joint = "_".join(parts[1:])

    if joint not in configs:
        return None

    if side not in configs[joint]:
        return None

    return diff / (configs[joint][side] * DEG_TO_US)

def show_joint_status(names):
    print("\n=== JOINT STATUS ===")

    shown = False
    for n in names:
        ang = get_angle(n)
        if ang is None:
            continue

        keep_text = "KEEP" if n in keep_registry else "FREE"
        print(f"{n:<20} : angle={round(ang,1):>5} deg | {keep_text}")
        shown = True

    if not shown:
        print("(no angle information)")

def show_keep_status():
    print("\n=== KEEP STATUS ===")
    if len(keep_registry) == 0:
        print("No joint is in KEEP mode.")
    else:
        for n in keep_registry:
            print(f"{n} is KEEP")

def emergency_stop():
    print("STOP: all PWM released")

    for n in list(pwms.keys()):
        pwms[n].deinit()
        del pwms[n]

    keep_registry.clear()
    update_led_status()

def write_us(name, us):
    us = max(800, min(2200, us))
    duty = int(us * 65535 / 20000)

    if name not in pwms:
        pwms[name] = PWM(Pin(pins[name]))
        pwms[name].freq(50)

    pwms[name].duty_u16(duty)
    state[name] = us

# ================= move =================
def move_angle(input_name, input_angle, duration=0.5, keep=False):
    targets = {}
    names = resolve_names(input_name)

    if keep:
        new_keep = []
        for n in names:
            if n in base_state and n not in keep_registry:
                new_keep.append(n)

        if len(keep_registry) + len(new_keep) > MAX_KEEP_JOINTS:
            print("WARNING: keep joints cannot be more than", MAX_KEEP_JOINTS)
            print("Current keep joints:", keep_registry)
            update_led_status()
            show_keep_status()
            return
    

    for name in names:
        if name not in base_state:
            print("Cannot find servo:", name)
            continue

        parts = name.split("_")
        side = parts[0]
        joint = "_".join(parts[1:])

        if joint not in configs or side not in configs[joint]:
            print("This joint has no angle config:", name)
            continue

        mn, mx = LIMITS.get(joint, (-60, 60))
        angle = max(mn, min(input_angle, mx))

        if angle != input_angle:
            print(f"Angle limited: {input_angle} -> {angle} for {name}")

        targets[name] = base_state[name] + angle * DEG_TO_US * configs[joint][side]

    if not targets:
        print("No valid servo target.")
        update_led_status()
        return

    start_state = state.copy()

    for i in range(21):
        for n in targets:
            us = int(start_state[n] + (targets[n] - start_state[n]) * i / 20)
            write_us(n, us)

        time.sleep(duration / 20)

    for n in targets:
        write_us(n, targets[n])

    for n in targets:
        if keep:
            keep_registry.add(n)
        else:
            # If this joint is already in keep mode, keep holding it.
            # If not in keep mode, release PWM after moving.
            if n not in keep_registry and n in pwms:
                pwms[n].deinit()
                del pwms[n]

    update_led_status()
    show_joint_status(targets.keys())
    show_keep_status()

def release_joint(input_name):
    names = resolve_names(input_name)
    released_any = False

    for n in names:
        if n in keep_registry:
            keep_registry.remove(n)

            if n in pwms:
                pwms[n].deinit()
                del pwms[n]

            print("released", n)
            released_any = True
        else:
            print(n, "is not in KEEP mode")

    if not released_any:
        print("No keep joint was released.")

    update_led_status()
    show_keep_status()

def release_all():
    # Release every active PWM, not only KEEP joints.
    for n in list(pwms.keys()):
        pwms[n].deinit()
        del pwms[n]

    keep_registry.clear()
    print("released ALL PWM")

    update_led_status()
    show_keep_status()

def show_limits(input_joint=None):
    def get_arrow(val):
        if val == 1:
            return "+ PWM"
        elif val == -1:
            return "- PWM"
        else:
            return "-"

    def show_one_joint(j):
        if j not in LIMITS:
            print("Cannot find joint:", j)
            return

        mn, mx = LIMITS[j]
        cfg = configs.get(j, {})

        if "L" in cfg or "R" in cfg:
            L_arrow = get_arrow(cfg.get("L", 0))
            R_arrow = get_arrow(cfg.get("R", 0))
            print(f"{j:<15} : {mn:>4} deg ~ {mx:<4} deg | L:{L_arrow:<6} R:{R_arrow:<6}")

        elif "C" in cfg:
            C_arrow = get_arrow(cfg["C"])
            print(f"{j:<15} : {mn:>4} deg ~ {mx:<4} deg | C:{C_arrow}")

        else:
            print(f"{j:<15} : {mn:>4} deg ~ {mx:<4} deg")

    if input_joint:
        show_one_joint(input_joint)
    else:
        print("\n=== JOINT LIMITS + REAL PWM DIRECTION ===")
        for j in LIMITS:
            show_one_joint(j)


# ================= climbing ready pose =================
# Each tuple means: (joint_name, target_angle, keep_after_move)
# Movement order:
#   left hand -> right hand -> left leg -> right leg
CLIMB_READY_SEQUENCE = [
    
      # Left hand
    ("L_shoulder_roll", 70, True),

    # Right hand
    ("R_shoulder_roll", -20, True),
    ("R_arm", -60, True),

    # Left leg
    ("L_pelvis", 80, True),
    ("L_thigh", 85, True),
    ("L_knee", 85, True),
    ("L_hip", 5, True),
    
    # Right leg
    ("R_pelvis", 80, True),
    ("R_thigh", 55, True),
    ("R_knee", 55, True),

]

JOINT_DETECT_SEQUENCE = [
    ("L_hand", 0, True), ("L_hand", 30, True), ("L_hand", -30, True), ("L_hand", 0, True),
    ("L_arm", 0, True), ("L_arm", -30, True), ("L_arm", -60, True), ("L_arm", 0, True),
    ("L_shoulder_roll", 0, True), ("L_shoulder_roll", 30, True), ("L_shoulder_roll", 60, True), ("L_shoulder_roll", 0, True),
    ("L_shoulder_pitch", 0, True), ("L_shoulder_pitch", 30, True), ("L_shoulder_pitch", 60, True), ("L_shoulder_pitch", 0, True),
    ("L_knee", 0, True), ("L_knee", 30, True), ("L_knee", 60, True), ("L_knee", 0, True),
    ("L_thigh", 0, True), ("L_thigh", 30, True), ("L_thigh", 60, True), ("L_thigh", 0, True),
    ("L_pelvis", 0, True), ("L_pelvis", 30, True), ("L_pelvis", 60, True), ("L_pelvis", 0, True),
    ("L_hip", 0, True), ("L_hip", 30, True), ("L_hip", 60, True), ("L_hip", 0, True),

]

def run_climb_ready_pose():
    print("")
    print("===================================")
    print("Running CLIMB READY POSE")
    print("Order: left hand -> right hand -> left leg -> right leg")
    print("===================================")

    for name, angle, keep in CLIMB_READY_SEQUENCE:
        print("Pose step:", name, angle, "keep" if keep else "")
        move_angle(name, angle, duration=0.8, keep=keep)
        time.sleep(0.5)

    print("")
    print("CLIMB READY POSE finished.")
    show_joint_status(state.keys())
    show_keep_status()


def run_joint_detect_pose():
    print("Running JOINT DETECTION SEQUENCE")


    print("Running JOINT DETECTION SEQUENCE")
    for name, angle, keep in JOINT_DETECT_SEQUENCE:
        print("Detect step:", name, angle)
        move_angle(name, angle, duration=0.8, keep=keep)
        time.sleep(0.5)

    print("JOINT DETECTION finished.")
    show_joint_status(state.keys())

def save_pose():
    print("pose = {")
    for n in state:
        ang = get_angle(n)
        if ang is not None:
            print(f'    "{n}": {round(ang,1)},')
    print("}")

# ================= main =================
try:
    led_slow()

    # Move all joints to standby pose at startup
    for n in base_state:
        write_us(n, base_state[n])
        time.sleep(0.05)
        release_all()
        
    print("READY")
    show_help()

    while True:
        cmd_in = input("\nPWM > ").split()

        if not cmd_in:
            show_help()
            continue

        cmd = cmd_in[0]

        # Show help every command, so you always remember how to use it
        show_help()

        # ========= HELP =========
        if cmd in ["help", "?"]:
            continue

        # ========= STOP =========
        if cmd in ["stop", "emergency"]:
            emergency_stop()
            continue

        # ========= LIMITS =========
        if cmd == "limits":
            if len(cmd_in) >= 2:
                show_limits(cmd_in[1])
            else:
                show_limits()
            continue

        # ========= RELEASE =========
        if cmd in ["release", "free"]:
            if len(cmd_in) >= 2:
                release_joint(cmd_in[1])
            else:
                release_all()
            continue

        # ========= ANGLES =========
        if cmd == "angles":
            if len(cmd_in) >= 2 and cmd_in[1] == "keep":
                show_joint_status(keep_registry)
            else:
                show_joint_status(state.keys())

            show_keep_status()
            continue

        # ========= CLIMB READY POSE =========
        if cmd in ["ready", "climb_ready"]:
            run_climb_ready_pose()
            continue

        # ========= JOINT DETECTION =========
        if cmd in ["detect", "joint_detect"]:
            run_joint_detect_pose()
            continue

        # ========= SAVE POSE =========
        if cmd in ["save", "record"]:
            save_pose()
            continue

        # ========= MOVE =========
        if len(cmd_in) >= 2:
            try:
                angle = int(cmd_in[1])
                keep = (len(cmd_in) >= 3 and cmd_in[2] == "keep")
                move_angle(cmd, angle, keep=keep)
            except ValueError:
                print("Angle must be a number. Example: L_knee 30")
        else:
            print("Wrong command. Please check the operation guide above.")

except KeyboardInterrupt:
    emergency_stop()

except Exception as e:
    print("ERROR:", e)
    emergency_stop()
