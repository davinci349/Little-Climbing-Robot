from machine import Pin, PWM, Timer
import time

# ================= LED =================
led = Pin("LED", Pin.OUT)
led_timer = Timer()

def blink(timer):
    led.toggle()

def led_fast():
    led_timer.init(freq=5, mode=Timer.PERIODIC, callback=blink)

def led_slow():
    led_timer.init(freq=1, mode=Timer.PERIODIC, callback=blink)

# ================= pins =================
pins = {
    "R_hand": 1, "R_knee": 2, "R_arm": 4, "R_thigh": 14,
    "R_pelvis": 15, "R_hip": 13, "R_shoulder_pitch": 26, "R_shoulder_roll": 3,
    "L_ankle": 21, "L_hand": 5, "L_knee": 6, "L_arm": 0,
    "L_thigh": 8, "L_pelvis": 9, "L_hip": 11,
    "L_shoulder_pitch": 12, "L_shoulder_roll": 7,
    "C_spine": 27
}

base_state = {
    "L_ankle": 1400, "L_knee": 2050, "L_thigh": 2200, "L_pelvis": 2200, "L_hip": 950,
    "C_spine": 1500,
    "R_knee": 900, "R_thigh": 1000, "R_pelvis": 900, "R_hip": 2000,
    "L_hand": 1500, "L_arm": 1500, "L_shoulder_pitch": 1400, "L_shoulder_roll": 1400,
    "R_hand": 1600, "R_arm": 1600, "R_shoulder_pitch": 1400, "R_shoulder_roll": 1400
}

state = base_state.copy()
pwms = {}
keep_registry = set()
DEG_TO_US = 10

# ================= limits =================
LIMITS = {
    "knee": (0, 90),
    "thigh": (0, 90),
    "pelvis": (0, 60),
    "hip": (0, 60),
    "arm": (-60, 60),
    "shoulder_pitch": (-60, 60),
    "shoulder_roll": (-60, 60),
    "ankle": (-60, 60),
    "spine": (-60, 60)
}

configs = {
    "knee": {"L": -1, "R": 1},
    "thigh": {"L": -1, "R": 1},
    "pelvis": {"L": -1, "R": 1},
    "hip": {"L": 1, "R": -1},
    "ankle": {"L": -1},
    "spine": {"C": 1},
    "arm": {"L": -1, "R": 1},
    "shoulder_pitch": {"L": 1, "R": -1},
    "shoulder_roll": {"L": 1, "R": -1},
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
    print("Move:")
    print("  L_knee 30")
    print("  shoulder_pitch 20")
    print("")
    print("Keep Torque:")
    print("  L_knee 30 keep")
    print("")
    print("Commands:")
    print("  release       -> release keep servos")
    print("  angles        -> show ALL joint angles")
    print("  angles keep   -> show KEEP joints only")
    print("  limits        -> show all limits")
    print("  limits knee   -> show one joint limit")
    print("  save          -> save current pose")
    print("")
    print("LED:")
    print("  slow blink = normal")
    print("  fast blink = KEEP ACTIVE ⚠️")
    print("===================================")

def get_angle(name):
    diff = state[name] - base_state[name]
    parts = name.split("_")
    side = parts[0]
    joint = "_".join(parts[1:])

    if joint not in configs:
        return None

    return diff / (configs[joint][side] * DEG_TO_US)

def show_joint_status(names):
    print("\n=== JOINT STATUS ===")
    for n in names:
        ang = get_angle(n)
        if ang is None:
            continue

        keep_text = "KEEP 🔒" if n in keep_registry else "FREE 🔓"
        print(f"{n:<20} : angle={round(ang,1):>5}° | {keep_text}")

def emergency_stop():
    print("🚨 STOP")
    for n in list(pwms.keys()):
        pwms[n].deinit()
        del pwms[n]

    keep_registry.clear()
    update_led_status()

def write_us(name, us):
    us = max(800, min(2200, us))
    duty = int(us * 65535 / 20000)

    if name not in pwms:
        pwms[name] = PWM(Pin(pins[name]), freq=50)

    pwms[name].duty_u16(duty)
    state[name] = us

# ================= move =================
def move_angle(input_name, input_angle, duration=0.5, keep=False):
    targets = {}

    mapping = {
        "spine": ["C_spine"],
        "arm": ["L_arm", "R_arm"],
        "shoulder_pitch": ["L_shoulder_pitch", "R_shoulder_pitch"],
        "shoulder_roll": ["L_shoulder_roll", "R_shoulder_roll"],
        "knee": ["L_knee", "R_knee"],
        "hip": ["L_hip", "R_hip"],
        "ankle": ["L_ankle"],
        "thigh": ["L_thigh", "R_thigh"],
        "pelvis": ["L_pelvis", "R_pelvis"]
    }

    names = mapping.get(input_name, [input_name])

    if keep:
        new = [n for n in names if n not in keep_registry and n in base_state]
        if len(keep_registry) + len(new) > 3:
            print("⚠️ keep 太多:", keep_registry)
            update_led_status()
            return

    for name in names:
        if name not in base_state:
            print("❌ 找不到 servo:", name)
            continue

        parts = name.split("_")
        side = parts[0]
        joint = "_".join(parts[1:])

        if joint in configs:
            mn, mx = LIMITS.get(joint, (-60, 60))
            angle = max(mn, min(input_angle, mx))
            targets[name] = base_state[name] + angle * DEG_TO_US * configs[joint][side]

    if not targets:
        print("❌ 沒有可移動的目標")
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
            if n in keep_registry:
                pass
            else:
                if n in pwms:
                    pwms[n].deinit()
                    del pwms[n]

    update_led_status()
    show_joint_status(targets.keys())

# ================= main =================
try:
    led_slow()

    for n, p in base_state.items():
        write_us(n, p)
        time.sleep(0.05)

    print("READY")

    while True:
        show_help()

        cmd_in = input("PWM > ").split()

        if not cmd_in:
            continue

        cmd = cmd_in[0]

        # ========= LIMITS =========
        if cmd == "limits":

            def get_arrow(val):
                if val == 1:
                    return "↻ (+)"
                elif val == -1:
                    return "↺ (-)"
                else:
                    return "·"

            def show_joint(j):
                if j in LIMITS:
                    mn, mx = LIMITS[j]
                    cfg = configs.get(j, {})

                    if "L" in cfg or "R" in cfg:
                        L_arrow = get_arrow(cfg.get("L", 0))
                        R_arrow = get_arrow(cfg.get("R", 0))
                        print(f"{j:<15} : {mn:>4}° ~ {mx:<4}° | L:{L_arrow:<6} R:{R_arrow:<6}")

                    elif "C" in cfg:
                        C_arrow = get_arrow(cfg["C"])
                        print(f"{j:<15} : {mn:>4}° ~ {mx:<4}° | C:{C_arrow}")

                    else:
                        print(f"{j:<15} : {mn:>4}° ~ {mx:<4}°")
                else:
                    print(f"❌ 找不到關節: {j}")

            if len(cmd_in) == 2:
                show_joint(cmd_in[1])
            else:
                print("\n=== JOINT LIMITS + REAL DIRECTION ===")
                for j in LIMITS:
                    show_joint(j)

            continue

        if cmd in ["release", "free"]:

        # release single joint
            if len(cmd_in) >= 2:

                target = cmd_in[1]

                if target in keep_registry:

                    keep_registry.remove(target)

                    if target in pwms:
                        pwms[target].deinit()
                        del pwms[target]

                    print(f"🔓 released {target}")

                else:
                    print(f"⚠️ {target} not in keep mode")

        # release all
            else:

                for n in list(keep_registry):
                    if n in pwms:
                        pwms[n].deinit()
                        del pwms[n]

                keep_registry.clear()

                print("🔓 released ALL")

            update_led_status()
            continue

            # ========= ANGLES =========
            if cmd == "angles":
                if len(cmd_in) >= 2 and cmd_in[1] == "keep":
                    show_joint_status(keep_registry)
                else:
                    show_joint_status(state.keys())

            continue

        # ========= SAVE POSE =========
        if cmd in ["save", "record"]:
            print("pose = {")
            for n in state:
                ang = get_angle(n)
                if ang is not None:
                    print(f'    "{n}": {round(ang,1)},')
            print("}")
            continue

        # ========= MOVE =========
        if len(cmd_in) >= 2:
            try:
                angle = int(cmd_in[1])
                keep = (len(cmd_in) >= 3 and cmd_in[2] == "keep")
                move_angle(cmd, angle, keep=keep)
            except ValueError:
                print("❌ angle 必須是數字，例如：L_knee 30")
        else:
            print("❌ 指令錯誤，請參考上方操作說明")

except KeyboardInterrupt:
    emergency_stop()

except Exception as e:
    print("ERROR:", e)
    emergency_stop()