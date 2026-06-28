import cv2
import numpy as np
import math
import time
import serial

# ================= SERIAL =================
PICO_PORT = "COM4"
BAUD = 115200

pico = serial.Serial(PICO_PORT, BAUD, timeout=0.5)
time.sleep(3)
print("Connected to Pico:", PICO_PORT)

def send_pico(cmd):
    pico.write((cmd + "\n").encode())
    print("SEND:", cmd)

    time.sleep(0.05)

    while pico.in_waiting:
        print("PICO:", pico.readline().decode(errors="ignore").strip())

# ================= SETTINGS =================
AUTO_MOVE = False
last_send_time = 0

SEND_INTERVAL = 0.25
STEP = 3
MOVE_THRESHOLD = 5
DIST_OK = 20

# direction signs
ROLL_SIGN = 1
ARM_SIGN = 1

clicked_point = None
manual_lhand_point = None

# ================= HELPERS =================
def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mouse_callback(event, x, y, flags, param):
    global clicked_point, AUTO_MOVE, last_send_time, manual_lhand_point

    # Left click: select any wall point as target
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        AUTO_MOVE = False
        last_send_time = 0

        print("Selected target point:", clicked_point)
        print("Press m to start AUTO_MOVE")

    # Right click: manually select L_hand green marker
    elif event == cv2.EVENT_RBUTTONDOWN:
        manual_lhand_point = (x, y)
        AUTO_MOVE = False
        print("Manual L_hand selected near:", manual_lhand_point)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Detection")
cv2.setMouseCallback("Detection", mouse_callback)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read camera frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)

    # ==================================================
    # RED: BODY MARKERS
    # ==================================================
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 80, 80])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 + red_mask2

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    red_contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    red_points = []

    for cnt in red_contours:
        area = cv2.contourArea(cnt)

        if area < 80 or area > 8000:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        red_points.append((cx, cy, area))

    red_points = sorted(red_points, key=lambda p: p[2], reverse=True)[:4]

    if len(red_points) == 4:
        pts = [(p[0], p[1]) for p in red_points]
        pts = sorted(pts, key=lambda p: p[1])

        top = sorted(pts[:2], key=lambda p: p[0])
        bottom = sorted(pts[2:], key=lambda p: p[0])

        P1 = top[0]
        P2 = top[1]
        P3 = bottom[0]
        P4 = bottom[1]

        for label, p in zip(["P1", "P2", "P3", "P4"], [P1, P2, P3, P4]):
            cv2.circle(frame, p, 8, (0, 0, 255), -1)
            cv2.putText(
                frame,
                label,
                (p[0] + 10, p[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        center_x = int((P1[0] + P2[0] + P3[0] + P4[0]) / 4)
        center_y = int((P1[1] + P2[1] + P3[1] + P4[1]) / 4)

        cv2.circle(frame, (center_x, center_y), 10, (255, 0, 255), -1)
        cv2.putText(
            frame,
            "BODY CENTER",
            (center_x + 15, center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

    # ==================================================
    # GREEN: HANDS / SOLES
    # ==================================================
    lower_green = np.array([35, 60, 60])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    green_contours, _ = cv2.findContours(
        green_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    green_points = []

    for cnt in green_contours:
        area = cv2.contourArea(cnt)

        if area < 80 or area > 8000:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect

        if w == 0 or h == 0:
            continue

        if w < h:
            angle += 90

        cx = int(cx)
        cy = int(cy)

        box = cv2.boxPoints(rect).astype(np.int32)

        green_points.append((cx, cy, area, box, angle))

    green_points = sorted(green_points, key=lambda p: p[2], reverse=True)[:4]

    L_hand = None
    R_hand = None
    L_sole = None
    R_sole = None

    if len(green_points) > 0:
        if manual_lhand_point is not None:
            mx, my = manual_lhand_point
            L_hand = min(
                green_points,
                key=lambda p: dist((p[0], p[1]), (mx, my))
            )
        else:
            gpts = sorted(green_points, key=lambda p: p[1])

            if len(gpts) >= 2:
                hands = sorted(gpts[:2], key=lambda p: p[0])
                L_hand = hands[0]
                R_hand = hands[1]

            if len(gpts) >= 4:
                soles = sorted(gpts[2:], key=lambda p: p[0])
                L_sole = soles[0]
                R_sole = soles[1]

    for i, item in enumerate(green_points):
        cx, cy, area, box, angle = item

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"G{i + 1}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    if L_hand is not None:
        hx, hy, area, box, angle = L_hand

        cv2.circle(frame, (hx, hy), 12, (255, 0, 255), 3)
        cv2.putText(
            frame,
            "L_hand",
            (hx + 15, hy + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

    # ==================================================
    # BLUE: WALL HOLDS DISPLAY ONLY
    # ==================================================
    lower_blue = np.array([100, 80, 20])
    upper_blue = np.array([135, 255, 180])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    blue_contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    holds = []

    for cnt in blue_contours:
        area = cv2.contourArea(cnt)

        if area < 800 or area > 12000:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), hold_angle = rect

        if w == 0 or h == 0:
            continue

        if max(w, h) / min(w, h) > 3.0:
            continue

        if area / (w * h) < 0.45:
            continue

        cx = int(cx)
        cy = int(cy)

        box = cv2.boxPoints(rect).astype(np.int32)
        holds.append((cx, cy, box))

    holds = sorted(holds, key=lambda h: (h[1], h[0]))


    
    for hold_id, (cx, cy, box) in enumerate(holds, start=1):
        cv2.drawContours(frame, [box], 0, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        # ===== find top edge of rotated hold =====
        pts = box.tolist()

        edges = []
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]

            mid_y = (p1[1] + p2[1]) / 2
            edge_len = dist(p1, p2)

            edges.append((mid_y, edge_len, p1, p2))

        # smaller y means upper edge in image
        edges = sorted(edges, key=lambda e: e[0])
        top_edge = edges[0]

        p1 = tuple(top_edge[2])
        p2 = tuple(top_edge[3])

        # draw top tilted edge
        cv2.line(frame, p1, p2, (0, 255, 255), 4)

        cv2.putText(
            frame,
            f"H{hold_id}",
            (cx + 30, cy - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
    
    # ==================================================
    # TARGET: ANY CLICKED POINT
    # ==================================================
    target_point = clicked_point

    if target_point is not None:
        tx, ty = target_point

        cv2.circle(frame, (tx, ty), 10, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "TARGET",
            (tx + 15, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ==================================================
    # VISUAL SERVO CONTROL
    # ==================================================
    if AUTO_MOVE and target_point is not None and L_hand is not None:
        now = time.time()

        if now - last_send_time > SEND_INTERVAL:
            tx, ty = target_point
            hx, hy, area, hbox, hand_angle = L_hand

            ex = tx - hx
            ey = ty - hy
            d = dist((hx, hy), (tx, ty))

            print(f"[VS] d={d:.1f}, ex={ex}, ey={ey}")

            if d < DIST_OK:
                print("Target reached")
                AUTO_MOVE = False
                send_pico("stop")

            else:
                sent = False

                if abs(ex) > MOVE_THRESHOLD:
                    if ex > 0:
                        send_pico(f"rel L_shoulder_roll {ROLL_SIGN * -STEP}")
                    else:
                        send_pico(f"rel L_shoulder_roll {ROLL_SIGN * STEP}")

                    sent = True

                if abs(ey) > MOVE_THRESHOLD:
                    if ey > 0:
                        send_pico(f"rel L_arm {ARM_SIGN * STEP}")
                    else:
                        send_pico(f"rel L_arm {ARM_SIGN * -STEP}")

                    sent = True

                if not sent:
                    print("Error small, no command sent")

            last_send_time = now

    # ==================================================
    # DISPLAY
    # ==================================================
    status = "AUTO_MOVE: ON" if AUTO_MOVE else "AUTO_MOVE: OFF"

    cv2.putText(
        frame,
        status,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0) if AUTO_MOVE else (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "Left click any target | Right click L_hand | m=start | c=clear | ESC=quit",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Detection", frame)
    cv2.imshow("Blue Mask", blue_mask)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    elif key == ord("m"):
        if target_point is None:
            print("No target. Left click any point first.")
        elif L_hand is None:
            print("No L_hand detected.")
        else:
            AUTO_MOVE = not AUTO_MOVE
            last_send_time = 0
            print("AUTO_MOVE:", AUTO_MOVE)

    elif key == ord("c"):
        clicked_point = None
        AUTO_MOVE = False
        last_send_time = 0
        print("Selection cleared")

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
pico.close()