import cv2

CAMERA_FLIP_X = False

def robot_to_pixel(x, z, center_x=320, center_y=300, scale=1800):
    if CAMERA_FLIP_X:
        x = -x

    px = int(center_x + x * scale)
    py = int(center_y - z * scale)
    return px, py


joint_pos_robot = {
    "waist": (0.0000, 0.0000),
    "pelvis": (-0.0326, -0.0283),

    "L_shoulder": (0.0428, -0.0068),
    "L_elbow": (0.0287, 0.0185),
    "L_wrist": (0.0478, 0.0234),
    "L_hand": (0.0565, 0.0450),

    "R_shoulder": (-0.1113, -0.0065),
    "R_elbow": (-0.0940, 0.0185),
    "R_wrist": (-0.1130, 0.0234),
    "R_hand": (0.0565, -0.0450),

    "L_hip": (-0.0046, -0.0098),
    "L_knee": (-0.0070, 0.0018),
    "L_ankle": (-0.0027, -0.0172),
    "L_foot": (-0.0030, 0.0200),

    "R_hip": (-0.0606, -0.0090),
    "R_knee": (-0.0585, 0.0019),
    "R_ankle": (-0.0629, -0.0173),
    "R_foot": (-0.0630, 0.0200),
}

bones = [
    ("waist", "pelvis"),

    ("waist", "L_shoulder"),
    ("L_shoulder", "L_elbow"),
    ("L_elbow", "L_wrist"),
    ("L_wrist", "L_hand"),

    ("waist", "R_shoulder"),
    ("R_shoulder", "R_elbow"),
    ("R_elbow", "R_wrist"),
    ("R_wrist", "R_hand"),

    ("pelvis", "L_hip"),
    ("L_hip", "L_knee"),
    ("L_knee", "L_ankle"),
    ("L_ankle", "L_foot"),

    ("pelvis", "R_hip"),
    ("R_hip", "R_knee"),
    ("R_knee", "R_ankle"),
    ("R_ankle", "R_foot"),
]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    joint_pos_pixel = {}

    for name, (x, z) in joint_pos_robot.items():
        joint_pos_pixel[name] = robot_to_pixel(x, z)

    for a, b in bones:
        cv2.line(frame, joint_pos_pixel[a], joint_pos_pixel[b], (0, 255, 0), 2)

    for name, (px, py) in joint_pos_pixel.items():
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        cv2.putText(frame, name, (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    cv2.imshow("Full Body Joint Overlay", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()