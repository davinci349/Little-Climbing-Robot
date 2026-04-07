import mujoco
import mujoco.viewer
import time
import math

model = mujoco.MjModel.from_xml_path("models/XS-Robot(Alex).xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for j in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        qid = model.jnt_qposadr[j]
        qmin, qmax = model.jnt_range[j]

        print("\nTesting:", joint_name)
        print("Range (deg):", qmin * 180 / math.pi, "to", qmax * 180 / math.pi)

        # 最小角
        data.qpos[qid] = qmin
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.5)

        # 中間角
        data.qpos[qid] = (qmin + qmax) / 2
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.5)

        # 最大角
        data.qpos[qid] = qmax
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.5)

        # 回中立
        data.qpos[qid] = 0
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0)