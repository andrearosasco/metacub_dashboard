import time
from uuid import uuid4
import numpy as np
from metacub_dashboard.interfaces.action_interface import ActionInterface
from metacub_dashboard.interfaces.camera_interface import CameraInterface
from metacub_dashboard.interfaces.encoders_interface import EncodersInterface
from metacub_dashboard.interfaces.utils.observation_reader import ObservationReader
from metacub_dashboard.visualizer.visualizer import Visualizer, Camera
from metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from metacub_dashboard.data_logger.data_logger import DataLogger, flatten_dict
import os

os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp
import pickle
from scipy.spatial.transform import Rotation as R
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core

urdf_parser_core.on_error = lambda x: x


def main():
    # Initialize Streams
    yarp.Network.init()

    id = uuid4()

    action_reader = ActionInterface(
        remote_prefix="/metaControllClient",
        local_prefix=f"/metacub_dashboard/{id}",
    )

    obs_reader = ObservationReader(
        {
            "camera": CameraInterface(
                remote_prefix="/ergocubSim",
                local_prefix=f"/metacub_dashboard/{id}",
                rgb_shape=(640, 480),
                depth_shape=(640, 480),
            ),
            "encoders": EncodersInterface(
                remote_prefix="/ergocubSim",
                local_prefix=f"/metacub_dashboard/{id}",
                control_boards=["head", "left_arm", "right_arm", "torso"],
                concatenate=True,
            ),
        },
        frequency=60,
        blocking=False,
    )

    # Create Visualizer
    urdf_path = yarp.ResourceFinder().findFileByName("model.urdf")
    urdf = urdf_parser.URDF.from_xml_file(urdf_path)
    urdf.path = urdf_path

    camera_path = "/".join(
        urdf.get_chain(root=urdf.get_root(), tip="realsense_depth_frame")[0::2]
    )

    image_paths = [
            f"{camera_path}/cameras/agentview_rgb",
            f"{camera_path}/cameras/agentview_depth",
    ]

    eef_paths = [
            "/".join(urdf.get_chain(root=urdf.get_root(), tip=eef)[0::2])
            for eef in ["l_hand_palm", "r_hand_palm"]
    ]

    blueprint = build_blueprint(
        image_paths=image_paths,
        eef_paths=eef_paths,
        poses=["proprio", "action"],
    )

    visualizer = Visualizer(urdf=urdf, blueprint=blueprint, gradio=False)

    visualizer.log(
        cameras={
            f"{camera_path}/cameras/agentview_depth": Camera(
                fov=45.0,
                width=240,
                height=240,
                pos=np.array([np.array([0, 0, 0])]),
                ori=R.from_rotvec([0, 0, 0]).as_matrix(),
            ),
        },
    )

    data_logger = DataLogger(
        path="assets/dataset.zarr.zip", flush_every=200, exist_ok=True
    )
    # Connect Ports
    i = 0

    prev_seq_number = None
    prev_t = time.perf_counter()
    while True:
        # Read data from YARP ports
        action = action_reader.read()

        if 

        time.sleep(1/10)
        obs = obs_reader.read()
        
        def action_fn():
            for p in action:
                if p.name in ActionInterface.action_fmt['fingers']:
                    p.name = f'{eef_paths[0] if p.name.startswith('l_') else eef_paths[1]}/target/{p.name}'
                else:
                    p.name = f'target/{p.name}'

                yield p

        # Visualize
        visualizer.log(
            joints=obs['encoders'],
            images={f"{camera_path}/cameras/agentview_rgb": obs["camera"]['rgb']},
            depths={f"{camera_path}/cameras/agentview_depth": obs["camera"]['depth']},
            poses=action_fn(),
            _time=time.perf_counter(),
            timestamp=i * 0.1,
            static=False,
        )
        # Save
        data_logger.log({pkt.name: pkt.data.numpy() for pkt in flatten_dict(obs)}, {pkt.name: pkt.data.numpy() for pkt in action})
        print(1 / (time.perf_counter() - prev_t))
        prev_t = time.perf_counter()

        i += 1
        # time.sleep(0.05)


if __name__ == "__main__":
    main()
