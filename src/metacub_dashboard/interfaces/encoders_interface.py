import yarp
import time
import numpy as np
from collections import defaultdict

from metacub_dashboard.interfaces.data_packet import DataPacket


board_joints = {
    "left_arm": [
        'l_shoulder_pitch',
        'l_shoulder_roll',
        'l_shoulder_yaw',
        'l_elbow',
        'l_wrist_yaw',
        'l_wrist_roll',
        'l_wrist_pitch',
        "l_thumb_add",
        "l_thumb_prox",
        "l_index_add",
        "l_index_prox",
        "l_middle_prox",
        "l_ring_prox",
    ],
    "right_arm": [
        'r_shoulder_pitch',
        'r_shoulder_roll',
        'r_shoulder_yaw',
        'r_elbow',
        'r_wrist_yaw',
        'r_wrist_roll',
        'r_wrist_pitch',
        "r_thumb_add",
        "r_thumb_prox",
        "r_index_add",
        "r_index_prox",
        "r_middle_prox",
        "r_ring_prox",
    ],
    "torso": ["torso_roll", "torso_pitch", "torso_yaw"],
    "head": ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt"],
    "left_leg": [
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
    ],
    "right_leg": [
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
    ],
}


class EncodersInterface:
    def __init__(
        self,
        remote_prefix,
        local_prefix,
        control_boards: list[str],
        concatenate: bool = False,
    ):
        assert (
            remote_prefix.startswith("/")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        self.encoders = {}
        self.concatenate = concatenate

        for board in control_boards:
            port = yarp.BufferedPortVector()
            port.open(f"{local_prefix}/{board}/state:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/{board}/state:o", f"{local_prefix}/{board}/state:i", 'tcp'
            ):
                print(f"Waiting for {remote_prefix}/{board}/state:o port to connect...")
                time.sleep(0.1)
            self.encoders[board] = port

        self.prev_encoder_packet = defaultdict(lambda: None)
        self.read()

    def read(self):
        stamp = yarp.Stamp()
        encoder_package = {}

        for name, port in self.encoders.items():
            s = yarp.now()
            read_attempts = 0
            while (bottle := port.read(False)) is None:
                read_attempts += 1

            read_time = yarp.now()
            read_delay = read_time - s

            port.getEnvelope(stamp)

            data = np.array(
                [bottle[i] for i in range(bottle.length())], dtype=np.float64
            )
            encoder_package[name] = DataPacket(
                name=name,
                data=data,
                data_labels=board_joints[name],
                data_type="encoders",
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
            )
            encoder_package[name].compute_frequency(self.prev_encoder_packet[name])
            self.prev_encoder_packet[name] = encoder_package[name]

        if self.concatenate:
            return list(encoder_package.values())

        return encoder_package

    def close(self):
        for port in self.encoders.values():
            port.close()
        self.encoders.clear()


def test_encoder_interface():
    from uuid import uuid4

    yarp.Network.init()

    encoders = EncodersInterface(
        remote_prefix="/ergocubSim",
        local_prefix=f"/metacub_dashboard/{uuid4()}",
        control_boards=["head", "left_arm", "right_arm", "torso"],
    )

    try:
        for _ in range(30):
            data = encoders.read()
            for board, packet in data.items():
                print(f"{board} frequency: {packet.freq} Hz")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        encoders.close()
        yarp.Network.fini()  # Clean up YARP network
