import yarp
import time
import numpy as np

from metacub_dashboard.interfaces.stream_data import StreamInterface, StreamData


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


class EncodersInterface(StreamInterface):
    def __init__(
        self,
        remote_prefix,
        local_prefix,
        control_boards: list[str],
        concatenate: bool = False,
        stream_name: str = "encoders"
    ):
        super().__init__()
        assert (
            remote_prefix.startswith("/")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        self.encoders = {}
        self.concatenate = concatenate
        self.stream_name = stream_name

        # Connect control boards 
        for board in control_boards:
            port = yarp.BufferedPortVector()
            port.open(f"{local_prefix}/{board}/state:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/{board}/state:o", f"{local_prefix}/{board}/state:i", 'tcp'
            ):
                print(f"Waiting for {remote_prefix}/{board}/state:o port to connect...")
                time.sleep(0.1)
            self.encoders[board] = port

        # Test connection and establish frequency baseline
        self._initial_read()

    def _initial_read(self):
        """Perform initial reads to establish frequency baseline."""
        print(f"Establishing frequency baseline for {self.stream_name}...")
        
        # Perform 2-3 reads to establish frequency
        for i in range(3):
            try:
                self.read()
                time.sleep(0.1)  # Small delay between reads
            except Exception as e:
                print(f"Warning: Initial read {i+1} failed: {e}")
                continue
        
        print(f"Frequency baseline established for {self.stream_name}")

    def read(self) -> list[StreamData]:
        """Read encoder data from all boards and return separate StreamData for each."""
        encoder_streams = []

        # Read from each port separately to get individual metadata
        for board_name, port in self.encoders.items():
            stamp = yarp.Stamp()
            
            s = yarp.now()
            read_attempts = 0
            while (bottle := port.read(False)) is None:
                read_attempts += 1
            
            read_time = yarp.now()
            read_delay = read_time - s
            port.getEnvelope(stamp)
            
            # Process board data
            data = np.array([bottle[i] for i in range(bottle.length())], dtype=np.float64)
            board_data = {
                board_name: {
                    'values': data,
                    'labels': board_joints[board_name]
                }
            }
            
            # Create metadata for this specific board
            metadata = self._create_metadata(
                stream_name=f"{self.stream_name}_{board_name}",
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts
            )

            # Create StreamData for this board
            stream_data = StreamData(
                name=f"{self.stream_name}_{board_name}",  # e.g., "robot_joints_head"
                data=board_data,
                metadata=metadata,
                stream_type="encoders"
            )
            
            encoder_streams.append(stream_data)

        if self.concatenate:
            # If concatenate is True, return a single StreamData with all boards combined
            # but still preserve individual metadata in separate streams
            concatenated_values = []
            concatenated_labels = []
            combined_data = {}
            
            for stream in encoder_streams:
                board_name = list(stream.data.keys())[0]
                board_info = stream.data[board_name]
                concatenated_values.extend(board_info['values'])
                concatenated_labels.extend(board_info['labels'])
                combined_data[board_name] = board_info
            
            # Return both individual streams and concatenated stream
            concatenated_stream = StreamData(
                name=f"{self.stream_name}_concatenated",
                data={
                    'concatenated': {
                        'values': np.array(concatenated_values),
                        'labels': concatenated_labels
                    },
                    **combined_data  # Also include individual board data
                },
                metadata=encoder_streams[0].metadata,  # Use first board's metadata as representative
                stream_type="encoders"
            )
            return encoder_streams + [concatenated_stream]

        return encoder_streams

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
            stream_data = encoders.read()
            print(f"Encoders frequency: {stream_data.metadata.frequency} Hz")
            print(f"Available boards: {stream_data.get_data_names()}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        encoders.close()
        yarp.Network.fini()  # Clean up YARP network
