import yarp
import time
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation as R

from metacub_dashboard.interfaces.stream_data import StreamInterface, StreamData, Pose


class ActionInterface(StreamInterface):
    action_fmt = {
        'neck': [],
        'left_arm': [],
        'right_arm': [],
        'fingers': ['l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
                    'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky']
    }

    def __init__(self, remote_prefix, local_prefix, stream_name="poses"):
        super().__init__()
        assert (
            remote_prefix.startswith("/")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        self.stream_name = stream_name
        self.format = {
            'neck': ['float'] * 9, 
            'left_arm': ['float'] * 7,
            'right_arm': ['float'] * 7,
            'fingers': [['float'] * 3] * 10
        }
        
        self.port = yarp.BufferedPortBottle()
        self.port.open(f'{local_prefix}/action:i')
        while not yarp.Network.connect(f'{remote_prefix}/action:o', f'{local_prefix}/action:i'): 
            time.sleep(0.1)
        
        # Perform initial read to establish frequency baseline
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

    def read(self) -> List[StreamData]:
        stamp = yarp.Stamp()

        s = yarp.now()
        read_attempts = 0
        while (bottle := self.port.read(False)) is None:
            read_attempts += 1

        read_time = yarp.now()
        read_delay = read_time - s

        self.port.getEnvelope(stamp)
        
        # Parse bottle data
        raw_data = self.cast_bottle(bottle, self.format)
        pose_data = self.convert_poses(raw_data)
        
        # Create metadata
        metadata = self._create_metadata(
            stream_name=self.stream_name,
            timestamp=stamp.getTime(),
            seq_number=stamp.getCount(),
            read_timestamp=read_time,
            read_delay=read_delay,
            read_attempts=read_attempts
        )

        return [StreamData(
            name=self.stream_name,
            data=pose_data,
            metadata=metadata,
            stream_type="poses"
        )]

    def cast_bottle(self, bottle, format, name=''):
        if isinstance(format, dict):
            return {key: self.cast_bottle(bottle.find(key), format[key], key) for key in format}
        elif isinstance(format, list):
            if bottle.asList().size() == 0:
                return None
            return [self.cast_bottle(bottle.asList().get(i), format[i]) for i in range(len(format))]
        elif isinstance(format, str):
            if format == 'int':
                return bottle.asInt()
            elif format == 'float':
                return bottle.asFloat64()
            elif format == 'string':
                return bottle.asString()
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def convert_poses(self, data):
        """Convert raw data to Pose objects."""
        pose_data = {}
        
        for key, value in data.items():
            assert value is not None
            
        pose_data['neck'] = Pose(ori=np.array(data['neck']).reshape(3, 3))
        pose_data['left_arm'] = Pose(
            pos=np.array(data['left_arm'][:3]), 
            ori=R.from_quat(data['left_arm'][3:]).as_matrix()
        )
        pose_data['right_arm'] = Pose(
            pos=np.array(data['right_arm'][:3]), 
            ori=R.from_quat(data['right_arm'][3:]).as_matrix()
        )
        
        # Add finger poses
        for name, finger in zip(ActionInterface.action_fmt['fingers'], data['fingers']):
            pose_data[name] = Pose(pos=np.array(finger))

        return pose_data

    def close(self):
        """Close the action interface."""
        self.port.close()


def test_action_interface():
    from uuid import uuid4

    yarp.Network.init()

    action_reader = ActionInterface(
        remote_prefix="/metaControllClient",
        local_prefix=f"/metacub_dashboard/{uuid4()}",
        stream_name="target_poses"
    )

    try:
        for _ in range(30):
            stream_data = action_reader.read()
            print(f"Poses frequency: {stream_data.metadata.frequency} Hz")
            print(f"Available poses: {stream_data.get_data_names()}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        action_reader.close()
        yarp.Network.fini()


if __name__ == "__main__":
    test_action_interface()
