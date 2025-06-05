import yarp
import time

from metacub_dashboard.interfaces.data_packet import DataPacket, Pose
from scipy.spatial.transform import Rotation as R
import numpy as np



class ActionInterface:
    action_fmt = {
        'neck': [],
        'left_arm': [],
        'right_arm': [],
        'fingers': ['l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
                    'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky']
    }

    def __init__(self, remote_prefix, local_prefix):
        assert (
            remote_prefix.startswith("/")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        format = {'neck': ['float']*9, 'left_arm': ['float']*7,'right_arm': ['float']*7,'fingers': [['float']*3]*10}

        self.format = format
        self.port = yarp.BufferedPortBottle()
        self.port.open(f'{local_prefix}/action:i')
        while not yarp.Network.connect(f'{remote_prefix}/action:o', f'{local_prefix}/action:i'): 
            time.sleep(0.1)

        self.prev_freq_packet = None

    def read(self):
        action_packets = []
        stamp = yarp.Stamp()

        s = yarp.now()
        read_attempts = 0
        while (bottle := self.port.read(False)) is None:
            read_attempts += 1

        read_time = yarp.now()
        read_delay = read_time - s

        self.port.getEnvelope(stamp)
        data = self.cast_bottle(bottle, self.format)
        data = self.convert_poses(data)

        for k, v in data.items():
            action_packet = DataPacket(
                name=k,
                data=v,
                data_type="pose",
            )
            action_packets.append(action_packet)
        action_freq = DataPacket(
            name='action',
            timestamp=stamp.getTime(),
            seq_number=stamp.getCount(),
            read_timestamp=read_time,
            read_delay=read_delay,
            read_attempts=read_attempts,
        )
        action_freq.compute_frequency(self.prev_freq_packet)
        action_packets.append(action_freq)
        self.prev_freq_packet = action_freq
        return action_packets

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
        data_out = {}
        for key, value in data.items():
            assert value is not None 
        data_out['neck'] = Pose(ori=np.array(data['neck']).reshape(3,3))
        data_out['left_arm'] = Pose(pos=np.array(data['left_arm'][:3]), ori=R.from_quat(data['left_arm'][3:]).as_matrix())
        data_out['right_arm'] = Pose(pos=np.array(data['right_arm'][:3]), ori=R.from_quat(data['right_arm'][3:]).as_matrix())
        data_out.update({name: Pose(pos=np.array(finger)) for name, finger in zip(ActionInterface.action_fmt['fingers'], data['fingers'])})

        return data_out
        