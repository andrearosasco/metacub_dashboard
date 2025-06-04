import yarp
import time
import numpy as np

from interfaces.data_packet import DataPacket


class CameraInterface:
    def __init__(self, remote_prefix, local_prefix, rgb_shape=None, depth_shape=None, concatenate=False):
        assert rgb_shape or depth_shape, "At least one of rgb or depth must be True"
        assert (
            remote_prefix.startswith("/")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.concatenate = concatenate

        if rgb_shape:
            self.rgb_port = yarp.BufferedPortImageRgb()
            self.rgb_buffer = bytearray(
                np.zeros((rgb_shape[1], rgb_shape[0], 3), dtype=np.uint8)
            )
            self.yarp_rgb_image = yarp.ImageRgb()
            self.yarp_rgb_image.resize(*(rgb_shape))
            self.yarp_rgb_image.setExternal(self.rgb_buffer, *rgb_shape)
            self.rgb_port.open(f"{local_prefix}/rgb:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/depthCamera/rgbImage:o",
                f"{local_prefix}/rgb:i",
                "mjpeg",
            ):
                print("Waiting for RGB port to connect...")
                time.sleep(0.1)

        if depth_shape:
            self.depth_port = yarp.BufferedPortImageFloat()
            self.depth_buffer = bytearray(
                np.zeros((depth_shape[1], depth_shape[0]), dtype=np.float32)
            )
            self.yarp_depth_image = yarp.ImageFloat()
            self.yarp_depth_image.resize(*depth_shape)
            self.yarp_depth_image.setExternal(self.depth_buffer, *depth_shape)
            self.depth_port.open(f"{local_prefix}/depth:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/depthCamera/depthImage:o",
                f"{local_prefix}/depth:i",
                "fast_tcp+send.portmonitor+file.bottle_compression_zlib+recv.portmonitor+file.bottle_compression_zlib+type.dll",
            ):
                print("Waiting for Depth port to connect...")
                time.sleep(0.1)

        self.prev_packets = {"rgb": None, "depth": None}
        self.read()

    def read(self):
        packets = {}
        rgb_stamp, depth_stamp = yarp.Stamp(), yarp.Stamp()

        if self.rgb_shape:
            s = yarp.now()
            read_attempts = 0
            while (data := self.rgb_port.read(False)) is None:
                read_attempts += 1

            read_time = yarp.now()
            read_delay = read_time - s
            self.rgb_port.getEnvelope(rgb_stamp)

            self.yarp_rgb_image.copy(data)
            rgb_image = np.frombuffer(self.rgb_buffer, dtype=np.uint8).reshape(
                self.rgb_shape[1], self.rgb_shape[0], 3
            )

            rgb_packet = DataPacket(
                name="agentview_rgb",
                data=rgb_image,
                data_type="rgb",
                timestamp=rgb_stamp.getTime(),
                seq_number=rgb_stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
            )
            rgb_packet.compute_frequency(self.prev_packets["rgb"])
            self.prev_packets["rgb"] = rgb_packet
            packets["rgb"] = rgb_packet

        if self.depth_shape:
            s = yarp.now()
            read_attempts = 0
            while (data := self.depth_port.read(False)) is None:
                read_attempts += 1
            read_time = yarp.now()
            read_delay = read_time - s

            self.depth_port.getEnvelope(depth_stamp)
            self.yarp_depth_image.copy(data)
            depth_image = (
                np.frombuffer(self.depth_buffer, dtype=np.float32).reshape(
                    self.depth_shape[1], self.depth_shape[0]
                )
                * 1000
            ).astype(np.uint16)

            depth_packet = DataPacket(
                name="agentview_depth",
                data=depth_image,
                data_type="depth",
                timestamp=depth_stamp.getTime(),
                seq_number=depth_stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
            )
            depth_packet.compute_frequency(self.prev_packets["depth"])
            self.prev_packets["depth"] = depth_packet
            packets["depth"] = depth_packet

        if self.concatenate:
            return list(packets.values())

        return packets

    def close(self):
        if self.rgb_shape:
            self.rgb_port.close()
        if self.depth_shape:
            self.depth_port.close()


def test_camera_interface():
    from uuid import uuid4

    yarp.Network.init()

    camera_interface = CameraInterface(
        remote_prefix="/ergocubSim",
        local_prefix=f"/metacub_dashboard/{uuid4()}",
        rgb_shape=(640, 480),
        depth_shape=(640, 480),
    )

    for _ in range(30):
        data = camera_interface.read()
        if data["rgb"] is not None:
            print(f"RGB Image freq: {data['rgb'].freq}")
        if data["depth"] is not None:
            print(f"Depth Image freq: {data['depth'].freq}")
        time.sleep(1 / 20)

    camera_interface.close()
