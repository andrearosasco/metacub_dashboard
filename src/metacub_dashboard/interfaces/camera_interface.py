import yarp
import time
import numpy as np
from typing import List

from metacub_dashboard.interfaces.stream_data import StreamInterface, StreamData


class CameraInterface(StreamInterface):
    def __init__(self, remote_prefix, local_prefix, rgb_shape=None, depth_shape=None, stream_name="camera"):
        super().__init__()
        assert rgb_shape or depth_shape, "At least one of rgb or depth must be True"
        assert (
            (remote_prefix.startswith("/") or remote_prefix == "")
            and not remote_prefix.endswith("/")
            and local_prefix.startswith("/")
            and not local_prefix.endswith("/")
        ), "Both prefixes must start with '/' and not end with '/'"

        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.stream_name = stream_name

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

    def read(self) -> List[StreamData]:
        camera_streams = []

        if self.rgb_shape:
            stamp = yarp.Stamp()
            s = yarp.now()
            read_attempts = 0
            while (data := self.rgb_port.read(False)) is None:
                read_attempts += 1

            read_time = yarp.now()
            read_delay = read_time - s
            self.rgb_port.getEnvelope(stamp)

            self.yarp_rgb_image.copy(data)
            rgb_image = np.frombuffer(self.rgb_buffer, dtype=np.uint8).reshape(
                self.rgb_shape[1], self.rgb_shape[0], 3
            )

            # Create metadata for RGB stream
            rgb_metadata = self._create_metadata(
                stream_name=f"{self.stream_name}_rgb",
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts
            )

            rgb_stream = StreamData(
                name=f"{self.stream_name}_rgb",
                data={"rgb": rgb_image},
                metadata=rgb_metadata,
                stream_type="camera"
            )
            camera_streams.append(rgb_stream)

        if self.depth_shape:
            stamp = yarp.Stamp()
            s = yarp.now()
            read_attempts = 0
            while (data := self.depth_port.read(False)) is None:
                read_attempts += 1
            
            read_time = yarp.now()
            read_delay = read_time - s
            self.depth_port.getEnvelope(stamp)
            
            self.yarp_depth_image.copy(data)
            depth_image = (
                np.frombuffer(self.depth_buffer, dtype=np.float32).reshape(
                    self.depth_shape[1], self.depth_shape[0]
                )
                * 1000
            ).astype(np.uint16)

            # Create metadata for depth stream
            depth_metadata = self._create_metadata(
                stream_name=f"{self.stream_name}_depth",
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts
            )

            depth_stream = StreamData(
                name=f"{self.stream_name}_depth",
                data={"depth": depth_image},
                metadata=depth_metadata,
                stream_type="camera"
            )
            camera_streams.append(depth_stream)

        return camera_streams

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
        stream_name="test_camera"
    )

    for _ in range(30):
        stream_data_list = camera_interface.read()
        for stream_data in stream_data_list:
            print(f"Stream {stream_data.name}: frequency {stream_data.metadata.frequency} Hz")
            print(f"Available data: {stream_data.get_data_names()}")
        time.sleep(1 / 20)

    camera_interface.close()
