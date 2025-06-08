"""
Refactored ObservationReader that works with StreamData and provides
a clean interface for reading multiple streams synchronously.
"""
import time
from typing import Dict, List, Optional
from ..stream_data import StreamData, StreamInterface


class ObservationReader:
    """
    Synchronous reader for multiple StreamInterface objects.
    Returns a list of StreamData objects at a controlled frequency.
    """
    
    def __init__(self, streams: Dict[str, StreamInterface], frequency: float = 20, blocking: bool = False):
        """
        Args:
            streams: Dictionary mapping stream names to StreamInterface objects
            frequency: Target read frequency in Hz
            blocking: Whether to block until target frequency is met
        """
        self.streams = streams
        self.frequency = frequency
        self.blocking = blocking
        self._next_read_time = time.perf_counter() + (1.0 / self.frequency)

        # Perform initial reads to set up stream frequencies
        print("Initializing observation reader...")
        for _ in range(10):
            self._read_from_streams()
            time.sleep(1 / self.frequency)

        # Verify desired frequency is feasible
        initial_data = self._read_from_streams()
        frequencies = [
            stream_data.metadata.frequency
            for stream_data in initial_data
            if stream_data.metadata.frequency is not None
        ]
        
        if frequencies:
            min_freq = min(frequencies)
            if frequency >= min_freq:
                print(f"Warning: Observation frequency ({frequency} Hz) is close to or exceeds "
                      f"minimum stream frequency ({min_freq} Hz). Consider reducing observation frequency.")

    def _read_from_streams(self) -> List[StreamData]:
        """Read data from all configured streams."""
        stream_data_list = []
        for name, reader in self.streams.items():
            try:
                stream_data = reader.read()
                # Ensure the stream has the expected name
                if stream_data.name != name:
                    stream_data.name = name
                stream_data_list.append(stream_data)
            except Exception as e:
                print(f"Error reading from stream {name}: {e}")
                
        return stream_data_list

    def read(self) -> Optional[List[StreamData]]:
        """
        Read observations at configured frequency.
        
        Returns:
            List of StreamData objects, one from each stream, or None if non-blocking and too early
        """
        current_time = time.perf_counter()
        time_to_wait = self._next_read_time - current_time

        # Handle non-blocking mode if too early
        if time_to_wait > 0:
            if not self.blocking:
                return None
            time.sleep(time_to_wait)

        # Read from all streams and schedule next read
        data = self._read_from_streams()
        self._next_read_time = time.perf_counter() + (1.0 / self.frequency)
        return data

    def get_stream_by_name(self, stream_data_list: List[StreamData], name: str) -> Optional[StreamData]:
        """Helper to find a specific stream by name."""
        for stream_data in stream_data_list:
            if stream_data.name == name:
                return stream_data
        return None

    def get_streams_by_type(self, stream_data_list: List[StreamData], stream_type: str) -> List[StreamData]:
        """Helper to find all streams of a specific type."""
        return [stream_data for stream_data in stream_data_list if stream_data.stream_type == stream_type]

    def close(self):
        """Close all stream connections."""
        for reader in self.streams.values():
            reader.close()


def test_observation_reader():
    """Test the new ObservationReader with StreamData."""
    from uuid import uuid4
    import yarp
    from metacub_dashboard.interfaces.camera_interface import CameraInterface
    from metacub_dashboard.interfaces.encoders_interface import EncodersInterface

    yarp.Network.init()

    id = uuid4()

    obs_reader = ObservationReader(
        {
            "camera": CameraInterface(
                remote_prefix="/ergocubSim",
                local_prefix=f"/metacub_dashboard/{id}",
                rgb_shape=(640, 480),
                depth_shape=(640, 480),
                stream_name="main_camera"
            ),
            "encoders": EncodersInterface(
                remote_prefix="/ergocubSim",
                local_prefix=f"/metacub_dashboard/{id}",
                control_boards=["head", "left_arm", "right_arm", "torso"],
                stream_name="robot_encoders"
            ),
        },
        frequency=20,
        blocking=False,
    )

    counter = 0
    prev_time = None
    
    try:
        while counter < 40:
            stream_data_list = obs_reader.read()

            if stream_data_list is not None:
                if prev_time is not None:
                    print(f"Collection freq: {1 / (time.perf_counter() - prev_time):.1f} Hz")
                
                # Print stream info
                for stream_data in stream_data_list:
                    freq = stream_data.metadata.frequency or 0.0
                    print(f"  {stream_data.name} ({stream_data.stream_type}): {freq:.1f} Hz")
                
                counter += 1
                prev_time = time.perf_counter()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        obs_reader.close()
        yarp.Network.fini()


if __name__ == "__main__":
    test_observation_reader()
