import time


class ObservationReader:
    def __init__(self, streams={}, frequency=20, blocking=False):
        self.streams = streams
        self.frequency = frequency
        self.blocking = blocking
        self._next_read_time = +(1.0 / self.frequency)

        # Perform initial reads to set up stream frequencies
        for _ in range(10):
            self._read_from_streams()
            time.sleep(1 / self.frequency)

        # Verify desired frequency is feasible
        initial_data = self._read_from_streams()
        frequencies = [
            packet.freq
            for stream_data in initial_data.values()
            for packet in (
                stream_data.values() if isinstance(stream_data, dict) else stream_data
            )
            if packet and packet.freq
        ]
        assert frequency < min(frequencies), (
            "Observation frequency must be lower than the slowest stream frequency"
        )

    def _read_from_streams(self):
        """Read data from all configured streams"""
        return {name: reader.read() for name, reader in self.streams.items()}

    def read(self):
        """Read observations at configured frequency"""
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

    def close(self):
        """Close all stream connections"""
        for reader in self.streams.values():
            reader.close()


def test_observation_reader():
    from uuid import uuid4
    import yarp
    from metacub_dashboard.interfaces.interfaces import CameraInterface
    from metacub_dashboard.interfaces.interfaces import EncodersInterface

    yarp.Network.init()

    id = uuid4()

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
            ),
        },
        frequency=20,
        blocking=False,
    )

    counter = 0
    prev_time = None
    while True:
        obs = obs_reader.read()

        if obs is not None:
            if prev_time is not None:
                print(f"Freq: {1 / (time.perf_counter() - prev_time)}")
            counter += 1
            prev_time = time.perf_counter()

        if counter >= 40:
            break

    obs_reader.close()
