from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

@dataclass
class DataPacket:
    name: str
    data: Any
    data_labels: list[str]|None=None
    data_type: str = "unknown"
    timestamp: float|None=None
    seq_number: int|None=None
    freq:float|None=None

    read_timestamp: float|None=None
    read_delay:float|None=None
    read_attempts:int|None=None

    def compute_frequency(self, prev: Self | None):
        if prev is None:
            return

        self.freq = ((self.seq_number - prev.seq_number) /
                     (self.timestamp - prev.timestamp))


@dataclass
class Pose:
    """
    A dataclass to represent pose with position, orientation, and grip state.

    Attributes:
        pos (np.ndarray | None): Position as a NumPy array. Defaults to [0, 0, 0].
        ori (np.ndarray | None): Orientation as a NumPy array (identity matrix). Defaults to a 3x3 identity matrix.
        grip (np.ndarray | None): Grip state as a NumPy array. Defaults to [0].
    """
    pos: np.ndarray | None = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    ori: np.ndarray | None = field(default_factory=lambda: np.identity(3))
    grip: np.ndarray | None = field(default_factory=lambda: np.array([0.0]))