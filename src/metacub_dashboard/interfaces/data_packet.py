from dataclasses import dataclass
from typing import Any, Self

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