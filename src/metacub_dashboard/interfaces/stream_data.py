"""
Clean data structures for streaming interfaces.
Separates stream metadata from actual data content.
"""
from dataclasses import dataclass, field
from typing import Any, Self, Dict, List, Generator, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class StreamMetadata:
    """Diagnostic information for a stream read operation."""
    timestamp: float
    seq_number: int
    read_timestamp: float
    read_delay: float
    read_attempts: int
    frequency: float | None = None
    missed_packets: int | None = None
    
    def compute_frequency(self, prev: Self | None):
        """Update frequency based on previous metadata."""
        if prev is None:
            self.frequency = None
            return
        
        dt = self.timestamp - prev.timestamp
        if dt > 0:
            self.frequency = (self.seq_number - prev.seq_number) / dt
        else:
            self.frequency = None
    
    def compute_missed_packets(self, prev: Self | None):
        """Compute number of missed packets based on sequence number gaps."""
        if prev is None:
            self.missed_packets = None
            return
        
        expected_seq = prev.seq_number + 1
        actual_seq = self.seq_number
        self.missed_packets = max(0, actual_seq - expected_seq)


@dataclass 
class Pose:
    """Represents a pose with position, orientation, and grip state."""
    pos: np.ndarray | None = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    ori: np.ndarray | None = field(default_factory=lambda: np.identity(3))
    grip: np.ndarray | None = field(default_factory=lambda: np.array([0.0]))
    
    def to_numpy(self) -> np.ndarray:
        """Convert to concatenated numpy array."""
        pos_array = self.pos.flatten() if self.pos is not None else np.array([0.0, 0.0, 0.0])
        ori_array = self.ori.flatten() if self.ori is not None else np.identity(3).flatten()
        grip_array = self.grip.flatten() if self.grip is not None else np.array([0.0])
        return np.concatenate([pos_array, ori_array, grip_array])


@dataclass
class StreamData:
    """
    Container for stream data with metadata.
    Each stream read returns one of these, containing all related data.
    """
    name: str  # Name/identifier for this stream (e.g., "encoders", "camera", "action")
    data: Dict[str, Any]  # The actual data keyed by component name
    metadata: StreamMetadata  # Stream diagnostic info
    stream_type: str  # "encoders", "camera", "action"
    entity_path: str = ""  # Where to log this stream in the visualizer (e.g., "poses", "joints", "palm_path")
    
    def get_data_by_name(self, name: str) -> Any:
        """Get specific data component by name."""
        return self.data.get(name)
    
    def get_data_names(self) -> List[str]:
        """Get all available data component names."""
        return list(self.data.keys())

    @staticmethod
    def get_streams_by_type(streams: List['StreamData'], stream_type: str) -> List['StreamData']:
        """Filter streams by type."""
        return [stream for stream in streams if stream.stream_type == stream_type]
    
    @classmethod
    def create_processor(cls) -> 'StreamProcessor':
        """Create a new stream processor for applying rules."""
        return StreamProcessor()


class StreamProcessor:
    """
    Simple processor that applies rules to streams using generators for efficiency.
    Rules are functions that modify streams in-place.
    """
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, condition: Callable[['StreamData'], bool], action: Callable[['StreamData'], None]) -> 'StreamProcessor':
        """
        Add a rule that applies an action to streams matching a condition.
        
        Args:
            condition: Function that takes a StreamData and returns True if rule should apply
            action: Function that modifies the StreamData in-place
            
        Returns:
            Self for method chaining
        """
        self.rules.append((condition, action))
        return self
    
    def process_streams(self, streams: List['StreamData']) -> Generator['StreamData', None, None]:
        """
        Process streams by applying rules without filtering by type.
        
        Args:
            streams: List of StreamData objects to process
            
        Yields:
            StreamData objects with rules applied
        """
        for stream in streams:
            # Apply all matching rules
            for condition, action in self.rules:
                if condition(stream):
                    action(stream)
            
            yield stream


class StreamInterface(ABC):
    """Base class for all streaming interfaces."""
    
    def __init__(self):
        self.prev_metadata: Dict[str, StreamMetadata] = {}  # Track metadata per stream/board
    
    @abstractmethod
    def read(self) -> List[StreamData]:
        """Read data from the stream. Always returns a list of StreamData objects, one per logical stream."""
        pass
    
    @abstractmethod 
    def close(self):
        """Close the stream."""
        pass
    
    def _create_metadata(self, stream_name: str, timestamp: float, seq_number: int, 
                        read_timestamp: float, read_delay: float, 
                        read_attempts: int) -> StreamMetadata:
        """Helper to create metadata and compute frequency for a specific stream."""
        metadata = StreamMetadata(
            timestamp=timestamp,
            seq_number=seq_number, 
            read_timestamp=read_timestamp,
            read_delay=read_delay,
            read_attempts=read_attempts
        )
        
        # Get previous metadata for this specific stream/board
        prev_metadata = self.prev_metadata.get(stream_name)
        metadata.compute_frequency(prev_metadata)
        metadata.compute_missed_packets(prev_metadata)
        
        # Store current metadata for next time
        self.prev_metadata[stream_name] = metadata
        return metadata
