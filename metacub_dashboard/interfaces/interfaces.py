"""
Pure Polars-based streaming interfaces (now the default).
Uses native Polars DataFrames with schemas for type safety and powerful operations.
"""
import polars as pl
import numpy as np
import yarp
import time
from typing import Dict, List, Any
from abc import ABC, abstractmethod

# Define schemas for type safety
METADATA_SCHEMA = pl.Schema({
    "timestamp": pl.Float64,
    "seq_number": pl.Int64,
    "read_timestamp": pl.Float64,
    "read_delay": pl.Float64,
    "read_attempts": pl.Int64,
    "frequency": pl.Float64,
})

STREAM_SCHEMA = pl.Schema({
    "name": pl.String,
    "stream_type": pl.String,
    "entity_path": pl.String,
    "data": pl.Object,  # Flexible for heterogeneous data
    "metadata": pl.Struct(METADATA_SCHEMA),
})


class Interface(ABC):
    """Base class for streaming interfaces using native Polars DataFrames."""
    
    def __init__(self, stream_name: str):
        self.stream_name = stream_name
        self.prev_metadata = {}  # Track previous metadata for frequency calculation

    @abstractmethod
    def connect(self):
        """Connect YARP ports after network initialization."""
        pass

    @abstractmethod
    def read(self) -> pl.DataFrame:
        """Read data and return as Polars DataFrame with enforced schema."""
        pass

    @abstractmethod
    def close(self):
        """Close the interface."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the interface to initial state."""
        pass

    def _create_metadata_dict(self, timestamp: float, seq_number: int, 
                             read_timestamp: float, read_delay: float, 
                             read_attempts: int, metadata_key: str = None) -> Dict[str, Any]:
        """Create metadata dictionary with frequency calculation."""
        # Use provided key or fall back to stream_name
        key = metadata_key or self.stream_name
        
        # Calculate frequency based on previous metadata
        prev = self.prev_metadata.get(key)
        if prev and timestamp > prev["timestamp"]:
            dt = timestamp - prev["timestamp"]
            frequency = (seq_number - prev["seq_number"]) / dt if dt > 0 else 0.0
        else:
            frequency = 0.0

        metadata = {
            "timestamp": timestamp,
            "seq_number": seq_number,
            "read_timestamp": read_timestamp,
            "read_delay": read_delay,
            "read_attempts": read_attempts,
            "frequency": frequency,
        }
        
        self.prev_metadata[key] = metadata
        return metadata


class ActionInterface(Interface):
    """Action interface using native Polars DataFrames."""
    
    # Default format mappings for each control board type
    DEFAULT_BOARD_FORMATS = {
        'neck': ['float']*9,
        'left_arm': ['float']*7,
        'right_arm': ['float']*7,
        'fingers': [['float']*3]*10
    }
    
    def __init__(self, remote_prefix: str, local_prefix: str, control_boards: List[str] = None, stream_name: str = "poses"):
        super().__init__(stream_name)
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.control_boards = control_boards or ["head", "left_arm", "right_arm", "torso"]
        
        # Build dynamic format based on control_boards
        self.format = {board: self.DEFAULT_BOARD_FORMATS[board] for board in self.control_boards}
        
        # YARP ports will be created and opened in connect() method
        self.port = None
        self.reset_port = None

    def connect(self):
        """Connect YARP ports after network initialization."""
        # Create and open YARP ports
        self.port = yarp.BufferedPortBottle()
        self.port.open(f"{self.local_prefix}/action:i")
        
        self.reset_port = yarp.BufferedPortBottle()
        self.reset_port.open(f"{self.local_prefix}/reset:o")
        
        # Connect action port
        while not yarp.Network.connect(f"{self.remote_prefix}/action:o", f"{self.local_prefix}/action:i"):
            print(f"Waiting for {self.remote_prefix}/action:o port to connect...")
            time.sleep(0.1)
        
        # Connect reset port
        while not yarp.Network.connect(f"{self.local_prefix}/reset:o", f"{self.remote_prefix}/reset:i"):
            print(f"Waiting for {self.remote_prefix}/reset:i port to connect...")
            time.sleep(0.1)

        # self.read()

    def reset(self, blocking: bool = True):
        """Send reset signal to the action server."""
        bottle = self.reset_port.prepare()
        bottle.clear()
        bottle.addInt32(1)  # Send boolean True as integer 1
        self.reset_port.write()
        print("🔄 Reset signal sent to action server")
        if blocking:
            time.sleep(10)  # Allow time for reset to take effect
        print("✅ Reset complete, ready for new actions")

    def read(self) -> pl.DataFrame:
        """Read action data and return as Polars DataFrame."""
        stamp = yarp.Stamp()
        
        start_time = yarp.now()
        read_attempts = 0
        while (bottle := self.port.read(False)) is None or (poses_data := self.parse_bottle(bottle, self.format)) is None:
            read_attempts += 1
            
        read_time = yarp.now()
        read_delay = read_time - start_time
        self.port.getEnvelope(stamp)
        
        metadata = self._create_metadata_dict(
            timestamp=stamp.getTime(),
            seq_number=stamp.getCount(),
            read_timestamp=read_time,
            read_delay=read_delay,
            read_attempts=read_attempts,
        )

        # Create DataFrame row
        row_data = {
            "name": self.stream_name,
            "stream_type": "poses",
            "entity_path": "",  # Will be set by processing logic
            "data": poses_data,
            "metadata": metadata,
        }

        return pl.DataFrame([row_data], schema=STREAM_SCHEMA)

    def close(self):
        """Close the YARP port."""
        if self.port:
            self.port.close()
        if self.reset_port:
            self.reset_port.close()

    def parse_bottle(self, bottle, format, name=''):
        if isinstance(format, dict):
            result = {}
            for key in format:
                parsed_value = self.parse_bottle(bottle.find(key), format[key], key)
                if parsed_value is None:
                    return None
                result[key] = parsed_value
            return result
        elif isinstance(format, list):
            if bottle.asList().size() == 0:
                return None
            result = []
            for i in range(len(format)):
                parsed_value = self.parse_bottle(bottle.asList().get(i), format[i])
                if parsed_value is None:
                    return None
                result.append(parsed_value)
            return result
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


class EncodersInterface(Interface):
    """Pure Polars encoders interface."""
    
    def __init__(self, remote_prefix: str, local_prefix: str, 
                 control_boards: List[str], stream_name: str = "encoders"):
        super().__init__(stream_name)
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.control_boards = control_boards
        
        # Joint mappings
        self.board_joints = {
            "left_arm": [ 'l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'l_wrist_yaw',
                          'l_wrist_roll', 'l_wrist_pitch', "l_thumb_add", "l_thumb_prox", "l_index_add",
                            "l_index_prox", "l_middle_prox", "l_ring_prox",],
            "right_arm": [ 'r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'r_wrist_yaw',
                           'r_wrist_roll', 'r_wrist_pitch', "r_thumb_add", "r_thumb_prox", "r_index_add",
                             "r_index_prox", "r_middle_prox", "r_ring_prox",],
            "torso": ["torso_roll", "torso_pitch", "torso_yaw"],
            "head": ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt"],
            "left_leg": [ "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",],
            "right_leg": [ "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",],
        }
        
        # Initialize YARP ports for each board (but don't connect yet)
        self.encoders = {}

    def connect(self):
        """Connect YARP ports after network initialization."""
        # Create and open YARP ports for each board
        for board in self.control_boards:
            port = yarp.BufferedPortVector()
            port.open(f"{self.local_prefix}/{board}/state:i")
            self.encoders[board] = port
        
        # Connect ports
        for board, port in self.encoders.items():
            while not yarp.Network.connect(
                f"{self.remote_prefix}/{board}/state:o", f"{self.local_prefix}/{board}/state:i", 'tcp'
            ):
                print(f"Waiting for {self.remote_prefix}/{board}/state:o port to connect...")
                time.sleep(0.1)

        # self.read()

    def read(self) -> pl.DataFrame:
        """Read encoder data and return as Polars DataFrame."""
        rows = []
        
        for board_name, port in self.encoders.items():
            stamp = yarp.Stamp()
            
            start_time = yarp.now()
            read_attempts = 0
            while (bottle := port.read(False)) is None:
                read_attempts += 1
            
            read_time = yarp.now()
            read_delay = read_time - start_time
            port.getEnvelope(stamp)
            
            # Process board data
            values = np.array([bottle[i] for i in range(bottle.size())], dtype=np.float64)
            labels = self.board_joints[board_name]
            
            board_data = {
                board_name: {
                    'values': values,
                    'labels': labels
                }
            }
            
            # Create metadata
            metadata = self._create_metadata_dict(
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
                metadata_key=f"{self.stream_name}_{board_name}"
            )
            
            # Create DataFrame row
            row_data = {
                "name": f"{self.stream_name}_{board_name}",
                "stream_type": "encoders",
                "entity_path": "",  # Will be set by processing logic
                "data": board_data,
                "metadata": metadata,
            }
            
            rows.append(row_data)
        
        return pl.DataFrame(rows, schema=STREAM_SCHEMA)
    
    def reset(self):
        pass

    def close(self):
        """Close all YARP ports."""
        if self.encoders:
            for port in self.encoders.values():
                port.close()


class CameraInterface(Interface):
    """Pure Polars camera interface."""
    
    def __init__(self, remote_prefix: str, local_prefix: str, 
                 rgb_shape: tuple = None, depth_shape: tuple = None,
                 stream_name: str = "camera"):
        super().__init__(stream_name)
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        
        # Port objects will be created in connect() method
        self.rgb_port = None
        self.depth_port = None
        self.yarp_rgb_image = None
        self.yarp_depth_image = None
        self.rgb_buffer = None
        self.depth_buffer = None

    def connect(self):
        """Connect YARP ports after network initialization."""
        # Create and open RGB port if needed
        if self.rgb_shape:
            self.rgb_port = yarp.BufferedPortImageRgb()
            self.rgb_port.open(f"{self.local_prefix}/rgb:i")
            
            self.yarp_rgb_image = yarp.ImageRgb()
            self.yarp_rgb_image.resize(self.rgb_shape[0], self.rgb_shape[1])
            self.rgb_buffer = bytearray(self.rgb_shape[0] * self.rgb_shape[1] * 3)
            self.yarp_rgb_image.setExternal(self.rgb_buffer, self.rgb_shape[0], self.rgb_shape[1])
            
            while not yarp.Network.connect(
                f"{self.remote_prefix}/depthCamera/rgbImage:o",
                f"{self.local_prefix}/rgb:i",
                "mjpeg",
            ):
                print("Waiting for RGB port to connect...")
                time.sleep(0.1)
        
        # Create and open depth port if needed
        if self.depth_shape:
            self.depth_port = yarp.BufferedPortImageFloat()
            self.depth_port.open(f"{self.local_prefix}/depth:i")
            
            self.yarp_depth_image = yarp.ImageFloat()
            self.yarp_depth_image.resize(self.depth_shape[0], self.depth_shape[1])
            self.depth_buffer = bytearray(self.depth_shape[0] * self.depth_shape[1] * 4)
            self.yarp_depth_image.setExternal(self.depth_buffer, self.depth_shape[0], self.depth_shape[1])
            
            while not yarp.Network.connect(
                f"{self.remote_prefix}/depthCamera/depthImage:o",
                f"{self.local_prefix}/depth:i",
                "fast_tcp+send.portmonitor+file.bottle_compression_zlib+recv.portmonitor+file.bottle_compression_zlib+type.dll",
            ):
                print("Waiting for Depth port to connect...")
                time.sleep(0.1)

        # self.read()

    def read(self) -> pl.DataFrame:
        """Read camera data and return as Polars DataFrame."""
        rows = []
        
        if self.rgb_shape:
            stamp = yarp.Stamp()
            start_time = yarp.now()
            read_attempts = 0
            while (data := self.rgb_port.read(False)) is None:
                read_attempts += 1
                
            read_time = yarp.now()
            read_delay = read_time - start_time
            self.rgb_port.getEnvelope(stamp)
            
            self.yarp_rgb_image.copy(data)
            rgb_image = np.frombuffer(self.rgb_buffer, dtype=np.uint8).reshape(
                self.rgb_shape[1], self.rgb_shape[0], 3
            ).copy()  # Create a copy to avoid buffer reuse issues
            
            # Create metadata
            metadata = self._create_metadata_dict(
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
                metadata_key=f"{self.stream_name}_rgb"
            )
            
            # Create DataFrame row
            row_data = {
                "name": f"{self.stream_name}_rgb",
                "stream_type": "camera",
                "entity_path": "",
                "data": {"rgb": rgb_image},
                "metadata": metadata,
            }
            
            rows.append(row_data)
        
        if self.depth_shape:
            stamp = yarp.Stamp()
            start_time = yarp.now()
            read_attempts = 0
            while (data := self.depth_port.read(False)) is None:
                read_attempts += 1
                
            read_time = yarp.now()
            read_delay = read_time - start_time
            self.depth_port.getEnvelope(stamp)
            
            self.yarp_depth_image.copy(data)
            depth_image = (
                np.frombuffer(self.depth_buffer, dtype=np.float32).reshape(
                    self.depth_shape[1], self.depth_shape[0]
                ).copy() * 1000  # Create a copy to avoid buffer reuse issues
            ).astype(np.uint16)
            
            # Create metadata
            metadata = self._create_metadata_dict(
                timestamp=stamp.getTime(),
                seq_number=stamp.getCount(),
                read_timestamp=read_time,
                read_delay=read_delay,
                read_attempts=read_attempts,
                metadata_key=f"{self.stream_name}_depth"
            )
            
            # Create DataFrame row
            row_data = {
                "name": f"{self.stream_name}_depth",
                "stream_type": "camera",
                "entity_path": "",
                "data": {"depth": depth_image},
                "metadata": metadata,
            }
            
            rows.append(row_data)
        
        return pl.DataFrame(rows, schema=STREAM_SCHEMA)
    
    def reset(self):
        """Reset the camera interface (no specific reset needed)."""
        pass

    def close(self):
        """Close all YARP ports."""
        if hasattr(self, 'rgb_port') and self.rgb_port:
            self.rgb_port.close()
        if hasattr(self, 'depth_port') and self.depth_port:
            self.depth_port.close()
