"""
Pure Polars-based streaming interfaces (now the default).
Uses native Polars DataFrames with schemas for type safety and powerful operations.
"""
import polars as pl
import numpy as np
import yarp
import time
from typing import Dict, List, Any, Optional
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
    def read(self) -> pl.DataFrame:
        """Read data and return as Polars DataFrame with enforced schema."""
        pass

    @abstractmethod
    def close(self):
        """Close the interface."""
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
    
    def __init__(self, remote_prefix: str, local_prefix: str, stream_name: str = "poses"):
        super().__init__(stream_name)
        self.remote_prefix = remote_prefix
        self.local_prefix = local_prefix
        
        # Initialize YARP port
        self.port = yarp.BufferedPortBottle()
        self.port.open(f"{local_prefix}/action:i")
        yarp.Network.connect(f"{remote_prefix}/action:o", f"{local_prefix}/action:i")
        
        # Action format
        self.format = [
            ("neck", 7),
            ("left_arm", 7), 
            ("right_arm", 7),
            ("fingers", 6)
        ]

        self.read()

    def read(self) -> pl.DataFrame:
        """Read action data and return as Polars DataFrame."""
        stamp = yarp.Stamp()
        
        start_time = yarp.now()
        read_attempts = 0
        while (bottle := self.port.read(False)) is None:
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

        # Parse bottle data
        poses_data = {}
        for pose_name, pose_size in self.format:
            pose_list = None
            for i in range(bottle.size()):
                if bottle.get(i).asList().get(0).asString() == pose_name:
                    pose_list = bottle.get(i).asList().get(1).asList()
                    break
            
            if pose_list:
                pose_array = np.array([pose_list.get(j).asFloat64() for j in range(pose_size)])
                poses_data[pose_name] = pose_array            # Create metadata


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
        self.port.close()


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
        
        # Initialize YARP ports for each board
        self.encoders = {}
        for board in control_boards:
            port = yarp.BufferedPortVector()
            port.open(f"{local_prefix}/{board}/state:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/{board}/state:o", f"{local_prefix}/{board}/state:i", 'tcp'
            ):
                print(f"Waiting for {remote_prefix}/{board}/state:o port to connect...")
                time.sleep(0.1)
            self.encoders[board] = port

        self.read()

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

    def close(self):
        """Close all YARP ports."""
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
        
        # Initialize RGB port if needed
        if rgb_shape:
            self.rgb_port = yarp.BufferedPortImageRgb()
            self.rgb_port.open(f"{local_prefix}/rgb:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/depthCamera/rgbImage:o",
                f"{local_prefix}/rgb:i",
                "mjpeg",
            ):
                print("Waiting for RGB port to connect...")
                time.sleep(0.1)
            
            self.yarp_rgb_image = yarp.ImageRgb()
            self.yarp_rgb_image.resize(rgb_shape[0], rgb_shape[1])
            self.rgb_buffer = bytearray(rgb_shape[0] * rgb_shape[1] * 3)
            self.yarp_rgb_image.setExternal(self.rgb_buffer, rgb_shape[0], rgb_shape[1])
        
        # Initialize depth port if needed
        if depth_shape:
            self.depth_port = yarp.BufferedPortImageFloat()
            self.depth_port.open(f"{local_prefix}/depth:i")
            while not yarp.Network.connect(
                f"{remote_prefix}/depthCamera/depthImage:o",
                f"{local_prefix}/depth:i",
                "fast_tcp+send.portmonitor+file.bottle_compression_zlib+recv.portmonitor+file.bottle_compression_zlib+type.dll",
            ):
                print("Waiting for Depth port to connect...")
                time.sleep(0.1)
            
            self.yarp_depth_image = yarp.ImageFloat()
            self.yarp_depth_image.resize(depth_shape[0], depth_shape[1])
            self.depth_buffer = bytearray(depth_shape[0] * depth_shape[1] * 4)
            self.yarp_depth_image.setExternal(self.depth_buffer, depth_shape[0], depth_shape[1])

        self.read()

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
            )
            
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
                ) * 1000
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

    def close(self):
        """Close all YARP ports."""
        if hasattr(self, 'rgb_port'):
            self.rgb_port.close()
        if hasattr(self, 'depth_port'):
            self.depth_port.close()
