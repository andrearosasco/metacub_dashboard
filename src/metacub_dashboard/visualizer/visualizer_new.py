"""
Refactored visualizer that works with StreamData and separates concerns:
- Poses: 3D reference frames with position/orientation
- Encoders: Joint values for robot movement  
- Images: RGB/depth data
- Diagnostics: Stream metadata (frequency, delays, etc.)
"""
from dataclasses import dataclass, field
import math
import os
os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
os.environ['WGPU_BACKEND'] = 'vulkan'
import time
import sys
import threading
import numpy as np
from pathlib import Path
import rerun as rr
from uuid import uuid4
from scipy.spatial.transform import Rotation as R

# from gradio_rerun import Rerun
from .utils.blueprint import build_blueprint
# from .utils.gradio_interface import build_gradio_interface
from .utils.urdf_logger import URDFLogger
from ..interfaces.stream_data import StreamData, Pose


def log_pose_frame(name: str, pose: Pose, rec: rr.RecordingStream, scale: float = 0.1, static: bool = False):
    """Log a pose as a 3D reference frame."""
    rec.log(f'{name}', rr.Transform3D(translation=pose.pos, mat3x3=pose.ori), static=static)
    rec.log(f'{name}/axes', rr.Arrows3D(
        origins=np.zeros([3, 3]), 
        vectors=np.eye(3) * scale, 
        colors=np.eye(3)
    ), static=static)
    rec.log(f'{name}/point', rr.Points3D(
        [0, 0, -0.04], 
        labels=[f'{name.split("/")[-1]} ({pose.grip})'], 
        radii=[0.001]
    ), static=static)
    
    # Log pose components as scalars
    pose_array = np.concatenate([pose.pos, R.from_matrix(pose.ori).as_rotvec(), pose.grip])
    pose_names = ['x', 'y', 'z', 'ax', 'ay', 'az', 'g']
    for n, p in zip(pose_names, pose_array):
        rec.log(f'{name}/components/{n}', rr.Scalars(p), static=static)


@dataclass
class Camera:
    pos: np.ndarray
    ori: np.ndarray
    fov: float = 45.0
    width: int = 240
    height: int = 240


class Visualizer:
    """
    Clean visualizer that accepts StreamData objects and logs:
    - Poses as 3D reference frames
    - Encoders for robot joint movement
    - Images (RGB/depth) 
    - Stream diagnostics
    """
    
    def __init__(self, urdf, blueprint, robot_pose=None, gradio=True):
        rid = uuid4()
        self.rec = rr.RecordingStream(application_id="metacub_dashboard", recording_id=rid)
        
        if gradio:
            pass
            # self.stop_streaming_event = threading.Event()
            # demo = build_gradio_interface(self.rec, self.stop_streaming_event)
            # demo.launch(ssr_mode=False, share=False, prevent_thread_lock=True)
        else:
            rr.spawn(recording=self.rec, memory_limit="50%")
            self.rec.connect_grpc()

        self.rec.set_time("real_time", duration=0.0)

        # Log the URDF
        self.urdf_logger = URDFLogger(urdf, self.rec)
        self.urdf_logger.init()
        
        if robot_pose is not None:
            self.rec.log(self.urdf_logger.urdf.get_root(), 
                        rr.Transform3D(translation=robot_pose.pos, mat3x3=robot_pose.ori))

        # Log blueprint
        self.rec.send_blueprint(blueprint)
        self.prev_time = 0.0

    def log_stream_diagnostics(self, stream_data: StreamData, static: bool = False):
        """Log diagnostic information for a stream."""
        metadata = stream_data.metadata
        stream_name = stream_data.name
        
        # Log to /streams path to match blueprint expectations
        self.rec.log(f'/streams/write_frequency/{stream_name}', 
                    rr.Scalars(metadata.frequency if metadata.frequency else 0.0), static=static)
        self.rec.log(f'/streams/write_timestamp/{stream_name}', 
                    rr.Scalars(metadata.timestamp), static=static)
        self.rec.log(f'/streams/read_timestamp/{stream_name}', 
                    rr.Scalars(metadata.read_timestamp), static=static)
        self.rec.log(f'/streams/read_delay/{stream_name}', 
                    rr.Scalars(metadata.read_delay), static=static)
        self.rec.log(f'/streams/read_attempts/{stream_name}', 
                    rr.Scalars(metadata.read_attempts), static=static)
        self.rec.log(f'/streams/missed_packets/{stream_name}', 
                    rr.Scalars(metadata.missed_packets if metadata.missed_packets else 0), static=static)

    def log_poses(self, poses_stream: StreamData, entity_path_prefix: str = "poses", static: bool = False):
        """Log poses as 3D reference frames at the specified entity path."""
        if poses_stream.stream_type != "poses":
            print(f"Warning: Expected poses stream, got {poses_stream.stream_type}")
            return
            
        self.log_stream_diagnostics(poses_stream, static)
        
        for pose_name, pose in poses_stream.data.items():
            if isinstance(pose, Pose):
                entity_path = f"{entity_path_prefix}/{pose_name}"
                log_pose_frame(entity_path, pose, self.rec, static=static)
            else:
                print(f"Warning: Expected Pose object for {pose_name}, got {type(pose)}")

    def log_encoders(self, encoders_stream: StreamData, static: bool = False):
        """Log encoder data and move robot joints."""
        if encoders_stream.stream_type != "encoders":
            print(f"Warning: Expected encoders stream, got {encoders_stream.stream_type}")
            return
            
        self.log_stream_diagnostics(encoders_stream, static)
        
        # Move robot joints and log scalar values
        for board_name, board_data in encoders_stream.data.items():
            if isinstance(board_data, dict) and 'values' in board_data and 'labels' in board_data:
                values = board_data['values']
                labels = board_data['labels']
                
                for joint_name, angle in zip(labels, values):
                    # Move the robot joint
                    self.urdf_logger.log(joint_name, angle)
                    
                    # Log joint value as scalar at specified entity path
                    base_path = encoders_stream.entity_path if encoders_stream.entity_path else "joints"
                    self.rec.log(f'{base_path}/{joint_name}', rr.Scalars(angle), static=static)

    def log_images(self, camera_stream: StreamData, path_prefix: str = "", static: bool = False):
        """Log camera images (RGB and depth)."""
        if camera_stream.stream_type != "camera":
            print(f"Warning: Expected camera stream, got {camera_stream.stream_type}")
            return
            
        self.log_stream_diagnostics(camera_stream, static)
        
        for image_type, image_data in camera_stream.data.items():
            if image_type == "rgb":
                path = f'{path_prefix}/rgb' if path_prefix else 'camera/rgb'
                self.rec.log(path, rr.Image(image_data), static=static)
            elif image_type == "depth":
                path = f'{path_prefix}/depth' if path_prefix else 'camera/depth'
                self.rec.log(path, rr.DepthImage(image_data), static=static)

    def log_cameras(self, cameras: dict[str, Camera], static: bool = False):
        """Log virtual camera poses and parameters."""
        for name, camera in cameras.items():
            # Calculate camera matrix
            f = 0.5 * camera.height / math.tan(camera.fov * math.pi / 360)
            camera_matrix = np.array([
                [f, 0, camera.width / 2], 
                [0, f, camera.height / 2], 
                [0, 0, 1]
            ])
            
            self.rec.log(name, rr.Pinhole(image_from_camera=camera_matrix), static=static)
            self.rec.log(name, rr.Transform3D(translation=camera.pos, mat3x3=camera.ori), static=static)

    def log_trajectories(self, trajectories: dict[str, np.ndarray], static: bool = False):
        """Log 3D trajectories."""
        import matplotlib.cm as cm
        colormap = cm.get_cmap('viridis')
        
        for name, traj in trajectories.items():
            # Main trajectory line
            self.rec.log(f"trajectories/{name}", 
                        rr.LineStrips3D(strips=traj[..., :3]), static=static)
            
            # End point orientation
            if traj.shape[1] >= 6:  # Has orientation data
                self.rec.log(f"trajectories/{name}/end_pose", 
                            rr.Transform3D(
                                translation=traj[-1, :3], 
                                mat3x3=R.from_rotvec(traj[-1, 3:6]).as_matrix(), 
                                axis_length=0.01
                            ), static=static)
            
            # Color based on trajectory value if available
            if traj.shape[1] > 6:
                color_value = traj[-1, -1]
                self.rec.log(f"trajectories/{name}", 
                            rr.LineStrips3D.from_fields(colors=colormap(color_value)), static=static)

    def log(self, 
            poses_streams: list[StreamData] = None,
            encoders_streams: list[StreamData] = None, 
            camera_streams: list[StreamData] = None,
            cameras: dict[str, Camera] = None,
            trajectories: dict[str, np.ndarray] = None,
            timestamp: float = 0.0,
            static: bool = False):
        """
        Main logging method that accepts StreamData objects.
        
        Args:
            poses_streams: List of StreamData with stream_type="poses"
            encoders_streams: List of StreamData with stream_type="encoders" 
            camera_streams: List of StreamData with stream_type="camera"
            cameras: Virtual cameras to display
            trajectories: 3D trajectory data
            timestamp: Current timestamp
            static: Whether data is static
        """
        self.rec.set_time("real_time", duration=timestamp)
        
        # Log collection frequency
        current_time = time.perf_counter()
        if self.prev_time != 0.0:
            collection_freq = 1.0 / (current_time - self.prev_time)
            self.rec.log('/streams/write_frequency/data_collection', 
                        rr.Scalars(collection_freq), static=static)
        self.prev_time = current_time
        
        # Log poses
        if poses_streams:
            for poses_stream in poses_streams:
                # Use default "poses" entity path for regular poses
                self.log_poses(poses_stream, entity_path_prefix="poses", static=static)
                
        # Log encoders
        if encoders_streams:
            for encoders_stream in encoders_streams:
                self.log_encoders(encoders_stream, static)
                
        # Log camera data
        if camera_streams:
            for i, camera_stream in enumerate(camera_streams):
                # Use stream name for path, or default indexing
                path_prefix = camera_stream.name if camera_stream.name != "camera" else f"camera_{i}"
                self.log_images(camera_stream, path_prefix, static)
                
        # Log virtual cameras
        if cameras:
            self.log_cameras(cameras, static)
            
        # Log trajectories  
        if trajectories:
            self.log_trajectories(trajectories, static)
