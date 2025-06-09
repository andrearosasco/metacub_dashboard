"""
Visualizer that works directly with Polars DataFrames (now the default).
Eliminates the need for StreamData objects and wrapper classes.
"""
from dataclasses import dataclass, field
import math
import os
os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
# os.environ['WGPU_BACKEND'] = 'vulkan'
import time
import numpy as np
import rerun as rr
from uuid import uuid4
from scipy.spatial.transform import Rotation as R
import polars as pl

from .utils.blueprint import build_blueprint
from .utils.urdf_logger import URDFLogger

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
    
    def numpy(self) -> np.ndarray:
        """
        Returns a concatenated numpy array of position, flattened orientation, and grip.
        
        Returns:
            np.ndarray: A 1D numpy array containing pos, flattened ori, and grip values
        """
        pos_array = self.pos.flatten() if self.pos is not None else np.array([0.0, 0.0, 0.0])
        ori_array = self.ori.flatten() if self.ori is not None else np.identity(3).flatten()
        grip_array = self.grip.flatten() if self.grip is not None else np.array([0.0])
        
        return np.concatenate([pos_array, ori_array, grip_array])
    

@dataclass
class Camera:
    pos: np.ndarray
    ori: np.ndarray
    fov: float = 45.0
    width: int = 240
    height: int = 240


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




class Visualizer:
    """
    Visualizer that works directly with Polars DataFrames (now the default).
    Eliminates StreamData objects and wrapper classes.
    """
    
    def __init__(self, urdf, blueprint, robot_pose=None, gradio=True):
        rid = uuid4()
        self.rec = rr.RecordingStream(application_id="metacub_dashboard", recording_id=rid)
        
        if gradio:
            pass
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

    def log_dataframe_diagnostics(self, df: pl.DataFrame, static: bool = False):
        """Log diagnostic information directly from Polars DataFrame."""
        for row in df.iter_rows(named=True):
            stream_name = row['name']
            metadata = row['metadata']
            
            # Log to /streams path to match blueprint expectations
            self.rec.log(f'/streams/write_frequency/{stream_name}', 
                        rr.Scalars(metadata['frequency'] if metadata['frequency'] else 0.0), static=static)
            self.rec.log(f'/streams/write_timestamp/{stream_name}', 
                        rr.Scalars(metadata['timestamp']), static=static)
            self.rec.log(f'/streams/read_timestamp/{stream_name}', 
                        rr.Scalars(metadata['read_timestamp']), static=static)
            self.rec.log(f'/streams/read_delay/{stream_name}', 
                        rr.Scalars(metadata['read_delay']), static=static)
            self.rec.log(f'/streams/read_attempts/{stream_name}', 
                        rr.Scalars(metadata['read_attempts']), static=static)

    def log_poses_dataframe(self, poses_df: pl.DataFrame, entity_path_prefix: str = "poses", static: bool = False):
        """Log poses directly from Polars DataFrame."""
        poses_only_df = poses_df.filter(pl.col("stream_type") == "poses")
        self.log_dataframe_diagnostics(poses_only_df, static)
        
        for row in poses_only_df.iter_rows(named=True):
            data = row['data']
            for pose_name, pose in data.items():
                if isinstance(pose, Pose):
                    entity_path = f"{entity_path_prefix}/{pose_name}"
                    log_pose_frame(entity_path, pose, self.rec, static=static)

    def log_encoders_dataframe(self, encoders_df: pl.DataFrame, static: bool = False):
        """Log encoder data directly from Polars DataFrame."""
        encoders_only_df = encoders_df.filter(pl.col("stream_type") == "encoders")
        self.log_dataframe_diagnostics(encoders_only_df, static)
        
        for row in encoders_only_df.iter_rows(named=True):
            data = row['data']
            entity_path = row['entity_path'] if row['entity_path'] else "joints"
            
            for board_name, board_data in data.items():
                if isinstance(board_data, dict) and 'values' in board_data and 'labels' in board_data:
                    values = board_data['values']
                    labels = board_data['labels']
                    
                    for joint_name, angle in zip(labels, values):
                        # Move the robot joint
                        self.urdf_logger.log(joint_name, angle)
                        
                        # Log joint value as scalar
                        self.rec.log(f'{entity_path}/{joint_name}', rr.Scalars(angle), static=static)

    def log_camera_dataframe(self, camera_df: pl.DataFrame, static: bool = False):
        """Log camera data directly from Polars DataFrame."""
        camera_only_df = camera_df.filter(pl.col("stream_type") == "camera")
        self.log_dataframe_diagnostics(camera_only_df, static)
        
        for i, row in enumerate(camera_only_df.iter_rows(named=True)):
            data = row['data']
            stream_name = row['name']
            
            # Use stream name for path, or default indexing
            path_prefix = stream_name if stream_name != "camera" else f"camera_{i}"
            
            for image_type, image_data in data.items():
                if image_type == "rgb":
                    path = f'{path_prefix}/rgb'
                    self.rec.log(path, rr.Image(image_data), static=static)
                elif image_type == "depth":
                    path = f'{path_prefix}/depth'
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

    def log_dataframes(self, 
                      poses_df: pl.DataFrame = None,
                      encoders_df: pl.DataFrame = None, 
                      camera_df: pl.DataFrame = None,
                      cameras: dict[str, Camera] = None,
                      trajectories: dict[str, np.ndarray] = None,
                      timestamp: float = 0.0,
                      static: bool = False):
        """
        Main logging method that accepts pure Polars DataFrames.
        
        Args:
            poses_df: DataFrame with stream_type="poses"
            encoders_df: DataFrame with stream_type="encoders" 
            camera_df: DataFrame with stream_type="camera"
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
        if poses_df is not None and len(poses_df) > 0:
            self.log_poses_dataframe(poses_df, entity_path_prefix="poses", static=static)
                
        # Log encoders
        if encoders_df is not None and len(encoders_df) > 0:
            self.log_encoders_dataframe(encoders_df, static)
                
        # Log camera data
        if camera_df is not None and len(camera_df) > 0:
            self.log_camera_dataframe(camera_df, static)
                
        # Log virtual cameras
        if cameras:
            self.log_cameras(cameras, static)
            
        # Log trajectories  
        if trajectories:
            self.log_trajectories(trajectories, static)
