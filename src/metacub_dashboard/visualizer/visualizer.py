from dataclasses import dataclass, field
import math
import os
import time
# os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME'] = 'NVIDIA'
# os.environ['VK_ICD_FILENAMES'] = 'C:\Windows\System32\DriverStore\FileRepository\nvdmegpu.inf_amd64_95b916fb18cfa9ac\nv-vk64.json'
# os.environ['WGPU_BACKEND'] = 'gl'
# os.environ['WGPU_POWER_PREF'] = 'high'
import sys
import threading

from gradio_rerun import Rerun

from .utils.blueprint import build_blueprint
from .utils.gradio_interface import build_gradio_interface
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from uuid import uuid4 
import numpy as np
from pathlib import Path
import rerun as rr



from .utils.urdf_logger import URDFLogger
import uuid
from scipy.spatial.transform import Rotation as R




from matplotlib import pyplot as plt
def log_frame(name, pose, rec, scale=0.1, static=False):
    rec.log(f'{name}', rr.Transform3D(translation=pose.pos, mat3x3=pose.ori), static=static)
    rec.log(f'{name}/axes', rr.Arrows3D(origins=np.zeros([3,3]), vectors=np.eye(3) * scale, colors=np.eye(3)), static=static)
    rec.log(f'{name}/point', rr.Points3D([0,0,-0.04], labels=[f'{name.split('/')[-1]} ({pose.grip})'], radii=[0.001, ]), static=static)

    pose_array = np.concatenate([pose.pos, R.from_matrix(pose.ori).as_rotvec(), pose.grip])
    pose_names = ['x', 'y', 'z', 'ax', 'ay', 'az', 'g']
    for n, p, in zip(pose_names, pose_array):
        rec.log(f'{name}/components/{n}', rr.Scalars(p), static=static)

class Visualizer:
    dir_path: Path
    trajectory_length: int
    metadata: dict
    cameras: dict[str, np.ndarray]

    def __init__(self, urdf, blueprint, robot_pose=None, gradio=True):

        rid = uuid.uuid4()
        self.rec = rr.RecordingStream(application_id="rerun_example_gradio", recording_id=rid)
        
        if gradio:
            self.stop_streaming_event = threading.Event() # Added
            demo = build_gradio_interface(self.rec, self.stop_streaming_event)
            demo.launch(ssr_mode=False, share=False, prevent_thread_lock=True)
        else:
            rr.spawn(recording=self.rec, memory_limit="50%")
            self.rec.connect_grpc()

        self.rec.set_time("real_time", duration=0.0)

        # Log the URDF
        self.urdf_logger = URDFLogger(urdf, self.rec)
        self.urdf_logger.init()
        if robot_pose is not None:
            self.rec.log(self.urdf_logger.urdf.get_root(), rr.Transform3D(translation=robot_pose.pos, mat3x3=robot_pose.ori))

        # Log blueprint
        self.rec.send_blueprint(blueprint)
        self.prev_time = 0.0

        # Log virtual cameras
        # hand_path = '/'.join(self.urdf_logger.urdf.get_chain(root=self.urdf_logger.urdf.get_root(), tip='fr3_hand')[0::2])
        # self.rec.log(f'{hand_path}/front_camera', rr.Transform3D(translation=np.array([0.4, 0.0, 0.0]), mat3x3=R.from_euler('zyx', [0, -np.pi/2, np.pi/2]).as_matrix()))
        # self.rec.log(f'{hand_path}/front_camera', rr.Pinhole(fov_y=0.78, image_plane_distance=0.1, aspect_ratio=1.7777778))
        # self.rec.log(f'{hand_path}/right_camera', rr.Transform3D(translation=np.array([0.0, -0.4, 0.0]), mat3x3=R.from_euler('xyz', [-np.pi/2, np.pi, 0]).as_matrix()))
        # self.rec.log(f'{hand_path}/right_camera', rr.Pinhole(fov_y=0.78, image_plane_distance=0.1, aspect_ratio=1.7777778))

        
    def log_stats(self, packet, static):
        self.rec.log(f'/streams/write_frequency/{packet.name}', rr.Scalars(packet.freq), static=static)
        self.rec.log(f'/streams/write_timestamp/{packet.name}', rr.Scalars(packet.timestamp), static=static)
        self.rec.log(f'/streams/read_timestamp/{packet.name}', rr.Scalars(packet.read_timestamp), static=static)
        self.rec.log(f'/streams/read_delay/{packet.name}', rr.Scalars(packet.read_delay), static=static)
        self.rec.log(f'/streams/read_attempts/{packet.name}', rr.Scalars(packet.read_attempts), static=static)


    def log(self,
            joints: dict[str, np.ndarray]={},
            images: dict[str, np.ndarray]={},
            depths: dict[str, np.ndarray]={},
            poses: dict[str, np.ndarray]={},
            cameras: dict[str, np.ndarray]={},
            trajectories: dict[str, np.ndarray]={},
            _time:float=0.0,
            timestamp: float=0.0,
            static: bool=False,
        ):
        self.rec.set_time("real_time", duration=timestamp)
        if _time != 0.0 and self.prev_time != 0.0:
            self.rec.log('/streams/write_frequency/data_collection', rr.Scalars(1/(_time - self.prev_time)), static=static)
        self.prev_time = _time
        
        # Use joints to move the urdf
        for packet in joints:
            for joint_name, angle in zip(packet.data_labels, packet.data):
                self.log_stats(packet, static)
                # self.urdf_logger.log(joint_name, angle)
                # self.rec.log('joints/' + joint_name, rr.Scalars(angle), static=static)

        # Log the camera images
        for name, image in images.items():
            self.log_stats(image, static)
            # self.rec.log(name, rr.Image(image.data), static=static)
        # Log the camera depths
        for name, depth in depths.items():
            self.log_stats(depth, static)
            # self.rec.log(name, rr.DepthImage(depth.data), static=static)

        # Log reference frames
        for packet in poses:
            if packet.name == 'action':
                self.log_stats(packet, static)
            # log_frame(packet.name, packet.data, self.rec, static=static)

        # Log cameras
        for name, camera in cameras.items():
            f = 0.5 * camera.height / math.tan(camera.fov * math.pi / 360)
            camera_matrix = np.array(((f, 0, camera.width / 2), (0, f, camera.height / 2), (0, 0, 1)))
            self.rec.log(name, rr.Pinhole(image_from_camera=camera_matrix), static=static)
            self.rec.log(name, rr.Transform3D(translation=camera.pos, mat3x3=camera.ori), static=static)

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        colormap = cm.get_cmap('viridis')
        # norm = mcolors.Normalize(vmin=0, vmax=len(trajectories))

        # Log trajectories
        for name, traj in trajectories.items():
            self.rec.log(f"trajectories/{name}", rr.LineStrips3D(strips=traj[...,:3]), static=static)
            # self.rec.log(f"trajectories/{name}/end", rr.Points3D(traj[-1, :3], colors=[255,0,0], radii=0.001), static=static)
            self.rec.log(f"trajectories/{name}/ori", rr.Transform3D(translation=traj[-1, :3], mat3x3=R.from_rotvec(traj[-1,3:6]).as_matrix(), axis_length=0.01), static=static)
            self.rec.log(f"trajectories/{name}", rr.LineStrips3D.from_fields(colors=colormap(traj[-1, -1])), static=static)


@dataclass
class Camera:
    pos: np.ndarray
    ori: np.ndarray
    fov: float = 45.0
    width: int = 240
    height: int = 240
