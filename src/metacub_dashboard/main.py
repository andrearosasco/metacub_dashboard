"""
Complete example of the new clean StreamData architecture.
Shows how poses, encoders, cameras work together with clean separation of concerns.
"""
import time
import os
from uuid import uuid4
import numpy as np

# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp
from scipy.spatial.transform import Rotation as R
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core

# Import new clean interfaces
from metacub_dashboard.interfaces.action_interface import ActionInterface
from metacub_dashboard.interfaces.camera_interface import CameraInterface
from metacub_dashboard.interfaces.encoders_interface import EncodersInterface
from metacub_dashboard.interfaces.stream_data import StreamData, Pose
from metacub_dashboard.interfaces.utils.control_loop_reader import ControlLoopReader
from metacub_dashboard.visualizer.visualizer_new import Visualizer, Camera
from metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from metacub_dashboard.data_logger.data_logger import DataLogger
from metacub_dashboard.data_logger.stream_logger import StreamDataLogger

urdf_parser_core.on_error = lambda x: x


def main():
    """Main function demonstrating the clean new architecture."""
    print("🚀 Starting MetaCub Dashboard with new clean architecture...")
    
    # Initialize YARP
    yarp.Network.init()
    session_id = uuid4()

    # 1. Create unified control loop reader with action and observation streams
    print("📡 Setting up control loop reader...")
    
    action_interface = ActionInterface(
        remote_prefix="/metaControllClient",
        local_prefix=f"/metacub_dashboard/{session_id}",
        stream_name="target_poses"  # Clear semantic name
    )

    observation_interfaces = {
        "main_camera": CameraInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            rgb_shape=(640, 480),
            depth_shape=None,
            stream_name="main_camera"
        ),
        "robot_joints": EncodersInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            control_boards=["head", "left_arm", "right_arm", "torso"],
            concatenate=False,  # Keep individual boards
            stream_name="robot_joints"
        ),
    }

    control_reader = ControlLoopReader(
        action_interface=action_interface,
        observation_streams=observation_interfaces,
        control_frequency=10.0,  # Hz - this now controls the entire loop timing
        blocking=False,
    )

    # 2. Set up visualization
    print("🎨 Setting up visualization...")
    
    urdf_path = yarp.ResourceFinder().findFileByName("model.urdf")
    urdf = urdf_parser.URDF.from_xml_file(urdf_path)
    urdf.path = urdf_path

    # Build visualization blueprint
    camera_path = "/".join(
        urdf.get_chain(root=urdf.get_root(), tip="realsense_depth_frame")[0::2]
    )
    image_paths = [f"{camera_path}/cameras/agentview_rgb"]
    eef_paths = [
        "/".join(urdf.get_chain(root=urdf.get_root(), tip=eef)[0::2])
        for eef in ["l_hand_palm", "r_hand_palm"]
    ]

    blueprint = build_blueprint(
        image_paths=image_paths,
        eef_paths=eef_paths,
        poses=["target_poses", "robot_joints"],  # Clear semantic names
    )

    visualizer = Visualizer(urdf=urdf, blueprint=blueprint, gradio=False)

    # 3. Set up data logging with StreamDataLogger wrapper
    print("💾 Setting up data logging...")
    
    base_logger = DataLogger(
        path="assets/dataset_new.zarr.zip", 
        flush_every=200, 
        exist_ok=True
    )
    data_logger = StreamDataLogger(base_logger)

    # 4. Main execution loop with unified control timing
    print("🔄 Starting control loop...")
    
    iteration = 0
    prev_time = time.perf_counter()
    
    try:
        while iteration < 50:  # Run for 50 iterations as demo
            # Read synchronized action/observation pair at control frequency
            control_data = control_reader.read()
            
            # Extract action and observation streams
            pose_stream = control_data.action_stream
            observation_streams = control_data.observation_streams
            
            # Step 1: Filter streams by type using StreamData methods
            camera_streams = StreamData.get_streams_by_type(observation_streams, "camera")
            encoder_streams = StreamData.get_streams_by_type(observation_streams, "encoders")
            
            # Step 2: Apply entity path rules using StreamData processor
            processed_encoder_streams = (
                StreamData.create_processor()
                .add_rule(
                    condition=lambda s: "left_arm" in s.name,
                    action=lambda s: setattr(s, 'entity_path', f"{eef_paths[0]}/fingers")
                )
                .add_rule(
                    condition=lambda s: "right_arm" in s.name,
                    action=lambda s: setattr(s, 'entity_path', f"{eef_paths[1]}/fingers")
                )
                .add_rule(
                    condition=lambda s: "left_arm" not in s.name and "right_arm" not in s.name,
                    action=lambda s: setattr(s, 'entity_path', "joints")
                )
                .process_streams(encoder_streams)
            )
            
            # Log to visualizer with clean separation
            visualizer.log(
                poses_streams=[pose_stream],  # Target poses
                encoders_streams=processed_encoder_streams,  # Generator with entity paths
                camera_streams=camera_streams,  # RGB/depth images
                timestamp=control_data.loop_timestamp,
                static=False,
            )
            
            # Data logging is now much simpler!
            data_logger.log_streams(
                observation_streams=observation_streams,
                poses_stream=pose_stream
            )
            
            # Print diagnostic info
            current_time = time.perf_counter()
            if iteration % 10 == 0:
                collection_freq = 1 / (current_time - prev_time) if prev_time else 0
                print(f"📊 Iteration {iteration}: Collection freq: {collection_freq:.1f} Hz")
                print(f"   Pose freq: {pose_stream.metadata.frequency:.1f} Hz, missed: {pose_stream.metadata.missed_packets or 0}")
                for stream in observation_streams:
                    freq = stream.metadata.frequency or 0.0
                    missed = stream.metadata.missed_packets or 0
                    print(f"   {stream.name} freq: {freq:.1f} Hz, missed: {missed}")
            
            prev_time = current_time
            iteration += 1

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        print("🧹 Cleaning up...")
        data_logger.end_episode()  # Now includes diagnostic summary
        control_reader.close()
        yarp.Network.fini()
        print("✅ Clean shutdown complete!")



if __name__ == "__main__":
    # Uncomment to test individual interfaces first
    # demo_individual_interfaces()
    
    # Run main application
    main()
