"""
Pure Polars implementation of MetaCub Dashboard.
Uses native Polars DataFrames throughout with no wrapper classes.
"""
import time
import os
from uuid import uuid4
import numpy as np
from typing import List, Dict, Any

# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp
from scipy.spatial.transform import Rotation as R
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core
import polars as pl

# Import interfaces
from metacub_dashboard.interfaces.interfaces import (
    ActionInterface, 
    EncodersInterface, 
    CameraInterface
)
from metacub_dashboard.interfaces.utils.control_loop_reader import ControlLoopReader
from metacub_dashboard.visualizer.visualizer import Visualizer, Camera
from metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from metacub_dashboard.data_logger.logger import DataLogger as PolarsDataLogger
from metacub_dashboard.data_logger.data_logger import DataLogger
from metacub_dashboard.utils.keyboard_interface import KeyboardInterface, StatusAwarePrinter

urdf_parser_core.on_error = lambda x: x


def main():
    """Main function using pure Polars DataFrames throughout."""
    print("🚀 Starting MetaCub Dashboard with Pure Polars...")
    
    # Initialize YARP
    yarp.Network.init()
    session_id = uuid4()

    # Set up keyboard interface first
    print("⌨️  Setting up keyboard interface...")
    keyboard = KeyboardInterface("MetaCub Dashboard - Episode Control")
    printer = StatusAwarePrinter(keyboard)
    
    # Replace built-in print with status-aware print
    import builtins
    builtins.print = printer.print
    
    # From this point on, regular print() will use the status-aware printer
    print("🚀 MetaCub Dashboard with Enhanced Interface Started")

    # 1. Create interfaces
    print("📡 Setting up interfaces...")
    
    action_interface = ActionInterface(
        remote_prefix="/metaControllClient",
        local_prefix=f"/metacub_dashboard/{session_id}",
        stream_name="target_poses"
    )

    observation_interfaces = {
        "agentview": CameraInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            rgb_shape=(640, 480),
            depth_shape=None,
            stream_name="agentview"
        ),
        "encoders": EncodersInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            control_boards=["head", "left_arm", "right_arm", "torso"],
            stream_name="encoders"
        ),
    }

    control_reader = ControlLoopReader(
        action_interface=action_interface,
        observation_interfaces=observation_interfaces,
        control_frequency=10.0,
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
        poses=["target_poses", "robot_joints"],
    )

    visualizer = Visualizer(urdf=urdf, blueprint=blueprint, gradio=False)

    # 3. Set up data logging
    print("💾 Setting up data logging...")
    
    # 4. Main application loop with episode control
    print("🔄 Starting MetaCub Dashboard...")
    print("📋 Use keyboard commands:")
    print("   's' - Start new episode")
    print("   'e' - End current episode") 
    print("   'r' - Reset current episode")
    print("   'q' - Quit application")
    
    keyboard.update_status("Ready - Press 's' to start episode")
    keyboard.set_episode_state("STOPPED")
    
    episode_count = 0
    
    try:
        while True:  # Main application loop
            # Wait for start command
            while True:
                command = keyboard.get_command()
                if command == 'start':
                    episode_count += 1
                    print(f"🎬 Starting Episode {episode_count}")
                    keyboard.update_status(f"Episode {episode_count} - Recording...")
                    keyboard.set_episode_state("RECORDING")
                    break
                elif command == 'quit':
                    print("👋 Exiting application...")
                    keyboard.update_status("Shutting down...")
                    return
                # time.sleep(0.01)  # Small delay to prevent high CPU usage
            
            # Create data logger for this episode
            base_data_logger = DataLogger(
                path=f"assets/episode_{episode_count}.zarr.zip",
                flush_every=100,
                exist_ok=True
            )
            data_logger = PolarsDataLogger(base_data_logger)

            # Episode execution loop with DataFrame processing
            print("🔄 Starting control loop with DataFrame processing...")
            
            # Initialize control loop with first observation
            control_reader.reset()
            
            iteration = 0
            start_episode_time = time.perf_counter()
            
            try:
                while True:  # Episode loop
                    # Check for episode control commands
                    command = keyboard.get_command()
                    if command == 'end':
                        print(f"⏹️  Episode {episode_count} ended by user")
                        keyboard.update_status(f"Episode {episode_count} - Ending...")
                        keyboard.set_episode_state("STOPPED")
                        break
                    elif command == 'reset':
                        print(f"🔄 Episode {episode_count} reset by user")
                        keyboard.update_status(f"Episode {episode_count} - Resetting...")
                        keyboard.set_episode_state("RESET")
                        break
                    elif command == 'quit':
                        print(f"👋 Episode {episode_count} interrupted - Exiting...")
                        keyboard.update_status("Shutting down...")
                        data_logger.end_episode()
                        return
                    
                    # Read synchronized action/observation pair as pure DataFrames
                    control_data = control_reader.read()
                    
                    if control_data is None:
                        continue
                        
                    # ====== DATAFRAME PROCESSING ======
                    
                    # Extract DataFrames directly
                    action_df = control_data.actions_df
                    observation_df = control_data.observations_df
                    
                    # Step 1: Apply entity paths directly to all observations using when/then/otherwise
                    all_observation_df = observation_df.with_columns([
                        pl.when(pl.col("name").str.contains("left_arm"))
                        .then(pl.lit(f"{eef_paths[0]}/fingers"))
                        .when(pl.col("name").str.contains("right_arm"))  
                        .then(pl.lit(f"{eef_paths[1]}/fingers"))
                        .otherwise(pl.lit("joints"))
                        .alias("entity_path")
                    ])
                    
                    # Step 2: Split processed data by stream type
                    camera_df = all_observation_df.filter(pl.col("stream_type") == "camera")
                    encoder_df = all_observation_df.filter(pl.col("stream_type") == "encoders")
                    
                    # ====== PURE POLARS VISUALIZER ======
                    
                    # Log to visualizer with pure DataFrames
                    visualizer.log_dataframes(
                        poses_df=action_df,
                        encoders_df=encoder_df,
                        camera_df=camera_df,
                        timestamp=time.perf_counter() - start_episode_time,
                        static=False,
                    )
                    
                    # ====== END PURE POLARS VISUALIZATION ======
                    
                    # Data logging with optimized pure DataFrames (background processing)
                    data_logger.log_dataframes(
                        observations_df=all_observation_df,
                        actions_df=action_df
                    )
                    
                    iteration += 1
                    
                    # Update status periodically to show progress
                    if iteration % 100 == 0:
                        keyboard.update_status(f"Episode {episode_count} - Recording... (iter: {iteration})")
                    
            except KeyboardInterrupt:
                print(f"⏹️  Episode {episode_count} interrupted by Ctrl+C")
                keyboard.update_status(f"Episode {episode_count} - Interrupted")
                keyboard.set_episode_state("INTERRUPTED")
            
            finally:
                print(f"🧹 Cleaning up Episode {episode_count}...")
                
                # End episode with pure Polars diagnostics
                data_logger.end_episode()
                
                keyboard.update_status(f"Episode {episode_count} complete - Press 's' for new episode")
                keyboard.set_episode_state("STOPPED")
                print(f"✅ Episode {episode_count} complete! Press 's' to start new episode or 'q' to quit.")
                
    except KeyboardInterrupt:
        print("⏹️  Application interrupted by user")
    
    finally:
        print("🧹 Final cleanup...")
        keyboard.close()
        control_reader.close()
        yarp.Network.fini()
        print("✅ Pure Polars clean shutdown complete!")


if __name__ == "__main__":
    # Uncomment to test pure Polars operations
    # demo_pure_polars_operations()
    
    # Run main application with pure Polars
    main()
