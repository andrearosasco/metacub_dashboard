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
    # Set up keyboard interface FIRST before any other initialization
    keyboard = KeyboardInterface("MetaCub Dashboard - Episode Control")
    printer = StatusAwarePrinter(keyboard)
    
    # Replace built-in print with status-aware print immediately
    import builtins
    original_print = builtins.print  # Keep reference to original print
    builtins.print = printer.print
    
    try:
        print("🚀 Starting MetaCub Dashboard...")
        
        # Initialize YARP
        print("📡 Initializing YARP Network...")
        yarp.Network.init()
        session_id = uuid4()
        

        control_reader = ControlLoopReader(
            action_interface=ActionInterface(
                remote_prefix="/ergocub",
                local_prefix=f"/metacub_dashboard/{session_id}",
                control_boards=["head", "left_arm", "torso"],
                stream_name="actions"
            ),
            observation_interfaces={
                'agentview': CameraInterface(
                    remote_prefix="",
                    local_prefix=f"/metacub_dashboard/{session_id}",
                    rgb_shape=(640, 480),
                    depth_shape=None,
                    stream_name="agentview"
                ),
                'encoders': EncodersInterface(
                    remote_prefix="/ergocub",
                    local_prefix=f"/metacub_dashboard/{session_id}",
                    control_boards=["head", "left_arm", "torso"],
                    stream_name="encoders"
                )
            }
        )

        # Visualization setup with error handling
        print("🎨 Setting up visualization...")
        urdf_path = yarp.ResourceFinder().findFileByName("model.urdf")
        if not urdf_path:
            print("⚠️  model.urdf not found, skipping visualization setup")
            visualizer = None
            exit()
        else:
            print(f"📄 Found URDF at: {urdf_path}")
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
            print("✅ Visualizer created successfully")


        # Data logging setup
        print("💾 Setting up data logging...")
        try:
            base_data_logger = DataLogger(
                path="assets/episode_data.zarr.zip",
                flush_every=100,
                exist_ok=True
            )
            data_logger = PolarsDataLogger(base_data_logger)
            print("✅ Data logger created successfully")
        except Exception as e:
            print(f"⚠️  Data logger setup failed: {e}")
            data_logger = None
        
        # Main application loop
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
                # Wait for start command (blocking)
                command = keyboard.get_command(blocking=True)
                if command == 'start':
                    episode_count += 1
                    print(f"🎬 Starting Episode {episode_count}")                        
                elif command == 'quit':
                    print("👋 Exiting application...")
                    keyboard.update_status("Shutting down...")
                    return
                else:
                    # Ignore other commands when waiting for start/quit
                    continue

                # Episode execution loop with DataFrame processing
                print("🔄 Resetting the robot...")
                
                # Initialize control loop with first observation
                control_reader.reset()
                keyboard.update_status(f"Episode {episode_count} - Ready")
                keyboard.set_episode_state("READY")
                
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
                        
                        control_data = control_reader.read()
                        
                        if control_data is None:
                            continue
                            
                        # ====== DATAFRAME PROCESSING ======
                        
                        # Extract DataFrames directly
                        action_df = control_data.actions_df
                        observation_df = control_data.observations_df
                        
                        # Apply entity paths directly to all observations using when/then/otherwise
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
                        
                        # Log to visualizer with pure DataFrames
                        visualizer.log_dataframes(
                            poses_df=action_df,
                            encoders_df=encoder_df,
                            camera_df=camera_df,
                            timestamp=time.perf_counter() - start_episode_time,
                            static=False,
                        )
                        
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
                    
                    # Ask user if they want to keep or discard the episode
                    print("💾 Do you want to keep this episode?")
                    print("   'k' - Keep episode (save to disk)")
                    print("   'd' - Discard episode (delete data)")
                    keyboard.update_status(f"Episode {episode_count} - Keep (k) or Discard (d)?")
                    
                    # Wait for keep/discard decision (blocking)
                    while True:
                        decision = keyboard.get_command(blocking=True)
                        if decision == 'keep':
                            print("✅ Keeping episode - saving to disk...")
                            keyboard.update_status(f"Episode {episode_count} - Saving...")
                            # End episode with pure Polars diagnostics (saves data)
                            data_logger.end_episode()
                            print(f"💾 Episode {episode_count} saved successfully!")
                            break
                        elif decision == 'discard':
                            print("🗑️  Discarding episode - data will not be saved...")
                            keyboard.update_status(f"Episode {episode_count} - Discarding...")
                            # Discard episode data without saving
                            data_logger.discard_episode()
                            print(f"🗑️  Episode {episode_count} discarded!")
                            break
                        elif decision == 'quit':
                            print("👋 Exiting application...")
                            keyboard.update_status("Shutting down...")
                            return
                        else:
                            # Ignore other commands
                            continue
                    
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
    
    except Exception as e:
        original_print(f"❌ Unexpected error: {e}")
        original_print("🛑 Application terminated due to an error")
        
        # Final cleanup in case of error
        try:
            keyboard.close()
            control_reader.close()
            yarp.Network.fini()
            original_print("✅ Cleanup complete")
        except Exception as cleanup_e:
            original_print(f"⚠️  Cleanup error: {cleanup_e}")
        
        raise  # Re-raise the exception after cleanup


if __name__ == "__main__":
    # Uncomment to test pure Polars operations
    # demo_pure_polars_operations()
    
    # Run main application with pure Polars
    main()
