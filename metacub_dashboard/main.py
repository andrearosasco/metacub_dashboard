"""
MetaCub Dashboard.
"""
import time
import os
import signal
import sys
from uuid import uuid4

# Set environment before importing yarp
os.environ["RERUN"] = "off"and 
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp
import polars as pl

# Import interfaces
from metacub_dashboard.interfaces.interfaces import (
    ActionInterface, 
    EncodersInterface, 
    CameraInterface
)
from metacub_dashboard.interfaces.utils.control_loop_reader import ControlLoopReader
from metacub_dashboard.visualizer.visualizer import Visualizer
from metacub_dashboard.data_logger.logger import DataLogger as PolarsDataLogger
from metacub_dashboard.data_logger.data_logger import DataLogger
from metacub_dashboard.utils.keyboard_interface import KeyboardInterface


def main():
    """Main function to run the MetaCub Dashboard."""
    # Initialize resources that need cleanup
    keyboard = None
    control_reader = None 
    visualizer = None
    data_logger = None
    base_data_logger = None
    
    def cleanup_resources():
        """Cleanup function that properly closes all resources."""
        print("\n🧹 Shutting down...")

        # Discard current episode if recording
        if data_logger is not None:
            data_logger.discard_episode()
        
        # Close YARP ports and control reader
        if control_reader is not None:
            control_reader.close()
        
        # Close keyboard interface
        if keyboard is not None:
            keyboard.close()
        
        print("✅ Shutdown complete!")
    
    def signal_handler(signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n🛑 Received signal {signum} (Ctrl+C), shutting down...")
        cleanup_resources()
        sys.exit(0)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Set up keyboard interface FIRST (handles signal and print setup automatically)
    keyboard = KeyboardInterface("MetaCub Dashboard - Episode Control")
    
    try:
        print("🚀 Starting MetaCub Dashboard...")
        
        session_id = uuid4()
        
        control_reader = ControlLoopReader(
            action_interface=ActionInterface(
                remote_prefix="/metaControllClient",
                local_prefix=f"/metacub_dashboard/{session_id}",
                control_boards=["neck", "left_arm", "right_arm", "fingers"],
                stream_name="actions"
            ),
            observation_interfaces={
                'agentview': CameraInterface(
                    remote_prefix="/ergocubSim",
                    local_prefix=f"/metacub_dashboard/{session_id}",
                    rgb_shape=(640, 480),
                    depth_shape=None,
                    stream_name="agentview"
                ),
                'encoders': EncodersInterface(
                    remote_prefix="/ergocubSim",
                    local_prefix=f"/metacub_dashboard/{session_id}",
                    control_boards=["head", "left_arm", "right_arm", "torso"],
                    stream_name="encoders"
                )
            }
        )

        # Visualization setup - let visualizer handle URDF loading and blueprint creation
        visualizer = Visualizer(gradio=False)
        eef_paths = visualizer.eef_paths  # Get the eef_paths from the auto-setup

        # Data logging setup
        print("💾 Setting up data logging...")
        base_data_logger = DataLogger(
            path="assets/beer_data.zarr.zip",
            flush_every=100,
            exist_ok=True
        )
        data_logger = PolarsDataLogger(base_data_logger)
        print("✅ Data logger created successfully")

        # Initialize episode counter from existing dataset
        episode_count = data_logger.get_episode_count()
        if episode_count > 0:
            print(f"📊 Found {episode_count} existing episodes in dataset")
        
        while True:  # Main application loop
            # Auto-start next episode
            current_episode = episode_count + 1
            
            keyboard.set_episode_state("RESETTING")
            keyboard.update_status("Resetting the robot...")
            print("🔄 Resetting the robot...")
            
            control_reader.reset()

            keyboard.set_episode_state("READY")
            keyboard.update_status("Start the episode by controlling the robot")
            control_data = control_reader.read()

            print(f"🔄 Episode {current_episode} started - Recording data...")
            keyboard.update_status(f"Episode {current_episode} - Recording... (Space: keep & end, Delete: discard & end, q: quit)")
            keyboard.set_episode_state("RECORDING")
            
            iteration = 0
            start_episode_time = time.perf_counter()
            episode_decision = None
            
            while True:  # Episode loop
                # Check for episode control commands
                command = keyboard.get_command()
                if command == 'space':
                    print(f"⏹️  Episode {current_episode} ended - keeping data")
                    episode_decision = 'keep'
                    break
                elif command == 'delete':
                    print(f"⏹️  Episode {current_episode} ended - discarding data")
                    episode_decision = 'discard'
                    break
                
                control_data = control_reader.read()
                    
                # ====== DATAFRAME PROCESSING ======
                
                # Extract DataFrames directly
                action_df = control_data.actions_df
                observation_df = control_data.observations_df
                
                # Apply entity paths using when/then/otherwise
                all_observation_df = observation_df.with_columns([
                    pl.when(pl.col("name").str.contains("left_arm"))
                    .then(pl.lit(f"{eef_paths[0]}/fingers"))
                    .when(pl.col("name").str.contains("right_arm"))  
                    .then(pl.lit(f"{eef_paths[1]}/fingers"))
                    .otherwise(pl.lit("joints"))
                    .alias("entity_path")
                ])
                
                # Split processed data by stream type
                camera_df = all_observation_df.filter(pl.col("stream_type") == "camera")
                encoder_df = all_observation_df.filter(pl.col("stream_type") == "encoders")
                
                # Log to visualizer
                visualizer.log_dataframes(
                    poses_df=action_df,
                    encoders_df=encoder_df,
                    camera_df=camera_df,
                    timestamp=time.perf_counter() - start_episode_time,
                    static=False,
                )
                
                # Data logging (background processing)
                data_logger.log_dataframes(
                    observations_df=all_observation_df,
                    actions_df=action_df
                )
                
                iteration += 1
                
                # Update status periodically to show progress
                if iteration % 100 == 0:
                    keyboard.update_status(f"Episode {current_episode} - Recording... (iter: {iteration})")
            
            # Episode cleanup
            print(f"🧹 Cleaning up Episode {current_episode}...")
            
            # Handle episode decision
            if episode_decision == 'keep':
                print(f"✅ Keeping episode - saving as Episode {episode_count + 1}...")
                keyboard.update_status(f"Episode {episode_count + 1} - Saving...")
                data_logger.end_episode()
                print(f"💾 Episode {episode_count + 1} saved successfully!")
                episode_count += 1
            elif episode_decision == 'discard':
                print("🗑️  Discarding episode - data will not be saved...")
                keyboard.update_status(f"Episode {current_episode} - Discarding...")
                data_logger.discard_episode()
                print(f"🗑️  Episode {current_episode} discarded!")
            elif episode_decision == 'quit':
                print("🗑️  Discarding current episode and quitting...")
                data_logger.discard_episode()
                break
            
            # Continue to next episode (go back to main loop)
        

        print("🧹 Final cleanup...")
        cleanup_resources()
        print("✅ Clean shutdown complete!")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🛑 Application terminated due to an error")
        
        # Final cleanup in case of error
        try:
            cleanup_resources()
        except Exception as cleanup_e:
            print(f"⚠️  Cleanup error: {cleanup_e}")
        
        raise  # Re-raise the exception after cleanup


if __name__ == "__main__":
    main()
