"""
MetaCub Dashboard.
"""
import time
import os
from uuid import uuid4

# Set environment before importing yarp
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
                    remote_prefix="",
                    local_prefix=f"/metacub_dashboard/{session_id}",
                    rgb_shape=(640, 480),
                    depth_shape=None,
                    stream_name="agentview"
                ),
                'encoders': EncodersInterface(
                    remote_prefix="/ergocub",
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
            path="assets/demo_data.zarr.zip",
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
            # Wait for start command (blocking)
            keyboard.update_status(f"Ready - Press 's' for new episode (next: {episode_count + 1})")
            keyboard.set_episode_state("READY")
            command = keyboard.get_command(blocking=True)
            if command == 'start':
                current_episode = episode_count + 1  # Preview next episode number               
            elif command == 'quit':
                break

            keyboard.set_episode_state("RESETTING")
            keyboard.update_status("Resetting the robot...")
            print("🔄 Resetting the robot...")
            
            control_reader.reset()

            keyboard.update_status(f"Episode {current_episode} - Ready")
            keyboard.set_episode_state("READY")

            control_data = control_reader.read()
            print(f"🔄 Episode {current_episode} started - Recording data...")
            keyboard.update_status(f"Episode {current_episode} - Recording...")
            keyboard.set_episode_state("RECORDING")
            
            iteration = 0
            start_episode_time = time.perf_counter()
            
            while True:  # Episode loop
                # Check for episode control commands
                command = keyboard.get_command()
                if command == 'end':
                    print(f"⏹️  Episode {current_episode} ended by user")
                    keyboard.update_status(f"Episode {current_episode} - Ending...")
                    keyboard.set_episode_state("FINISHED")
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
            
            # Ask user if they want to keep or discard the episode
            keyboard.update_status(f"Episode {current_episode} - Keep (k) or Discard (d)?")
            
            # Wait for keep/discard decision (blocking)
            decision = keyboard.get_command(blocking=True)
            if decision == 'keep':
                # Increment episode counter only when saving
                print(f"✅ Keeping episode - saving as Episode {episode_count + 1}...")
                keyboard.update_status(f"Episode {episode_count + 1} - Saving...")
                data_logger.end_episode()
                print(f"💾 Episode {episode_count + 1} saved successfully!")
                episode_count += 1
            elif decision == 'discard':
                print("🗑️  Discarding episode - data will not be saved...")
                keyboard.update_status(f"Episode {current_episode} - Discarding...")
                data_logger.discard_episode()
                print(f"🗑️  Episode {current_episode} discarded!")
            
            # Continue to next episode (go back to main loop)
        

        print("🧹 Final cleanup...")
        keyboard.close()
        control_reader.close()
        print("✅ Clean shutdown complete!")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🛑 Application terminated due to an error")
        
        # Final cleanup in case of error
        try:
            keyboard.close()
            control_reader.close()
            print("✅ Cleanup complete")
        except Exception as cleanup_e:
            print(f"⚠️  Cleanup error: {cleanup_e}")
        
        raise  # Re-raise the exception after cleanup


if __name__ == "__main__":
    main()
