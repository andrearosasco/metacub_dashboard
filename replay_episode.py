#!/usr/bin/env python3
"""
Episode Replay Script for MetaCub Dashboard
Loads a saved zarr.zip dataset and replays it in the visualizer.
"""
import time
import os
import argparse
from pathlib import Path
import polars as pl
import zarr
from zarr.storage import ZipStore
import yarp
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core

# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"

from src.metacub_dashboard.visualizer.visualizer import Visualizer
from src.metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from src.metacub_dashboard.utils.keyboard_interface import KeyboardInterface, StatusAwarePrinter

urdf_parser_core.on_error = lambda x: x


def load_episode_data(zarr_path: str):
    """Load episode data from zarr.zip file."""
    print(f"📂 Loading episode data from: {zarr_path}")
    
    # Open zarr store
    store = ZipStore(zarr_path, mode='r')
    root = zarr.group(store=store)
    
    # Load observations and actions
    observations_data = []
    actions_data = []
    
    # Load observations
    if 'observations' in root:
        obs_group = root['observations']
        for key in obs_group.keys():
            obs_array = obs_group[key]
            # Convert zarr array to list of records
            for i in range(len(obs_array)):
                record = obs_array[i]
                observations_data.append({
                    'stream_name': key,
                    'timestamp': record.get('timestamp', i * 0.1),  # Default 10Hz if no timestamp
                    'data': record
                })
    
    # Load actions  
    if 'actions' in root:
        actions_group = root['actions']
        for key in actions_group.keys():
            action_array = actions_group[key]
            # Convert zarr array to list of records
            for i in range(len(action_array)):
                record = action_array[i]
                actions_data.append({
                    'stream_name': key,
                    'timestamp': record.get('timestamp', i * 0.1),  # Default 10Hz if no timestamp
                    'data': record
                })
    
    # Convert to DataFrames
    observations_df = pl.DataFrame(observations_data) if observations_data else pl.DataFrame()
    actions_df = pl.DataFrame(actions_data) if actions_data else pl.DataFrame()
    
    print(f"✅ Loaded {len(observations_df)} observations, {len(actions_df)} actions")
    return observations_df, actions_df


def setup_visualizer():
    """Set up the visualizer with proper URDF and blueprint."""
    print("🎨 Setting up visualizer...")
    
    # Initialize YARP (needed for ResourceFinder)
    yarp.Network.init()
    
    # Load URDF
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
    return visualizer, eef_paths


def process_dataframes_for_replay(observations_df, actions_df, eef_paths):
    """Process the loaded DataFrames to match the expected format for visualization."""
    
    if observations_df.is_empty():
        print("⚠️  No observations data found")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    
    # Add entity paths to observations (similar to main.py processing)
    processed_observations_df = observations_df.with_columns([
        pl.when(pl.col("stream_name").str.contains("left_arm"))
        .then(pl.lit(f"{eef_paths[0]}/fingers"))
        .when(pl.col("stream_name").str.contains("right_arm"))  
        .then(pl.lit(f"{eef_paths[1]}/fingers"))
        .otherwise(pl.lit("joints"))
        .alias("entity_path")
    ])
    
    # Split by stream type (assuming we can infer this from stream_name)
    camera_df = processed_observations_df.filter(
        pl.col("stream_name").str.contains("camera|agentview")
    )
    encoder_df = processed_observations_df.filter(
        pl.col("stream_name").str.contains("encoder|joint")
    )
    
    return actions_df, encoder_df, camera_df


def replay_episode(zarr_path: str, playback_speed: float = 1.0):
    """Replay an episode from a zarr.zip file."""
    
    # Set up keyboard interface
    print("⌨️  Setting up replay interface...")
    keyboard = KeyboardInterface("MetaCub Dashboard - Episode Replay")
    printer = StatusAwarePrinter(keyboard)
    
    # Replace built-in print with status-aware print
    import builtins
    builtins.print = printer.print
    
    print(f"🎬 Starting Episode Replay: {Path(zarr_path).name}")
    
    try:
        # Load episode data
        observations_df, actions_df = load_episode_data(zarr_path)
        
        if observations_df.is_empty() and actions_df.is_empty():
            print("❌ No data found in the episode file")
            return
        
        # Set up visualizer
        visualizer, eef_paths = setup_visualizer()
        
        # Process DataFrames for visualization
        actions_df, encoder_df, camera_df = process_dataframes_for_replay(
            observations_df, actions_df, eef_paths
        )
        
        # Get timestamps and sort data chronologically
        all_timestamps = []
        if not actions_df.is_empty() and 'timestamp' in actions_df.columns:
            all_timestamps.extend(actions_df['timestamp'].to_list())
        if not encoder_df.is_empty() and 'timestamp' in encoder_df.columns:
            all_timestamps.extend(encoder_df['timestamp'].to_list())
        if not camera_df.is_empty() and 'timestamp' in camera_df.columns:
            all_timestamps.extend(camera_df['timestamp'].to_list())
        
        if not all_timestamps:
            print("❌ No timestamp data found - cannot replay")
            return
        
        unique_timestamps = sorted(set(all_timestamps))
        total_frames = len(unique_timestamps)
        
        print(f"🔄 Replaying {total_frames} frames...")
        print("📋 Replay controls:")
        print("   'q' - Quit replay")
        print("   'r' - Restart replay")
        print("   ' ' (space) - Pause/Resume")
        
        keyboard.update_status("Ready - Starting replay...")
        keyboard.set_episode_state("REPLAYING")
        
        frame_idx = 0
        is_paused = False
        start_time = time.perf_counter()
        
        while frame_idx < total_frames:
            # Check for commands
            command = keyboard.get_command()
            if command == 'quit':
                print("👋 Replay stopped by user")
                break
            elif command == 'reset':
                print("🔄 Restarting replay...")
                frame_idx = 0
                start_time = time.perf_counter()
                continue
            elif command == 'start':  # Use 's' as space alternative for pause/resume
                is_paused = not is_paused
                if is_paused:
                    print("⏸️  Replay paused")
                    keyboard.set_episode_state("PAUSED")
                else:
                    print("▶️  Replay resumed")
                    keyboard.set_episode_state("REPLAYING")
            
            if is_paused:
                time.sleep(0.1)
                continue
            
            current_timestamp = unique_timestamps[frame_idx]
            
            # Get data for current timestamp
            frame_actions_df = actions_df.filter(pl.col("timestamp") == current_timestamp) if not actions_df.is_empty() else pl.DataFrame()
            frame_encoder_df = encoder_df.filter(pl.col("timestamp") == current_timestamp) if not encoder_df.is_empty() else pl.DataFrame()
            frame_camera_df = camera_df.filter(pl.col("timestamp") == current_timestamp) if not camera_df.is_empty() else pl.DataFrame()
            
            # Log to visualizer
            visualizer.log_dataframes(
                poses_df=frame_actions_df,
                encoders_df=frame_encoder_df,
                camera_df=frame_camera_df,
                timestamp=current_timestamp,
                static=False,
            )
            
            # Update progress
            progress = (frame_idx + 1) / total_frames * 100
            keyboard.update_status(f"Replaying... {progress:.1f}% (frame {frame_idx+1}/{total_frames})")
            
            frame_idx += 1
            
            # Control playback speed
            expected_time = current_timestamp / playback_speed
            actual_time = time.perf_counter() - start_time
            sleep_time = expected_time - actual_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        if frame_idx >= total_frames:
            print("✅ Replay completed!")
            keyboard.update_status("Replay completed - Press 'r' to restart or 'q' to quit")
            keyboard.set_episode_state("COMPLETED")
            
            # Wait for user command after completion
            while True:
                command = keyboard.get_command()
                if command == 'quit':
                    break
                elif command == 'reset':
                    # Restart replay
                    replay_episode(zarr_path, playback_speed)
                    break
                time.sleep(0.1)
    
    except Exception as e:
        print(f"❌ Error during replay: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🧹 Cleaning up...")
        keyboard.close()
        yarp.Network.fini()
        print("✅ Replay cleanup complete!")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Replay MetaCub Dashboard episodes from zarr.zip files")
    parser.add_argument("episode_path", nargs='?', help="Path to the episode zarr.zip file")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                       help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available episode files in assets/ directory")
    
    args = parser.parse_args()
    
    if args.list:
        # List available episodes
        assets_dir = Path("assets")
        if assets_dir.exists():
            episode_files = list(assets_dir.glob("episode_*.zarr.zip"))
            if episode_files:
                print("📁 Available episodes:")
                for ep_file in sorted(episode_files):
                    print(f"   {ep_file}")
            else:
                print("📁 No episode files found in assets/ directory")
        else:
            print("📁 assets/ directory not found")
        return
    
    if not args.episode_path:
        parser.error("episode_path is required when not using --list")
    
    episode_path = Path(args.episode_path)
    
    if not episode_path.exists():
        print(f"❌ Episode file not found: {episode_path}")
        return
    
    if not episode_path.suffix == '.zip' or 'zarr' not in episode_path.stem:
        print(f"❌ Invalid file format. Expected .zarr.zip file, got: {episode_path}")
        return
    
    print(f"🚀 Starting episode replay...")
    replay_episode(str(episode_path), args.speed)


if __name__ == "__main__":
    main()
