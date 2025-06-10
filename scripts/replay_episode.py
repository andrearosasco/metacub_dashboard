#!/usr/bin/env python3
"""
Episode Replay Script for MetaCub Dashboard
Loads a saved zarr.zip dataset and replays it in the visualizer.
"""
import time
import os
# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import argparse
from pathlib import Path
import polars as pl
import zarr
import yarp
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core

from metacub_dashboard.visualizer.visualizer import Visualizer
from metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from metacub_dashboard.utils.keyboard_interface import KeyboardInterface, StatusAwarePrinter

urdf_parser_core.on_error = lambda x: x


def load_episode_data(zarr_path: str, episode_idx: int = 0):
    """Load episode data from zarr.zip file."""
    print(f"📂 Loading episode data from: {zarr_path}")
    
    # Open zarr file directly
    root = zarr.open(zarr_path, mode='r')
    
    # Get episode boundaries
    episode_lengths = root['episode_length'][:]
    if episode_idx >= len(episode_lengths):
        print(f"❌ Episode {episode_idx} not found. Available episodes: 0-{len(episode_lengths)-1}")
        return pl.DataFrame(), pl.DataFrame()
    
    # Calculate episode boundaries
    start_idx = sum(episode_lengths[:episode_idx])
    end_idx = start_idx + episode_lengths[episode_idx]
    
    print(f"📊 Loading episode {episode_idx}: frames {start_idx}-{end_idx-1} (length: {episode_lengths[episode_idx]})")
    print(f"📋 Available episodes: {len(episode_lengths)} total")
    
    # Load observations data
    observations_data = []
    actions_data = []
    
    # Process observations
    if 'obs' in root:
        obs_group = root['obs']
        for key in obs_group.keys():
            obs_array = obs_group[key]
            episode_data = obs_array[start_idx:end_idx]
            
            # Convert to DataFrame records
            for i, data in enumerate(episode_data):
                observations_data.append({
                    'stream_name': key,
                    'timestamp': (start_idx + i) * 0.1,  # Assume 10Hz
                    'data': data
                })
    
    # Process actions
    if 'action' in root:
        action_group = root['action']
        for key in action_group.keys():
            action_array = action_group[key]
            episode_data = action_array[start_idx:end_idx]
            
            # Convert to DataFrame records
            for i, data in enumerate(episode_data):
                actions_data.append({
                    'stream_name': key,
                    'timestamp': (start_idx + i) * 0.1,  # Assume 10Hz
                    'data': data
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
    
    if observations_df.is_empty() and actions_df.is_empty():
        print("⚠️  No data found")
        return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()
    
    # Process observations data
    camera_records = []
    encoder_records = []
    
    if not observations_df.is_empty():
        for row in observations_df.iter_rows(named=True):
            stream_name = row['stream_name']
            timestamp = row['timestamp']
            data = row['data']
            
            if 'rgb' in stream_name or 'agentview' in stream_name:
                # Camera data - store as base data that will be processed later
                camera_records.append({
                    'timestamp': timestamp,
                    'entity_path': 'realsense_depth_frame/cameras/agentview_rgb',
                    'stream_type': 'camera',
                    'name': stream_name,
                    'data_idx': len(camera_records),  # Use index to reference original data
                })
            elif 'encoders' in stream_name:
                # Encoder data - map to appropriate entity path
                if 'left_arm' in stream_name:
                    entity_path = f"{eef_paths[0]}/encoders"
                elif 'right_arm' in stream_name:
                    entity_path = f"{eef_paths[1]}/encoders"  
                elif 'head' in stream_name:
                    entity_path = "head/encoders"
                elif 'torso' in stream_name:
                    entity_path = "torso/encoders"
                else:
                    entity_path = "robot_joints/encoders"
                
                encoder_records.append({
                    'timestamp': timestamp,
                    'entity_path': entity_path,
                    'stream_type': 'encoders',
                    'name': stream_name,
                    'data_idx': len(encoder_records),  # Use index to reference original data
                })
    
    # Process actions data
    pose_records = []
    if not actions_df.is_empty():
        for row in actions_df.iter_rows(named=True):
            stream_name = row['stream_name']
            timestamp = row['timestamp']  
            data = row['data']
            
            # Map action streams to entity paths
            if stream_name == 'left_arm':
                entity_path = f"{eef_paths[0]}/target_poses"
            elif stream_name == 'right_arm':
                entity_path = f"{eef_paths[1]}/target_poses"
            elif stream_name == 'fingers':
                entity_path = "fingers/target_poses"
            elif stream_name == 'neck':
                entity_path = "neck/target_poses"
            else:
                entity_path = "robot_joints/target_poses"
            
            pose_records.append({
                'timestamp': timestamp,
                'entity_path': entity_path,
                'stream_type': 'poses',
                'name': stream_name,
                'data_idx': len(pose_records),  # Use index to reference original data
            })
    
    # Create DataFrames with basic data
    camera_df = pl.DataFrame(camera_records) if camera_records else pl.DataFrame()
    encoder_df = pl.DataFrame(encoder_records) if encoder_records else pl.DataFrame()
    poses_df = pl.DataFrame(pose_records) if pose_records else pl.DataFrame()
    
    print(f"📊 Processed data: {len(camera_df)} camera frames, {len(encoder_df)} encoder readings, {len(poses_df)} poses")
    
    return poses_df, encoder_df, camera_df


def log_episode_data_to_visualizer(
    visualizer, 
    frame_actions_df, 
    frame_encoder_df, 
    frame_camera_df,
    observations_df,
    actions_df, 
    current_timestamp
):
    """Log episode data directly to the visualizer bypassing the DataFrame processing."""
    import rerun as rr
    import numpy as np
    from metacub_dashboard.visualizer.visualizer import Pose
    from scipy.spatial.transform import Rotation as R
    
    # Set time for this frame
    visualizer.rec.set_time("real_time", duration=current_timestamp)
    
    # Process and log camera data
    if not frame_camera_df.is_empty():
        for row in frame_camera_df.iter_rows(named=True):
            # Get the actual image data from observations_df
            matching_obs = observations_df.filter(
                (pl.col("timestamp") == current_timestamp) & 
                (pl.col("stream_name").str.contains("rgb"))
            )
            if not matching_obs.is_empty():
                image_data = matching_obs.row(0, named=True)['data']
                entity_path = row['entity_path']
                visualizer.rec.log(entity_path, rr.Image(image_data))
    
    # Process and log encoder data
    if not frame_encoder_df.is_empty():
        for row in frame_encoder_df.iter_rows(named=True):
            stream_name = row['name']
            entity_path = row['entity_path']
            
            # Get the actual encoder data from observations_df
            matching_obs = observations_df.filter(
                (pl.col("timestamp") == current_timestamp) & 
                (pl.col("stream_name") == stream_name)
            )
            if not matching_obs.is_empty():
                encoder_data = matching_obs.row(0, named=True)['data']
                
                # Determine joint labels based on the stream name
                if 'left_arm' in stream_name:
                    joint_labels = ["l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow", 
                                   "l_wrist_yaw", "l_wrist_roll", "l_wrist_pitch", "l_thumb_add", 
                                   "l_thumb_prox", "l_index_add", "l_index_prox", "l_middle_prox", "l_ring_prox"]
                elif 'head' in stream_name:
                    joint_labels = ["neck_pitch", "neck_roll", "neck_yaw", "camera_tilt"]
                elif 'torso' in stream_name:
                    joint_labels = ["torso_roll", "torso_pitch", "torso_yaw"]
                else:
                    joint_labels = [f"joint_{i}" for i in range(len(encoder_data))]
                
                # Move the robot joints
                for joint_name, angle in zip(joint_labels, encoder_data):
                    visualizer.urdf_logger.log(joint_name, float(angle))
                    # Also log as scalar for the blueprint
                    visualizer.rec.log(f'{entity_path}/{joint_name}', rr.Scalars(float(angle)))
    
    # Process and log action/pose data
    if not frame_actions_df.is_empty():
        for row in frame_actions_df.iter_rows(named=True):
            stream_name = row['name']
            entity_path = row['entity_path']
            
            # Get the actual action data from actions_df
            matching_action = actions_df.filter(
                (pl.col("timestamp") == current_timestamp) & 
                (pl.col("stream_name") == stream_name)
            )
            if not matching_action.is_empty():
                action_data = matching_action.row(0, named=True)['data']
                
                if stream_name == 'left_arm' or stream_name == 'right_arm':
                    # For arm actions, treat as 7-DOF pose (position + quaternion)
                    if len(action_data) >= 7:
                        pos = action_data[:3]
                        quat = action_data[3:7]  # Assuming quaternion format
                        
                        # Convert quaternion to rotation matrix
                        ori = R.from_quat(quat).as_matrix()
                        
                        pose = Pose(pos=pos, ori=ori)
                        
                        # Log the pose frame
                        visualizer.rec.log(f'{entity_path}', rr.Transform3D(translation=pose.pos, mat3x3=pose.ori))
                        visualizer.rec.log(f'{entity_path}/axes', rr.Arrows3D(
                            origins=np.zeros([3, 3]), 
                            vectors=np.eye(3) * 0.1, 
                            colors=np.eye(3)
                        ))
                        
                        # Log pose components as scalars
                        pose_array = np.concatenate([pose.pos, R.from_matrix(pose.ori).as_rotvec(), [0]])
                        pose_names = ['x', 'y', 'z', 'ax', 'ay', 'az', 'g']
                        for n, p in zip(pose_names, pose_array):
                            visualizer.rec.log(f'{entity_path}/components/{n}', rr.Scalars(float(p)))
                
                elif stream_name == 'fingers':
                    # For finger actions, log as individual finger poses
                    if len(action_data.shape) >= 2 and action_data.shape[0] >= 10:
                        finger_names = ['l_thumb', 'l_index', 'l_middle', 'l_ring', 'l_pinky',
                                      'r_thumb', 'r_index', 'r_middle', 'r_ring', 'r_pinky']
                        
                        for i, finger_name in enumerate(finger_names):
                            if i < action_data.shape[0]:
                                finger_pos = action_data[i][:3] if action_data.shape[1] >= 3 else action_data[i]
                                finger_path = f"{entity_path}/{finger_name}"
                                visualizer.rec.log(finger_path, rr.Points3D([finger_pos], radii=[0.01]))
                
                elif stream_name == 'neck':
                    # For neck actions, treat as orientation matrix
                    if len(action_data) >= 9:
                        ori = action_data.reshape(3, 3)
                        pose = Pose(pos=np.array([0, 0, 0]), ori=ori)
                        
                        visualizer.rec.log(f'{entity_path}', rr.Transform3D(translation=pose.pos, mat3x3=pose.ori))
                        
                        # Log orientation components
                        rotvec = R.from_matrix(pose.ori).as_rotvec()
                        for i, component in enumerate(['ax', 'ay', 'az']):
                            visualizer.rec.log(f'{entity_path}/components/{component}', rr.Scalars(float(rotvec[i])))


def replay_episode(zarr_path: str, episode_idx: int = 0, playback_speed: float = 1.0):
    """Replay an episode from a zarr.zip file."""
    
    # Set up keyboard interface
    print("⌨️  Setting up replay interface...")
    keyboard = KeyboardInterface("MetaCub Dashboard - Episode Replay")
    printer = StatusAwarePrinter(keyboard)
    
    # Replace built-in print with status-aware print
    import builtins
    builtins.print = printer.print
    
    print(f"🎬 Starting Episode Replay: {Path(zarr_path).name} (Episode {episode_idx})")
    
    try:
        # Load episode data
        observations_df, actions_df_raw = load_episode_data(zarr_path, episode_idx)
        
        if observations_df.is_empty() and actions_df_raw.is_empty():
            print("❌ No data found in the episode file")
            return
        
        # Set up visualizer
        visualizer, eef_paths = setup_visualizer()
        
        # Process DataFrames for visualization
        processed_actions_df, encoder_df, camera_df = process_dataframes_for_replay(
            observations_df, actions_df_raw, eef_paths
        )
        
        # Get timestamps and sort data chronologically
        all_timestamps = []
        if not processed_actions_df.is_empty() and 'timestamp' in processed_actions_df.columns:
            all_timestamps.extend(processed_actions_df['timestamp'].to_list())
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
        print("   's' - Pause/Resume")
        
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
            elif command == 'start':  # 's' key for pause/resume
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
            frame_actions_df = processed_actions_df.filter(pl.col("timestamp") == current_timestamp) if not processed_actions_df.is_empty() else pl.DataFrame()
            frame_encoder_df = encoder_df.filter(pl.col("timestamp") == current_timestamp) if not encoder_df.is_empty() else pl.DataFrame()
            frame_camera_df = camera_df.filter(pl.col("timestamp") == current_timestamp) if not camera_df.is_empty() else pl.DataFrame()
            
            # Log to visualizer directly instead of using log_dataframes
            log_episode_data_to_visualizer(
                visualizer,
                frame_actions_df,
                frame_encoder_df, 
                frame_camera_df,
                observations_df,  # Pass original observations
                actions_df_raw,   # Pass original actions
                current_timestamp
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
                    replay_episode(zarr_path, episode_idx, playback_speed)
                    break
                time.sleep(0.1)
    
    except Exception as e:
        raise e
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
    parser.add_argument("--episode", "-e", type=int, default=0,
                       help="Episode index to replay (default: 0)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, 
                       help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available episode files in assets/ directory")
    
    args = parser.parse_args()
    
    if args.list:
        # List available episodes
        assets_dir = Path("assets")
        if assets_dir.exists():
            episode_files = list(assets_dir.glob("*.zarr.zip"))
            if episode_files:
                print("📁 Available episodes:")
                for ep_file in sorted(episode_files):
                    print(f"   {ep_file}")
                    # Show episode info
                    try:
                        root = zarr.open(str(ep_file), mode='r')
                        episode_lengths = root['episode_length'][:]
                        print(f"      Contains {len(episode_lengths)} episodes with lengths: {episode_lengths.tolist()}")
                    except Exception as e:
                        print(f"      Error reading file: {e}")
            else:
                print("📁 No zarr.zip files found in assets/ directory")
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
    replay_episode(str(episode_path), args.episode, args.speed)


if __name__ == "__main__":
    main()
