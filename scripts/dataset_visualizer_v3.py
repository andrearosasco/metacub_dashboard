#!/usr/bin/env python3
"""
Dataset Visualizer for MetaCub Dashboard
Loads episodes from zarr.zip files and replays them in the visualizer.
"""
import argparse
import time
import os
from pathlib import Path
import numpy as np
import zarr
import polars as pl

# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp

from metacub_dashboard.visualizer.visualizer import Visualizer
from metacub_dashboard.utils.keyboard_interface import KeyboardInterface
from metacub_dashboard.interfaces.interfaces import STREAM_SCHEMA


class DatasetVisualizer:
    """Visualizer for zarr dataset episodes."""
    
    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.visualizer = None
        self.keyboard = None
        
        # Load dataset
        self.root = zarr.open(zarr_path, mode='r')
        self.episode_lengths = self.root['episode_length'][:]
        self.obs_group = self.root['obs']
        self.action_group = self.root['action'] if 'action' in self.root else None
        
        print(f"📂 Loaded dataset: {zarr_path}")
        print(f"📊 Found {len(self.episode_lengths)} episodes")
        print(f"🔍 Observation streams: {list(self.obs_group.keys())}")
        if self.action_group:
            print(f"🎮 Action streams: {list(self.action_group.keys())}")
        
        # Initialize YARP for visualizer
        yarp.Network.init()
        
    def get_episode_frames(self, episode_idx: int):
        """Get the frame range for a specific episode."""
        if episode_idx >= len(self.episode_lengths):
            raise ValueError(f"Episode {episode_idx} not found (max: {len(self.episode_lengths) - 1})")
        
        episode_length = self.episode_lengths[episode_idx]
        if episode_length == 0:
            raise ValueError(f"Episode {episode_idx} is empty")
        
        # Calculate frame range
        start_frame = sum(self.episode_lengths[:episode_idx])
        end_frame = start_frame + episode_length
        
        return start_frame, end_frame, episode_length
    
    def frame_to_dataframes(self, frame_idx: int):
        """Convert single frame data to Polars DataFrames."""
        observations_list = []
        actions_list = []
        
        timestamp = frame_idx * 0.1  # Assume 10Hz
        
        # ===== OBSERVATION DATA =====
        
        # Camera data
        if 'agentview_rgb' in self.obs_group:
            rgb_image = self.obs_group['agentview_rgb'][frame_idx]
            obs_row = {
                "name": "agentview_rgb",
                "stream_type": "camera",
                "entity_path": "",
                "data": {"rgb": rgb_image},
                "metadata": {
                    "timestamp": timestamp,
                    "seq_number": frame_idx,
                    "read_timestamp": timestamp,
                    "read_delay": 0.0,
                    "read_attempts": 1,
                    "frequency": 10.0,
                }
            }
            observations_list.append(obs_row)
        
        # Encoder data for each board
        joint_mappings = {
            "head": ['neck_pitch', 'neck_roll', 'neck_yaw', 'camera_tilt'],
            "left_arm": ['l_shoulder_pitch', 'l_shoulder_roll', 'l_shoulder_yaw', 'l_elbow', 'l_wrist_yaw',
                         'l_wrist_roll', 'l_wrist_pitch', "l_thumb_add", "l_thumb_prox", "l_index_add",
                         "l_index_prox", "l_middle_prox", "l_ring_prox"],
            "right_arm": ['r_shoulder_pitch', 'r_shoulder_roll', 'r_shoulder_yaw', 'r_elbow', 'r_wrist_yaw',
                          'r_wrist_roll', 'r_wrist_pitch', "r_thumb_add", "r_thumb_prox", "r_index_add",
                          "r_index_prox", "r_middle_prox", "r_ring_prox"],
            "torso": ["torso_roll", "torso_pitch", "torso_yaw"],
        }
        
        for board_name, joint_labels in joint_mappings.items():
            obs_key = f"encoders_{board_name}"
            if obs_key in self.obs_group:
                encoder_values = self.obs_group[obs_key][frame_idx]
                
                obs_row = {
                    "name": f"encoders_{board_name}",
                    "stream_type": "encoders",
                    "entity_path": "",  # Will be set by processing logic
                    "data": {
                        board_name: {
                            'values': encoder_values,
                            'labels': joint_labels[:len(encoder_values)]  # Match length
                        }
                    },
                    "metadata": {
                        "timestamp": timestamp,
                        "seq_number": frame_idx,
                        "read_timestamp": timestamp,
                        "read_delay": 0.0,
                        "read_attempts": 1,
                        "frequency": 10.0,
                    }
                }
                observations_list.append(obs_row)
        
        # ===== ACTION DATA =====
        
        if self.action_group is not None:
            # Collect all action data for this frame
            action_data = {}
            for action_key in self.action_group.keys():
                action_values = self.action_group[action_key][frame_idx]
                action_data[action_key] = action_values
            
            # Create single action row with all pose data
            if action_data:
                action_row = {
                    "name": "actions",
                    "stream_type": "poses",
                    "entity_path": "",
                    "data": action_data,
                    "metadata": {
                        "timestamp": timestamp,
                        "seq_number": frame_idx,
                        "read_timestamp": timestamp,
                        "read_delay": 0.0,
                        "read_attempts": 1,
                        "frequency": 10.0,
                    }
                }
                actions_list.append(action_row)
        
        # Create DataFrames with the correct schema
        observations_df = pl.DataFrame()
        actions_df = pl.DataFrame()
        
        if observations_list:
            # Create observations DataFrame with proper schema
            observations_df = pl.DataFrame(observations_list, schema=STREAM_SCHEMA)
        
        if actions_list:
            # Create actions DataFrame with proper schema
            actions_df = pl.DataFrame(actions_list, schema=STREAM_SCHEMA)
        
        return observations_df, actions_df
    
    def visualize_episode(self, episode_idx: int, playback_speed: float = 1.0):
        """Visualize a specific episode."""
        try:
            # Get episode frame range
            start_frame, end_frame, episode_length = self.get_episode_frames(episode_idx)
            print(f"🎬 Visualizing Episode {episode_idx}: frames {start_frame}-{end_frame-1} (length: {episode_length})")
            
            # Setup visualizer
            if self.visualizer is None:
                print("🎨 Setting up visualizer...")
                self.visualizer = Visualizer(gradio=False)
                print("✅ Visualizer ready")
            
            # Setup keyboard interface
            if self.keyboard is None:
                self.keyboard = KeyboardInterface("Dataset Visualizer")
                self.keyboard.set_episode_state("REPLAYING")
            
            # Get end-effector paths from visualizer
            eef_paths = self.visualizer.eef_paths
            
            # Replay episode
            print(f"🔄 Replaying {episode_length} frames at {playback_speed}x speed...")
            frame_duration = 0.1 / playback_speed  # Assume 10Hz base rate
            
            start_time = time.time()
            paused = False
            frame = 0
            
            while frame < episode_length:
                # Check for user input
                command = self.keyboard.get_command()
                if command == 'q':
                    print("🛑 Stopping replay...")
                    break
                elif command == 's' or command == 'space':
                    paused = not paused
                    if paused:
                        print("⏸️  Paused (press 's' or space to resume)")
                        self.keyboard.update_status("PAUSED - Press 's' or space to resume")
                    else:
                        print("▶️  Resumed")
                        self.keyboard.update_status(f"Replaying Episode {episode_idx} - Frame {frame}/{episode_length}")
                elif command == 'r':
                    print("🔄 Restarting replay...")
                    frame = 0
                    start_time = time.time()
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                # Update status
                if frame % 10 == 0:
                    self.keyboard.update_status(f"Replaying Episode {episode_idx} - Frame {frame}/{episode_length}")
                
                # Get data for current frame
                current_frame_idx = start_frame + frame
                observations_df, actions_df = self.frame_to_dataframes(current_frame_idx)
                
                # Process dataframes to add entity paths (mimicking main.py logic)
                if not observations_df.is_empty():
                    processed_observations_df = observations_df.with_columns([
                        pl.when(pl.col("name").str.contains("left_arm"))
                        .then(pl.lit(f"{eef_paths[0]}/fingers"))
                        .when(pl.col("name").str.contains("right_arm"))  
                        .then(pl.lit(f"{eef_paths[1]}/fingers"))
                        .otherwise(pl.lit("joints"))
                        .alias("entity_path")
                    ])
                else:
                    processed_observations_df = observations_df
                
                # Split data by type
                camera_df = processed_observations_df.filter(pl.col("stream_type") == "camera") if not processed_observations_df.is_empty() else pl.DataFrame()
                encoder_df = processed_observations_df.filter(pl.col("stream_type") == "encoders") if not processed_observations_df.is_empty() else pl.DataFrame()
                
                # Log to visualizer
                if not processed_observations_df.is_empty() or not actions_df.is_empty():
                    self.visualizer.log_dataframes(
                        poses_df=actions_df,
                        encoders_df=encoder_df,
                        camera_df=camera_df,
                        timestamp=frame * 0.1,
                        static=False,
                    )
                
                # Timing control
                elapsed = time.time() - start_time
                expected_time = frame * frame_duration
                if elapsed < expected_time:
                    time.sleep(expected_time - elapsed)
                
                frame += 1
            
            print(f"✅ Episode {episode_idx} replay complete!")
            
        except Exception as e:
            print(f"❌ Error visualizing episode {episode_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def list_episodes(self):
        """List all available episodes."""
        print(f"\n📋 Available Episodes:")
        print(f"{'Episode':<8} {'Length':<8} {'Frames':<15} {'Status'}")
        print("-" * 45)
        
        cumulative = 0
        for i, length in enumerate(self.episode_lengths):
            status = "✅ Valid" if length > 0 else "❌ Empty"
            frame_range = f"{cumulative}-{cumulative + length - 1}" if length > 0 else "N/A"
            print(f"{i:<8} {length:<8} {frame_range:<15} {status}")
            cumulative += length
    
    def close(self):
        """Clean up resources."""
        if self.keyboard:
            self.keyboard.close()
        yarp.Network.fini()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize MetaCub dataset episodes")
    parser.add_argument("dataset", nargs='?', default="assets/beer_data.zarr.zip", help="Path to zarr.zip dataset file")
    parser.add_argument("-e", "--episode", type=int, default=0, help="Episode index to visualize")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("-l", "--list", action="store_true", help="List available episodes and exit")
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"❌ Dataset file not found: {args.dataset}")
        return 1
    
    try:
        # Create visualizer
        visualizer = DatasetVisualizer(args.dataset)
        
        if args.list:
            visualizer.list_episodes()
            return 0
        
        # Validate episode
        if args.episode >= len(visualizer.episode_lengths):
            print(f"❌ Episode {args.episode} not found (max: {len(visualizer.episode_lengths) - 1})")
            return 1
        
        if visualizer.episode_lengths[args.episode] == 0:
            print(f"❌ Episode {args.episode} is empty")
            return 1
        
        print(f"🎬 Starting dataset visualization...")
        print(f"📋 Controls:")
        print(f"   'q' - Quit")
        print(f"   's' or space - Pause/Resume")
        print(f"   'r' - Restart episode")
        
        # Visualize episode
        visualizer.visualize_episode(args.episode, args.speed)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        if 'visualizer' in locals():
            visualizer.close()


if __name__ == "__main__":
    exit(main())
