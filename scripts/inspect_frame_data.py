#!/usr/bin/env python3
"""
Detailed        # Examine observation data
        print(f"\n🔍 Observation structure:")
        print(f"   Type: {type(obs_group)}")
        if hasattr(obs_group, 'keys'):
            obs_keys = list(obs_group.keys())
            print(f"   Keys: {obs_keys[:10]}{'...' if len(obs_keys) > 10 else ''}")
            
            # Examine first few observation streams with sample data
            for key in obs_keys[:3]:  # Show first 3
                obs_array = obs_group[key]
                print(f"   📊 {key}:")
                print(f"      Shape: {obs_array.shape}, dtype: {obs_array.dtype}")
                if frame_idx < len(obs_array):
                    sample_data = obs_array[frame_idx]
                    print(f"      Frame {frame_idx}: {type(sample_data)} {getattr(sample_data, 'shape', getattr(sample_data, '__len__', 'scalar'))}")
                    if hasattr(sample_data, 'dtype'):
                        print(f"         dtype: {sample_data.dtype}")
                    
                    # Show actual values for smaller arrays
                    if hasattr(sample_data, 'shape') and np.prod(sample_data.shape) <= 20:
                        print(f"         values: {sample_data}")
                    elif hasattr(sample_data, '__len__') and len(sample_data) <= 10:
                        print(f"         values: {sample_data}")
        
        # Examine action dataExamines the actual structure of observations and actions in a zarr dataset.
"""
import zarr
import numpy as np
import pickle
from pathlib import Path

def inspect_frame_data(zarr_path: str, frame_idx: int = 0):
    """Inspect the structure of a specific frame."""
    print(f"📂 Inspecting frame {frame_idx} from: {zarr_path}")
    
    if not Path(zarr_path).exists():
        print(f"❌ File not found: {zarr_path}")
        return
    
    try:
        # Open zarr file
        root = zarr.open(zarr_path, mode='r')
        
        # Check if frame exists
        obs_array = root['obs']
        action_array = root['action'] if 'action' in root else None
        
        if frame_idx >= len(obs_array):
            print(f"❌ Frame {frame_idx} not found (max: {len(obs_array) - 1})")
            return
        
        print(f"✅ Found frame {frame_idx}")
        
        # Check if obs and action are groups or arrays
        obs_group = root['obs']
        action_group = root['action'] if 'action' in root else None
        
        print(f"\n🔍 Observation structure:")
        print(f"   Type: {type(obs_group)}")
        if hasattr(obs_group, 'keys'):
            obs_keys = list(obs_group.keys())
            print(f"   Keys: {obs_keys}")
            
            # Examine first few observation streams
            for key in obs_keys[:5]:  # Show first 5
                obs_array = obs_group[key]
                print(f"   📊 {key}:")
                print(f"      Shape: {obs_array.shape}, dtype: {obs_array.dtype}")
                if frame_idx < len(obs_array):
                    sample_data = obs_array[frame_idx]
                    print(f"      Frame {frame_idx}: {type(sample_data)} {getattr(sample_data, 'shape', getattr(sample_data, '__len__', 'scalar'))}")
                    if hasattr(sample_data, 'dtype'):
                        print(f"         dtype: {sample_data.dtype}")
        
        # Examine action data
        if action_group is not None:
            print(f"\n🔍 Action structure:")
            print(f"   Type: {type(action_group)}")
            if hasattr(action_group, 'keys'):
                action_keys = list(action_group.keys())
                print(f"   Keys: {action_keys}")
                
                # Examine first few action streams
                for key in action_keys[:5]:  # Show first 5
                    action_array = action_group[key]
                    print(f"   📊 {key}:")
                    print(f"      Shape: {action_array.shape}, dtype: {action_array.dtype}")
                    if frame_idx < len(action_array):
                        sample_data = action_array[frame_idx]
                        print(f"      Frame {frame_idx}: {type(sample_data)} {getattr(sample_data, 'shape', getattr(sample_data, '__len__', 'scalar'))}")
                        if hasattr(sample_data, 'dtype'):
                            print(f"         dtype: {sample_data.dtype}")
        else:
            print(f"\n🔍 No action data in dataset")
        
    except Exception as e:
        print(f"❌ Error inspecting frame data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_frame_data("assets/beer_data.zarr.zip", frame_idx=0)
    print("\n" + "="*50)
    inspect_frame_data("assets/beer_data.zarr.zip", frame_idx=1)
