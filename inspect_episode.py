#!/usr/bin/env python3
"""
Simple test script to inspect zarr.zip episode data without full visualization setup.
"""
import zarr
from zarr.storage import ZipStore
import zipfile
import argparse
from pathlib import Path


def inspect_zarr_structure(zarr_path: str):
    """Inspect the structure of a zarr.zip file."""
    print(f"📂 Inspecting: {zarr_path}")
    
    try:
        # Open zarr store from zip file
        store = ZipStore(zarr_path, mode='r')
        root = zarr.group(store=store)
        
        print(f"📊 Root groups: {list(root.keys())}")
        
        for group_name in root.keys():
            group = root[group_name]
            print(f"\n📁 Group: {group_name}")
            
            if hasattr(group, 'keys'):
                print(f"   Arrays: {list(group.keys())}")
                
                for array_name in group.keys():
                    try:
                        array = group[array_name]
                        print(f"   📄 {array_name}: shape={array.shape}, dtype={array.dtype}")
                        
                        # Show a sample of the data
                        if len(array) > 0:
                            sample = array[0]
                            print(f"      Sample: {sample}")
                            
                    except Exception as e:
                        print(f"   ❌ Error reading {array_name}: {e}")
            else:
                print(f"   Direct array: shape={group.shape}, dtype={group.dtype}")
                
    except Exception as e:
        print(f"❌ Error inspecting zarr file: {e}")
        
        # Try alternative method - inspect zip contents directly
        print("\n🔍 Trying to inspect zip contents directly...")
        try:
            with zipfile.ZipFile(zarr_path, 'r') as zf:
                files = zf.namelist()
                print(f"📋 Zip contents ({len(files)} files):")
                for f in sorted(files)[:20]:  # Show first 20 files
                    print(f"   {f}")
                if len(files) > 20:
                    print(f"   ... and {len(files) - 20} more files")
        except Exception as e2:
            print(f"❌ Error reading zip: {e2}")
            import traceback
            traceback.print_exc()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Inspect MetaCub Dashboard zarr.zip episode files")
    parser.add_argument("episode_path", help="Path to the episode zarr.zip file")
    
    args = parser.parse_args()
    
    episode_path = Path(args.episode_path)
    
    if not episode_path.exists():
        print(f"❌ Episode file not found: {episode_path}")
        return
    
    inspect_zarr_structure(str(episode_path))


if __name__ == "__main__":
    main()
