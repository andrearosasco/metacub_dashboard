from concurrent.futures import wait
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy
import multiprocessing
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
import zarr.storage
from .utils.depth_compression import depth2rgb
import zarr

import numcodecs
from numcodecs import Blosc
from .utils.imagecodecs_numcodecs import Jpeg2k, Qoi
numcodecs.register_codec(Jpeg2k)
numcodecs.register_codec(Qoi)

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Duplicate name:*')



def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dict_concat(d, acc=None, axis=0):
    """
    Recursively concatenate numpy array values from dictionary `d` into `acc`,
    adding a new first dimension to each array.
    Raises an exception if a leaf node is not a numpy array.
    """
    if isinstance(d, dict):
        if acc is None:
            acc = {}
        for k in d:
            acc[k] = dict_concat(d[k], acc.get(k), axis=axis)
        return acc
    else:
        if not isinstance(d, np.ndarray):
            raise ValueError("Value is not a numpy array")
        d_expanded = np.expand_dims(d, axis=axis)
        if acc is None:
            return d_expanded
        else:
            return np.concatenate([acc, d_expanded], axis=axis)

def copy_data(za, data, start_idx, end_idx, current_size):
    """
    Copies a slice of data to the Zarr array and prints process and CPU information.

    Parameters:
    - za: Zarr dataset object.
    - data: Source NumPy array.
    - start_idx: Starting index for the slice.
    - end_idx: Ending index for the slice.
    - key: Dataset key/name.
    """
    while True:
        try:
            pid = os.getpid()
            za[start_idx:end_idx] = data[start_idx - current_size:end_idx - current_size]
            _ = za[start_idx:end_idx]
            return True
        except Exception as e:
            continue

def process_image_data(image):
    return image

def process_depth_data(depth):
    depth_min, depth_max = depth.min(), depth.max()
    if depth.shape[-1] == 1:
        depth = depth.squeeze()
    original_shape = depth.shape
    depth = depth.reshape((-1,) + (depth.shape[-2:]))
    colored_depth = depth2rgb(depth, depth_min, depth_max)
    colored_depth = colored_depth.reshape(original_shape[:-2] + colored_depth.shape[-3:])
    return colored_depth, depth_min, depth_max


class DataLogger:
    def __init__(self, path, flush_every=200, exist_ok=False):
        self.flush_every = flush_every
        self.log_count = 0
        self.data_buffer = []  # Use list buffer instead of dict_concat

        if Path(path).exists() and not exist_ok:
            raise FileExistsError(f"Destination path '{path}' already exists")

        # self.num_workers = 1 # multiprocessing.cpu_count()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = []

        # Create or open a destination Zarr store
        self.path = path
        Path(path).parent.mkdir(exist_ok=True)
        
        if Path(path).exists() and exist_ok:
            # Open existing store in append mode
            self.dest_store = zarr.storage.DirectoryStore(path, mode='a')
            self.dest_group = zarr.group(store=self.dest_store, overwrite=False)
        else:
            # Create new store
            self.dest_store = zarr.storage.DirectoryStore(path, mode='w')
            self.dest_group = zarr.group(store=self.dest_store)

    def log_dataframes_raw(self, observations_df, actions_df=None):
        """Log raw DataFrames, defer conversion to background thread."""
        # Store DataFrames as lightweight reference, avoid expensive conversion on main thread
        step_data = {'observations_df': observations_df, 'actions_df': actions_df}
        self.data_buffer.append(step_data)

        if self.log_count % self.flush_every == self.flush_every - 1:
            # Move entire buffer processing to background thread
            self.futures.append(self.executor.submit(process_and_write_dataframes_buffer, self.data_buffer, self.path))
            self.data_buffer = []

        self.log_count += 1

    def log(self, obs, action):
        # Store data as lightweight reference, avoid expensive dict_concat on main thread
        step_data = {'action': action, 'obs': obs}
        self.data_buffer.append(step_data)

        if self.log_count % self.flush_every == self.flush_every - 1:
            # Move entire buffer processing to background thread
            self.futures.append(self.executor.submit(process_and_write_buffer, self.data_buffer, self.path))
            self.data_buffer = []

        self.log_count += 1

    def get_episode_count(self) -> int:
        if not Path(self.path).exists():
            return 0
            
        try:
            # Use the existing dest_group instead of opening a new connection
            if 'episode_length' in self.dest_group:
                return len(self.dest_group['episode_length'])
            else:
                return 0
                
        except (zipfile.BadZipFile, OSError, ValueError, AttributeError):
            return 0

    def end_episode(self, success=True):
        # First, flush any remaining data in the buffer
        if self.data_buffer:
            self.futures.append(self.executor.submit(process_and_write_dataframes_buffer, self.data_buffer, self.path))
            self.data_buffer = []

        # Wait for all futures to complete
        if self.futures:
            completed, _ = wait(self.futures)
            for f in completed:
                if not f.result():
                    raise ValueError("Error while copying data")
            self.futures = []

        # Only try to access the zip file if we've actually written data
        try:
            if Path(self.path).exists():
                dest_store = zarr.storage.DirectoryStore(self.path, mode='a')  # append mode
            else:
                dest_store = zarr.storage.DirectoryStore(self.path, mode='w')  # write mode
            dest_group = zarr.group(store=dest_store, overwrite=False)

            # Store episode length
            if 'episode_length' not in dest_group:
                za_length = dest_group.require_dataset('episode_length', shape=(1,), dtype=int, chunks=(1,), compression=Blosc(), overwrite=False)
                za_length[0] = self.log_count
            else:
                za_length = dest_group['episode_length']
                za_length.append([self.log_count])
            
            # Store episode success (True for successful completion)
            if 'success' not in dest_group:
                za_success = dest_group.require_dataset('success', shape=(1,), dtype=bool, chunks=(1,), compression=Blosc(), overwrite=False)
                za_success[0] = True
            else:
                za_success = dest_group['success']
                za_success.append([success])
            
            dest_store.close()
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            print(f"Warning: Could not access {self.path} as a zip file. Resetting data logger.")
            # Create new empty zip store
            Path(self.path).parent.mkdir(exist_ok=True)
            if Path(self.path).exists():
                os.remove(self.path)
            self.dest_store = zarr.storage.DirectoryStore(self.path, mode='w')
            self.dest_group = zarr.group(store=self.dest_store)
        
        self.log_count = 0

    def discard_episode(self):
        """Discard current episode data without saving to disk."""
        # Store buffer length before clearing
        buffer_length = len(self.data_buffer)
        
        # Cancel any pending futures to avoid writing data
        if self.futures:
            for future in self.futures:
                future.cancel()
            self.futures = []
        
        # Clear the data buffer
        self.data_buffer = []
        
        # Reset log count
        self.log_count = 0
        
        print(f"Episode data discarded ({buffer_length} buffered steps cleared)")


def process_and_write_dataframes_buffer(data_buffer, path):
    """Process buffer of DataFrame step data and write to Zarr file."""
    import polars as pl
    import numpy as np
    
    # Convert DataFrames to dict format in background thread
    accumulated_data = None
    for step_data in data_buffer:
        observations_df = step_data['observations_df']
        actions_df = step_data['actions_df']
        
        # Convert observations DataFrame to dictionary
        obs_dict = {}
        if observations_df is not None and len(observations_df) > 0:
            for row in observations_df.iter_rows(named=True):
                stream_name = row['name']
                stream_type = row['stream_type']
                stream_data = row['data']
                
                if stream_type == "camera":
                    # Camera data: stream_name_rgb, stream_name_depth
                    for data_type, image in stream_data.items():
                        key = f"{stream_name}"
                        obs_dict[key] = image
                        
                elif stream_type == "encoders":
                    # Encoder data: stream_name_board_name for each board
                    for board_name, board_data in stream_data.items():
                        if isinstance(board_data, dict) and 'values' in board_data:
                            key = f"{stream_name}"
                            obs_dict[key] = board_data['values']
                        else:
                            key = f"{stream_name}"
                            obs_dict[key] = board_data
                else:
                    # Generic handling
                    for data_name, data_value in stream_data.items():
                        key = f"{stream_name}"
                        obs_dict[key] = np.array(data_value) if not isinstance(data_value, np.ndarray) else data_value

        # Convert actions DataFrame to dictionary
        action_dict = {}
        if actions_df is not None and len(actions_df) > 0:
            for row in actions_df.iter_rows(named=True):
                stream_data = row['data']
                
                for pose_name, pose_data in stream_data.items():
                    if isinstance(pose_data, np.ndarray):
                        action_dict[pose_name] = pose_data
                    else:
                        action_dict[pose_name] = np.array(pose_data)
        
        # Accumulate using dict_concat
        step_dict = {'action': action_dict, 'obs': obs_dict}
        accumulated_data = dict_concat(step_dict, accumulated_data)
    
    # Write the accumulated data
    return write_data(accumulated_data, path)


def process_and_write_buffer(data_buffer, path):
    """Process buffer of step data and write to Zarr file."""
    # Process buffer into concatenated data structure
    accumulated_data = None
    for step_data in data_buffer:
        accumulated_data = dict_concat(step_data, accumulated_data)
    
    # Write the accumulated data
    return write_data(accumulated_data, path)


def write_data_with_copy(data, path):
    """Write data function that handles deepcopy in background thread."""
    return write_data(copy.deepcopy(data), path)


def write_data(data, path):
    # Ensure the parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Try to open existing store, create new one if it doesn't exist or is corrupted
    try:
        if Path(path).exists():
            dest_store = zarr.storage.DirectoryStore(path, mode='a')  # append mode
        else:
            dest_store = zarr.storage.DirectoryStore(path, mode='w')  # write mode
        dest_group = zarr.group(store=dest_store, overwrite=False)
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        # If the file is corrupted or has issues, remove it and create a new one
        if Path(path).exists():
            os.remove(path)
        dest_store = zarr.storage.DirectoryStore(path, mode='w')
        dest_group = zarr.group(store=dest_store)

    num_workers = 1
    executor = ThreadPoolExecutor(max_workers=num_workers)
    data = flatten_dict(data)

    futures = set()
    for key in data:
        
        source_array = data[key]

        if key.endswith('image'):
            source_array = process_image_data(source_array)
            chunk_shape = (1,) + source_array.shape[1:]
            compressor = Jpeg2k(level=50)
            attrs = {}
        elif key.endswith('depth'):
            source_array, depth_min, depth_max = process_depth_data(source_array)
            chunk_shape = (1,) + source_array.shape[1:]
            compressor = Qoi()
            attrs = {'d_min': depth_min, 'd_max': depth_max}
        else:
            chunk_shape = (1,) + source_array.shape[1:]
            compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
            attrs = {}

        if key not in dest_group:
            za = dest_group.require_dataset(
                    key, shape=source_array.shape,
                    dtype=source_array.dtype,
                    chunks=chunk_shape,
                    compression=compressor,
                    overwrite=False)
            
            for k in attrs:
                za.attrs[k] = attrs[k]
            za.attrs['_ARRAY_DIMENSIONS'] = [f'dim_{i}' for i, dim in enumerate(source_array.shape)]

        else:
            za = dest_group[key]
            za.resize((za.shape[0] + source_array.shape[0], *za.shape[1:]))
        current_size = za.nchunks_initialized
        
        split_indices = np.linspace(current_size, current_size + source_array.shape[0], num_workers + 1, dtype=int)
        for i in range(num_workers):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            futures.add(executor.submit(copy_data, za, source_array, start_idx, end_idx, current_size))
        

    completed, futures = wait(futures)
    for f in completed:
        if not f.result():
            raise ValueError("Error while copying data")
    dest_store.close()
    return True
    

class env:
    def step(action):
        return {
            'image': np.round(np.random.rand(64, 64, 3) * 255).astype(np.uint8),
            'depth': np.random.rand(64, 64),
        }
    
    def reset():
        return {
            'image': np.round(np.random.rand(64, 64, 3) * 255).astype(np.uint8),
            'depth': np.random.rand(64, 64),
        }
class oculus:
    def read():
        return {
            'right_hand': np.random.rand(7),
            'left_hand': np.random.rand(7),
        }

    
# data_logger = DataLogger('test.zarr.zip', exist_ok=True)
# for i in range(2):
#     obs = env.reset()
#     for _ in range(500):
#         action = oculus.read()
#         data_logger.log(obs, action)
#         obs = env.step(action)
#         time.sleep(0.01)

#     data_logger.end_episode()
# zarr_group = zarr.open('test.zarr.zip', mode='r')

# source_keys = []
# def collect_datasets(name, obj):
#     if isinstance(obj, zarr.core.Array):
#         source_keys.append(name)
# zarr_group.visititems(collect_datasets)

# for key in source_keys:
#     print(key)
#     _ = zarr_group[key][:]
