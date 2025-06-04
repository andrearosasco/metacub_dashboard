from concurrent.futures import wait
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import copy
import multiprocessing
import os
from pathlib import Path
import shutil
import time
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
        self.data = None

        if Path(path).exists() and not exist_ok:
            raise FileExistsError(f"Destination path '{path}' already exists")
        if Path(path).exists():
            # Open an existing Zarr store
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)

        # self.num_workers = 1 # multiprocessing.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.futures = []

        # Create a destination Zarr store
        self.path = path
        self.dest_store = zarr.storage.ZipStore(path, mode='w')
        self.dest_group = zarr.group(store=self.dest_store)

    def log(self, obs, action):
        step_data = {'action': action, 'obs': obs}
        self.data = dict_concat(step_data, self.data)

        if self.log_count % 100 == 99:
            print(self.log_count, flush=True)

        if self.log_count % self.flush_every == self.flush_every - 1:
            # if self.process and self.process.is_alive(): raise ValueError("Data writing took too much time")
            # self.process = multiprocessing.Process(target=write_data, args=(copy.deepcopy(self.data), self.path))
            # self.process.start()
            self.futures.append(self.executor.submit(write_data, copy.deepcopy(self.data), self.path))
            # write_data(copy.deepcopy(self.data), self.path)
            self.data = None

        self.log_count += 1

    def end_episode(self):
        completed, futures = wait(self.futures)
        for f in completed:
            if not f.result():
                raise ValueError("Error while copying data")
        self.futures = []

        dest_store = zarr.ZipStore(self.path)
        dest_group = zarr.group(store=dest_store, overwrite=False)

        if 'episode_length' not in dest_group:
            za = dest_group.require_dataset('episode_length', shape=(1,), dtype=int, chunks=(1,), compression=Blosc(), overwrite=False)
            za[0] = self.log_count
        else:
            za = dest_group['episode_length']
            za.append([self.log_count])
        self.log_count = 0

        dest_store.close()
        

def write_data(data, path):
    dest_store = zarr.ZipStore(path)
    dest_group = zarr.group(store=dest_store, overwrite=False)

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
