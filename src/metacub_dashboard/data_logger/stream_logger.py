"""
Helper utilities for logging StreamData to DataLogger.
Provides clean conversion from StreamData objects to log-friendly formats.
"""
import numpy as np
from typing import Dict, List, Any
from ..interfaces.stream_data import StreamData, Pose


def stream_data_to_obs_dict(stream_data_list: List[StreamData]) -> Dict[str, Any]:
    """
    Convert a list of StreamData objects to a flat observation dictionary 
    suitable for DataLogger.
    
    Args:
        stream_data_list: List of StreamData objects from observation reader
        
    Returns:
        Dictionary with flattened observation data
    """
    obs_dict = {}
    
    for stream_data in stream_data_list:
        stream_name = stream_data.name
        
        if stream_data.stream_type == "camera":
            # Camera data: stream_name_rgb, stream_name_depth
            for data_type, image in stream_data.data.items():
                key = f"{stream_name}_{data_type}"
                obs_dict[key] = image
                
        elif stream_data.stream_type == "encoders":
            # Encoder data: stream_name_board_name for each board
            for board_name, board_data in stream_data.data.items():
                if isinstance(board_data, dict) and 'values' in board_data:
                    key = f"{stream_name}_{board_name}"
                    obs_dict[key] = board_data['values']
                else:
                    # Fallback for other formats
                    key = f"{stream_name}_{board_name}"
                    obs_dict[key] = board_data
                    
        else:
            # Generic handling for other stream types
            for data_name, data_value in stream_data.data.items():
                key = f"{stream_name}_{data_name}"
                if isinstance(data_value, Pose):
                    obs_dict[key] = data_value.to_numpy()
                elif isinstance(data_value, np.ndarray):
                    obs_dict[key] = data_value
                else:
                    obs_dict[key] = np.array(data_value)
    
    return obs_dict


def stream_data_to_action_dict(poses_stream: StreamData) -> Dict[str, np.ndarray]:
    """
    Convert a poses StreamData object to an action dictionary.
    
    Args:
        poses_stream: StreamData with stream_type="poses"
        
    Returns:
        Dictionary mapping pose names to numpy arrays
    """
    action_dict = {}
    
    for pose_name, pose in poses_stream.data.items():
        if isinstance(pose, Pose):
            action_dict[pose_name] = pose.to_numpy()
        elif isinstance(pose, np.ndarray):
            action_dict[pose_name] = pose
        else:
            action_dict[pose_name] = np.array(pose)
    
    return action_dict


def extract_diagnostics(stream_data_list: List[StreamData]) -> Dict[str, Dict[str, float]]:
    """
    Extract diagnostic information from StreamData objects.
    
    Args:
        stream_data_list: List of StreamData objects
        
    Returns:
        Nested dictionary with diagnostic info per stream
    """
    diagnostics = {}
    
    for stream_data in stream_data_list:
        stream_name = stream_data.name
        metadata = stream_data.metadata
        
        diagnostics[stream_name] = {
            'frequency': metadata.frequency or 0.0,
            'timestamp': metadata.timestamp,
            'read_timestamp': metadata.read_timestamp,
            'read_delay': metadata.read_delay,
            'read_attempts': metadata.read_attempts,
            'seq_number': metadata.seq_number
        }
    
    return diagnostics


class StreamDataLogger:
    """
    Wrapper around DataLogger that provides convenient methods for logging StreamData.
    """
    
    def __init__(self, data_logger):
        """
        Args:
            data_logger: Instance of DataLogger
        """
        self.data_logger = data_logger
        self.diagnostics_history = []
    
    def log_streams(self, observation_streams: List[StreamData], 
                   poses_stream: StreamData = None):
        """
        Log StreamData objects to the data logger.
        
        Args:
            observation_streams: List of observation StreamData objects
            poses_stream: Optional poses StreamData object
        """
        # Convert to observation dictionary
        obs_dict = stream_data_to_obs_dict(observation_streams)
        
        # Convert poses to action dictionary
        action_dict = {}
        if poses_stream:
            action_dict = stream_data_to_action_dict(poses_stream)
        
        # Log to data logger
        self.data_logger.log(obs_dict, action_dict)
        
        # Store diagnostics for later analysis
        all_streams = observation_streams + ([poses_stream] if poses_stream else [])
        diagnostics = extract_diagnostics(all_streams)
        self.diagnostics_history.append({
            'timestamp': obs_dict.get('timestamp', 0.0),
            'streams': diagnostics
        })
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stream diagnostics."""
        if not self.diagnostics_history:
            return {}
        
        # Collect all stream names
        all_stream_names = set()
        for entry in self.diagnostics_history:
            all_stream_names.update(entry['streams'].keys())
        
        summary = {}
        for stream_name in all_stream_names:
            frequencies = []
            delays = []
            attempts = []
            
            for entry in self.diagnostics_history:
                if stream_name in entry['streams']:
                    stream_diag = entry['streams'][stream_name]
                    if stream_diag['frequency'] > 0:
                        frequencies.append(stream_diag['frequency'])
                    delays.append(stream_diag['read_delay'])
                    attempts.append(stream_diag['read_attempts'])
            
            summary[stream_name] = {
                'avg_frequency': np.mean(frequencies) if frequencies else 0.0,
                'min_frequency': np.min(frequencies) if frequencies else 0.0,
                'max_frequency': np.max(frequencies) if frequencies else 0.0,
                'avg_read_delay': np.mean(delays),
                'max_read_delay': np.max(delays),
                'avg_read_attempts': np.mean(attempts),
                'max_read_attempts': np.max(attempts)
            }
        
        return summary
    
    def end_episode(self):
        """End the current episode and print diagnostic summary."""
        summary = self.get_diagnostic_summary()
        
        print("📈 Stream Performance Summary:")
        for stream_name, stats in summary.items():
            print(f"  {stream_name}:")
            print(f"    Frequency: {stats['avg_frequency']:.1f} Hz "
                  f"(range: {stats['min_frequency']:.1f}-{stats['max_frequency']:.1f})")
            print(f"    Read delay: {stats['avg_read_delay']*1000:.1f} ms "
                  f"(max: {stats['max_read_delay']*1000:.1f} ms)")
            print(f"    Read attempts: {stats['avg_read_attempts']:.1f} "
                  f"(max: {stats['max_read_attempts']:.0f})")
        
        self.data_logger.end_episode()
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying data logger."""
        return getattr(self.data_logger, name)
