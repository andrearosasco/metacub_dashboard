"""
Data logger using native Polars DataFrames (now the default).
Converts Polars DataFrames directly to log format.
"""
import polars as pl
import numpy as np
import time
from typing import Dict, Any


class DataLogger:
    """Data logger that works directly with Polars DataFrames (now the default)."""
    
    def __init__(self, base_logger):
        self.base_logger = base_logger
        self.diagnostics_history = []

    def log_dataframes(self, observations_df: pl.DataFrame, actions_df: pl.DataFrame = None):
        """Log Polars DataFrames to the data logger - zero main-thread overhead."""
        # Store DataFrames directly, defer conversion to background thread
        self.base_logger.log_dataframes_raw(observations_df, actions_df)
        
        # Skip diagnostics processing entirely during main loop for maximum performance
        # Diagnostics will be computed only at end_episode() from logged data

    def _observations_df_to_dict(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Convert observations DataFrame to flat dictionary for logging."""
        obs_dict = {}
        
        for row in df.iter_rows(named=True):
            stream_name = row['name']
            stream_type = row['stream_type']
            stream_data = row['data']
            
            if stream_type == "camera":
                # Camera data: use stream_name directly (already includes data type)
                for data_type, image in stream_data.items():
                    # stream_name is already like "agentview_rgb", so use it directly
                    key = stream_name
                    obs_dict[key] = image
                    
            elif stream_type == "encoders":
                # Encoder data: use stream_name directly (already includes board name)
                for board_name, board_data in stream_data.items():
                    if isinstance(board_data, dict) and 'values' in board_data:
                        # stream_name is already like "encoders_head", so use it directly
                        key = stream_name
                        obs_dict[key] = board_data['values']
                    else:
                        key = stream_name
                        obs_dict[key] = board_data
            else:
                # Generic handling
                key = stream_name
                obs_dict[key] = stream_data
        
        return obs_dict

    def _actions_df_to_dict(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Convert actions DataFrame to dictionary for logging."""
        action_dict = {}
        
        for row in df.iter_rows(named=True):
            stream_data = row['data']
            
            for pose_name, pose_data in stream_data.items():
                if isinstance(pose_data, np.ndarray):
                    action_dict[pose_name] = pose_data
                else:
                    action_dict[pose_name] = np.array(pose_data)
        
        return action_dict

    def __getattr__(self, name):
        """Delegate unknown attributes to the base logger."""
        return getattr(self.base_logger, name)
