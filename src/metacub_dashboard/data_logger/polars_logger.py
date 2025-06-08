"""
Pure Polars data logger.
Converts Polars DataFrames directly to log format.
"""
import polars as pl
import numpy as np
from typing import Dict, Any


class PolarsDataLogger:
    """Data logger that works directly with Polars DataFrames."""
    
    def __init__(self, base_logger):
        self.base_logger = base_logger
        self.diagnostics_history = []

    def log_dataframes(self, observations_df: pl.DataFrame, actions_df: pl.DataFrame = None):
        """Log Polars DataFrames to the data logger."""
        # Convert observations DataFrame to dictionary
        obs_dict = self._observations_df_to_dict(observations_df)
        
        # Convert actions DataFrame to dictionary
        action_dict = {}
        if actions_df is not None and len(actions_df) > 0:
            action_dict = self._actions_df_to_dict(actions_df)
        
        # Log to base logger
        self.base_logger.log(obs_dict, action_dict)
        
        # Store diagnostics
        self._store_diagnostics(observations_df, actions_df)

    def _observations_df_to_dict(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Convert observations DataFrame to flat dictionary for logging."""
        obs_dict = {}
        
        for row in df.iter_rows(named=True):
            stream_name = row['name']
            stream_type = row['stream_type']
            stream_data = row['data']
            
            if stream_type == "camera":
                # Camera data: stream_name_rgb, stream_name_depth
                for data_type, image in stream_data.items():
                    key = f"{stream_name}_{data_type}"
                    obs_dict[key] = image
                    
            elif stream_type == "encoders":
                # Encoder data: stream_name_board_name for each board
                for board_name, board_data in stream_data.items():
                    if isinstance(board_data, dict) and 'values' in board_data:
                        key = f"{stream_name}_{board_name}"
                        obs_dict[key] = board_data['values']
                    else:
                        key = f"{stream_name}_{board_name}"
                        obs_dict[key] = board_data
            else:
                # Generic handling
                for data_name, data_value in stream_data.items():
                    key = f"{stream_name}_{data_name}"
                    obs_dict[key] = np.array(data_value) if not isinstance(data_value, np.ndarray) else data_value
        
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

    def _store_diagnostics(self, observations_df: pl.DataFrame, actions_df: pl.DataFrame = None):
        """Store diagnostic information from DataFrames."""
        timestamp = 0.0
        
        # Extract timing stats from observations
        if len(observations_df) > 0:
            obs_stats = observations_df.select([
                pl.col("name"),
                pl.col("metadata").struct.field("frequency").alias("frequency"),
                pl.col("metadata").struct.field("read_delay").alias("read_delay"),
                pl.col("metadata").struct.field("read_attempts").alias("read_attempts"),
                pl.col("metadata").struct.field("timestamp").alias("timestamp"),
            ])
            timestamp = obs_stats.select("timestamp").max().item()
        
        # Extract timing stats from actions
        action_stats = pl.DataFrame()
        if actions_df is not None and len(actions_df) > 0:
            action_stats = actions_df.select([
                pl.col("name"),
                pl.col("metadata").struct.field("frequency").alias("frequency"),
                pl.col("metadata").struct.field("read_delay").alias("read_delay"),
                pl.col("metadata").struct.field("read_attempts").alias("read_attempts"),
            ])
        
        self.diagnostics_history.append({
            'timestamp': timestamp,
            'observation_stats': obs_stats.to_dicts() if len(observations_df) > 0 else [],
            'action_stats': action_stats.to_dicts() if len(action_stats) > 0 else []
        })

    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get summary statistics using Polars operations."""
        if not self.diagnostics_history:
            return {}
        
        # Convert diagnostics to DataFrame for analysis
        all_stats = []
        for entry in self.diagnostics_history:
            all_stats.extend(entry['observation_stats'])
            all_stats.extend(entry['action_stats'])
        
        if not all_stats:
            return {}
        
        stats_df = pl.DataFrame(all_stats)
        
        # Calculate summary statistics using Polars
        summary_df = stats_df.group_by("name").agg([
            pl.col("frequency").filter(pl.col("frequency") > 0).mean().alias("avg_frequency"),
            pl.col("frequency").filter(pl.col("frequency") > 0).min().alias("min_frequency"),
            pl.col("frequency").filter(pl.col("frequency") > 0).max().alias("max_frequency"),
            pl.col("read_delay").mean().alias("avg_read_delay"),
            pl.col("read_delay").max().alias("max_read_delay"),
            pl.col("read_attempts").mean().alias("avg_read_attempts"),
            pl.col("read_attempts").max().alias("max_read_attempts")
        ])
        
        # Convert to dictionary
        summary = {}
        for row in summary_df.iter_rows(named=True):
            stream_name = row['name']
            summary[stream_name] = {
                'avg_frequency': row['avg_frequency'] or 0.0,
                'min_frequency': row['min_frequency'] or 0.0,
                'max_frequency': row['max_frequency'] or 0.0,
                'avg_read_delay': row['avg_read_delay'] or 0.0,
                'max_read_delay': row['max_read_delay'] or 0.0,
                'avg_read_attempts': row['avg_read_attempts'] or 0.0,
                'max_read_attempts': row['max_read_attempts'] or 0.0
            }
        
        return summary

    def end_episode(self):
        """End episode with diagnostic summary using Polars."""
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
        
        self.base_logger.end_episode()

    def __getattr__(self, name):
        """Delegate unknown attributes to the base logger."""
        return getattr(self.base_logger, name)
