"""
DataFrame processing utilities for MetaCub Dashboard.
Handles complex data transformations and entity path assignments.
"""
import polars as pl
from typing import List


class DataFrameProcessor:
    """Handles complex DataFrame transformations for visualization and logging."""
    
    def __init__(self, eef_paths: List[str]):
        """
        Initialize the processor with end-effector paths.
        
        Args:
            eef_paths: List of end-effector paths [left_hand, right_hand]
        """
        self.eef_paths = eef_paths
    
    def process_observations(self, observation_df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply entity paths to observation DataFrame using Polars when/then/otherwise.
        
        Args:
            observation_df: Raw observation DataFrame
            
        Returns:
            Processed DataFrame with entity_path column populated
        """
        return observation_df.with_columns([
            pl.when(pl.col("name").str.contains("left_arm"))
            .then(pl.lit(f"{self.eef_paths[0]}/fingers"))
            .when(pl.col("name").str.contains("right_arm"))  
            .then(pl.lit(f"{self.eef_paths[1]}/fingers"))
            .otherwise(pl.lit("joints"))
            .alias("entity_path")
        ])
    
    def split_by_stream_type(self, df: pl.DataFrame) -> dict:
        """
        Split DataFrame by stream type for targeted processing.
        
        Args:
            df: Input DataFrame with stream_type column
            
        Returns:
            Dictionary with stream types as keys and filtered DataFrames as values
        """
        return {
            "camera": df.filter(pl.col("stream_type") == "camera"),
            "encoders": df.filter(pl.col("stream_type") == "encoders"),
            "poses": df.filter(pl.col("stream_type") == "poses"),
        }
