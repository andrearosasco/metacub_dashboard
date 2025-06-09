"""
Control loop reader using native Polars DataFrames (now the default).
"""
import time
import numpy as np
import polars as pl
from typing import Dict, Optional
from dataclasses import dataclass
from ..interfaces import Interface


@dataclass
class ControlLoopData:
    """Container for control loop data using pure Polars DataFrames."""
    actions_df: pl.DataFrame
    observations_df: pl.DataFrame
    iteration: int
    loop_timestamp: float


class ControlLoopReader:
    """Control loop reader that manages timing and synchronization using native Polars DataFrames."""
    
    def __init__(self, action_interface: Interface, 
                 observation_interfaces: Dict[str, Interface],
                 control_frequency: float = 10.0, blocking: bool = False):
        self.action_interface = action_interface
        self.observation_interfaces = observation_interfaces
        self.control_frequency = control_frequency
        self.blocking = blocking
        self.period = 1.0 / control_frequency
        self.iteration = 0
        self.next_read_time = time.perf_counter() + self.period
        self.prev_observations_df = None  # Store previous observation for next action

    def reset(self):
        """Initialize the control loop by reading the first observation."""
        # Read initial observation data from all interfaces
        observation_dfs = []
        for name, interface in self.observation_interfaces.items():
            obs_df = interface.read()
            if len(obs_df) > 0:
                observation_dfs.append(obs_df)
        
        # Store initial observation for pairing with first action
        if observation_dfs:
            self.prev_observations_df = pl.concat(observation_dfs, how="vertical")
        else:
            self.prev_observations_df = pl.DataFrame()
        
        self.iteration = 0
        self.next_read_time = time.perf_counter() + self.period

    def read(self) -> Optional[ControlLoopData]:
        """Read synchronized action/observation data as Polars DataFrames.
        
        Actions are read first, then observations are read for the NEXT iteration.
        The observation returned is from the PREVIOUS iteration.
        """
        # Read action data FIRST
        actions_df = self.action_interface.read()
        
        # Use the observation from PREVIOUS iteration (or reset)
        observations_df = self.prev_observations_df if self.prev_observations_df is not None else pl.DataFrame()
        
        # Now read observation data for the NEXT iteration
        observation_dfs = []
        for name, interface in self.observation_interfaces.items():
            obs_df = interface.read()
            if len(obs_df) > 0:
                observation_dfs.append(obs_df)
        
        # Store observations for next iteration
        if observation_dfs:
            self.prev_observations_df = pl.concat(observation_dfs, how="vertical")
        else:
            self.prev_observations_df = pl.DataFrame()
            if self.blocking and self.iteration == 0:  # Only return None on first iteration if blocking
                return None
        
        # Return synchronized data (current action + previous observation)
        loop_timestamp = time.perf_counter()
        control_data = ControlLoopData(
            actions_df=actions_df,
            observations_df=observations_df,
            iteration=self.iteration,
            loop_timestamp=np.datetime64('now', 'ns')
        )
        
        self.iteration += 1
        return control_data

    def close(self):
        """Close all interfaces."""
        self.action_interface.close()
        for interface in self.observation_interfaces.values():
            interface.close()
