"""
Control loop reader using native Polars DataFrames (now the default).
"""
import time
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

    def read(self) -> Optional[ControlLoopData]:
        """Read synchronized action/observation data as Polars DataFrames."""
        current_time = time.perf_counter()
        
        # Wait for next scheduled read time
        if current_time < self.next_read_time:
            if self.blocking:
                time.sleep(self.next_read_time - current_time)
            else:
                return None
        
        self.next_read_time += self.period
        
        # Read action data
        actions_df = self.action_interface.read()
        actions_df = pl.DataFrame(schema=actions_df.schema if 'actions_df' in locals() else None)
    
        # Read observation data from all interfaces
        observation_dfs = []
        for name, interface in self.observation_interfaces.items():

            obs_df = interface.read()
            if len(obs_df) > 0:
                observation_dfs.append(obs_df)

        
        # Combine all observation DataFrames
        if observation_dfs:
            observations_df = pl.concat(observation_dfs, how="vertical")
        else:
            observations_df = pl.DataFrame()
            if self.blocking:
                return None
        
        # Return synchronized data
        loop_timestamp = time.perf_counter()
        control_data = ControlLoopData(
            actions_df=actions_df,
            observations_df=observations_df,
            iteration=self.iteration,
            loop_timestamp=loop_timestamp
        )
        
        self.iteration += 1
        return control_data

    def close(self):
        """Close all interfaces."""
        self.action_interface.close()
        for interface in self.observation_interfaces.values():
            interface.close()
