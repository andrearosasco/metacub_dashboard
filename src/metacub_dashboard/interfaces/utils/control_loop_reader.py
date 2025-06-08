"""
Unified reader for control loop that manages action/observation pairs.
Handles timing and synchronization between action commands and observations.
"""
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..stream_data import StreamData, StreamInterface


@dataclass
class ControlLoopData:
    """Container for synchronized action/observation pair."""
    action_stream: StreamData
    observation_streams: List[StreamData]
    iteration: int
    loop_timestamp: float


class ControlLoopReader:
    """
    Unified reader that manages control loop timing and action/observation synchronization.
    
    The control loop works as follows:
    1. Read action (what robot should do)
    2. Sleep for control period to maintain frequency
    3. Read observations (what robot sees/feels)
    4. Return synchronized action/observation pair
    """
    
    def __init__(self, 
                 action_interface: StreamInterface,
                 observation_streams: Dict[str, StreamInterface],
                 control_frequency: float = 10.0,
                 blocking: bool = False):
        """
        Args:
            action_interface: Interface for reading action commands
            observation_streams: Dict of observation interfaces keyed by name
            control_frequency: Control loop frequency in Hz
            blocking: Whether reads should block or return None if no data
        """
        self.action_interface = action_interface
        self.observation_streams = observation_streams
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.blocking = blocking
        self.iteration = 0
        
        print(f"ControlLoopReader: {control_frequency} Hz control loop")
        print(f"Action: {action_interface.stream_name}")
        print(f"Observations: {list(observation_streams.keys())}")
    
    def read(self) -> Optional[ControlLoopData]:
        """
        Execute one control loop iteration:
        1. Read action
        2. Sleep for control period  
        3. Read observations
        4. Return synchronized pair
        """
        loop_start = time.perf_counter()
        
        # 1. Read action (what robot should do)
        try:
            action_result = self.action_interface.read()
            # Action should always be a single StreamData (not a list)
            if isinstance(action_result, list):
                # If for some reason action returns a list, take the first one
                action_stream = action_result[0] if action_result else None
            else:
                action_stream = action_result
        except Exception as e:
            print(f"Failed to read action: {e}")
            if not self.blocking:
                return None
            raise
        
        # 2. Sleep for control period to maintain frequency
        elapsed = time.perf_counter() - loop_start
        sleep_time = self.control_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # 3. Read observations (what robot sees/feels)
        observation_streams = []
        for name, interface in self.observation_streams.items():
            try:
                obs_result = interface.read()
                if obs_result is not None:
                    # Handle interfaces that return single StreamData or List[StreamData]
                    if isinstance(obs_result, list):
                        observation_streams.extend(obs_result)
                    else:
                        observation_streams.append(obs_result)
            except Exception as e:
                print(f"Failed to read {name}: {e}")
                if not self.blocking:
                    continue  # Skip this observation
                raise
        
        if not observation_streams and not self.blocking:
            return None
        
        # 4. Return synchronized action/observation pair
        loop_timestamp = time.perf_counter()
        control_data = ControlLoopData(
            action_stream=action_stream,
            observation_streams=observation_streams,
            iteration=self.iteration,
            loop_timestamp=loop_timestamp
        )
        
        self.iteration += 1
        return control_data
    
    def close(self):
        """Close all interfaces."""
        self.action_interface.close()
        for interface in self.observation_streams.values():
            interface.close()
