"""
Stream routing utilities for visualization.
Handles entity path assignment and other visualization-specific stream processing.
"""
from typing import List, Generator, Callable
from ...interfaces.stream_data import StreamData, StreamCollection


class StreamRouter:
    """
    Routes streams to appropriate entity paths for visualization.
    This is purely a visualization concern and doesn't modify core stream functionality.
    """
    
    def __init__(self):
        self.rules = []
    
    def add_entity_path_rule(self, condition: Callable[[StreamData], bool], entity_path: str):
        """
        Add a rule to assign entity paths to streams for visualization routing.
        
        Args:
            condition: Function that takes a StreamData and returns True if rule should apply
            entity_path: Entity path where matching streams should be logged in the visualizer
        """
        action = lambda s: setattr(s, 'entity_path', entity_path)
        self.rules.append((condition, action))
    
    def route_streams(self, streams: List[StreamData]) -> Generator[StreamData, None, None]:
        """
        Apply entity path routing rules to streams.
        
        Args:
            streams: List of StreamData objects to route
            
        Yields:
            StreamData objects with entity_path set for visualization
        """
        for stream in streams:
            # Apply all matching entity path rules
            for condition, action in self.rules:
                if condition(stream):
                    action(stream)
            
            yield stream
    
    def route_collection(self, collection: StreamCollection) -> StreamCollection:
        """
        Route a StreamCollection and return a new collection with entity paths set.
        
        Args:
            collection: StreamCollection to route
            
        Returns:
            New StreamCollection with entity paths set
        """
        routed_streams = list(self.route_streams(collection.to_list()))
        return StreamCollection(routed_streams)


def create_default_router(eef_paths: List[str]) -> StreamRouter:
    """
    Create a StreamRouter with default entity path rules for typical robot setup.
    
    Args:
        eef_paths: List of end-effector paths [left_palm_path, right_palm_path]
        
    Returns:
        StreamRouter with default rules configured
    """
    router = StreamRouter()
    
    # Rule for left arm encoders (fingers) - log under left palm path
    router.add_entity_path_rule(
        condition=lambda s: s.stream_type == "encoders" and "left_arm" in s.name,
        entity_path=f"{eef_paths[0]}/fingers"
    )
    
    # Rule for right arm encoders (fingers) - log under right palm path  
    router.add_entity_path_rule(
        condition=lambda s: s.stream_type == "encoders" and "right_arm" in s.name,
        entity_path=f"{eef_paths[1]}/fingers"
    )
    
    # Rule for other encoders (default joints path)
    router.add_entity_path_rule(
        condition=lambda s: s.stream_type == "encoders" and "left_arm" not in s.name and "right_arm" not in s.name,
        entity_path="joints"
    )
    
    return router
