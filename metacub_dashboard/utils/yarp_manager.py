"""
YARP Network manager for handling initialization and cleanup.
"""
import yarp
from typing import Optional


class YARPManager:
    """Context manager for YARP Network initialization and cleanup."""
    
    def __init__(self):
        self.is_initialized = False
    
    def __enter__(self):
        """Initialize YARP Network."""
        print("📡 Initializing YARP Network...")
        yarp.Network.init()
        self.is_initialized = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup YARP Network."""
        if self.is_initialized:
            print("📡 Finalizing YARP Network...")
            yarp.Network.fini()
            self.is_initialized = False
    
    def close(self):
        """Explicit cleanup method."""
        if self.is_initialized:
            yarp.Network.fini()
            self.is_initialized = False
