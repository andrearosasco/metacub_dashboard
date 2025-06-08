# 🏗️ MetaCub Dashboard - Clean Stream Architecture

## 📋 Overview

The new architecture completely separates concerns and eliminates the confusion around DataPacket usage. Here's what changed:

## 🔄 Before vs After

### ❌ **Old Design Problems:**
- Mixed data + diagnostics in `DataPacket`
- Inconsistent return types (dict vs list)
- ActionInterface returned diagnostic packet separately
- Visualizer knew about "actions" (should only know poses)
- No clear stream naming/identification

### ✅ **New Clean Design:**

```python
# Each interface returns ONE StreamData object
stream_data = interface.read()

# Clean separation:
stream_data.name          # "robot_joints", "main_camera", "target_poses"
stream_data.data          # Dict of actual data
stream_data.metadata      # All diagnostic info (frequency, delays, etc.)
stream_data.stream_type   # "encoders", "camera", "poses"
```

## 🏛️ Architecture Components

### 1. **StreamData & StreamMetadata**
```python
@dataclass
class StreamMetadata:
    timestamp: float
    seq_number: int
    read_timestamp: float
    read_delay: float
    read_attempts: int
    frequency: float | None

@dataclass  
class StreamData:
    name: str                    # Clear semantic name
    data: Dict[str, Any]         # Actual data by component
    metadata: StreamMetadata     # Diagnostic info
    stream_type: str            # "encoders", "camera", "poses"
```

### 2. **StreamInterface Base Class**
```python
class StreamInterface(ABC):
    @abstractmethod
    def read(self) -> StreamData:
        pass
    
    @abstractmethod
    def close(self):
        pass
```

### 3. **Clean Interface Implementations**

#### **EncodersInterface**
```python
# Returns robot joint data
stream_data = encoders.read()
# stream_data.name = "robot_joints"
# stream_data.data = {
#     "left_arm": {"values": [...], "labels": [...]},
#     "right_arm": {"values": [...], "labels": [...]}
# }
```

#### **CameraInterface** 
```python
# Returns camera images
stream_data = camera.read()
# stream_data.name = "main_camera"  
# stream_data.data = {
#     "rgb": np.array(...),
#     "depth": np.array(...)
# }
```

#### **ActionInterface** → **PoseInterface**
```python
# Returns target poses (renamed from "action")
stream_data = poses.read()
# stream_data.name = "target_poses"
# stream_data.data = {
#     "neck": Pose(...),
#     "left_arm": Pose(...),
#     "l_thumb": Pose(...)
# }
```

### 4. **Clean Visualizer**
```python
visualizer.log(
    poses_streams=[target_poses],      # 3D reference frames
    encoders_streams=[robot_joints],   # Robot movement
    camera_streams=[main_camera],      # Images
    timestamp=time,
    static=False
)
```

**Visualizer now only knows about:**
- **Poses:** 3D reference frames (position + orientation)
- **Encoders:** Joint values that move the robot
- **Images:** RGB/depth data
- **Diagnostics:** Stream metadata (frequencies, delays)

### 5. **Simplified Data Logging**
```python
# Old way - manual dict construction
log_data = {"camera_rgb": ..., "left_arm_values": ...}
data_logger.log(obs_data, action_data)

# New way - automatic conversion
data_logger.log_streams(
    observation_streams=[camera_stream, encoder_stream],
    poses_stream=target_poses_stream
)
```

### 6. **ObservationReader**
```python
obs_reader = ObservationReader({
    "main_camera": CameraInterface(..., stream_name="main_camera"),
    "robot_joints": EncodersInterface(..., stream_name="robot_joints") 
})

# Returns list of StreamData objects
streams = obs_reader.read()

# Helper methods
camera_stream = obs_reader.get_stream_by_name(streams, "main_camera")
encoder_streams = obs_reader.get_streams_by_type(streams, "encoders")
```

## 🎯 Key Benefits

### **1. Clear Separation of Concerns**
- **Data:** What the robot sees/feels/should do
- **Diagnostics:** How well streams are performing
- **Visualization:** Clean semantic categories

### **2. Consistent Interface**
- Every interface returns exactly ONE `StreamData`
- No more dict vs list confusion
- Clear naming with `stream_name` parameter

### **3. Semantic Clarity**
- "action" → "poses" (3D reference frames)
- "encoders" → robot joint movement
- "camera" → images for visualization
- Each stream has a clear semantic name

### **4. Easier Debugging**
```python
# Automatic diagnostic logging
stream_data = interface.read()
print(f"{stream_data.name}: {stream_data.metadata.frequency} Hz")
print(f"Read delay: {stream_data.metadata.read_delay*1000:.1f} ms")
```

### **5. Simplified Main Loop**
```python
# Clean, readable main loop
target_poses = pose_reader.read()
observations = obs_reader.read()

visualizer.log(
    poses_streams=[target_poses],
    encoders_streams=obs_reader.get_streams_by_type(observations, "encoders"),
    camera_streams=obs_reader.get_streams_by_type(observations, "camera")
)

data_logger.log_streams(observations, target_poses)
```

## 📁 File Structure

```
interfaces/
├── stream_data.py              # Core data structures
├── encoders_interface.py       # Robot joints → StreamData
├── camera_interface.py         # Images → StreamData  
├── action_interface_new.py     # Poses → StreamData
└── utils/
    └── observation_reader_new.py   # Multi-stream reader

visualizer/
└── visualizer_new.py          # Clean visualizer for StreamData

data_logger/
└── stream_logger.py           # StreamData → DataLogger conversion

main_clean.py                  # Complete example
```

## 🚀 Migration Path

1. **Phase 1:** New interfaces work alongside old ones
2. **Phase 2:** Update main.py to use new architecture  
3. **Phase 3:** Remove old DataPacket-based code
4. **Phase 4:** Full cleanup and optimization

The new architecture is **much cleaner**, **easier to debug**, and **more maintainable**! 🎉
