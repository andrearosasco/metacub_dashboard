"""
Pure Polars implementation of MetaCub Dashboard.
Uses native Polars DataFrames throughout with no wrapper classes.
"""
import time
import os
from uuid import uuid4
import numpy as np
from typing import List, Dict, Any

# Set environment before importing yarp
os.environ["YARP_ROBOT_NAME"] = "ergoCubSN002"
import yarp
from scipy.spatial.transform import Rotation as R
from urdf_parser_py import urdf as urdf_parser
import urdf_parser_py.xml_reflection.core as urdf_parser_core
import polars as pl

# Import interfaces
from metacub_dashboard.interfaces.interfaces import (
    ActionInterface, 
    EncodersInterface, 
    CameraInterface
)
from metacub_dashboard.interfaces.utils.control_loop_reader import ControlLoopReader
from metacub_dashboard.visualizer.visualizer import Visualizer, Camera
from metacub_dashboard.visualizer.utils.blueprint import build_blueprint
from metacub_dashboard.data_logger.logger import DataLogger as PolarsDataLogger
from metacub_dashboard.data_logger.data_logger import DataLogger

urdf_parser_core.on_error = lambda x: x


def main():
    """Main function using pure Polars DataFrames throughout."""
    print("🚀 Starting MetaCub Dashboard with Pure Polars...")
    
    # Initialize YARP
    yarp.Network.init()
    session_id = uuid4()

    # 1. Create interfaces
    print("📡 Setting up interfaces...")
    
    action_interface = ActionInterface(
        remote_prefix="/metaControllClient",
        local_prefix=f"/metacub_dashboard/{session_id}",
        stream_name="target_poses"
    )

    observation_interfaces = {
        "main_camera": CameraInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            rgb_shape=(640, 480),
            depth_shape=None,
            stream_name="main_camera"
        ),
        "robot_joints": EncodersInterface(
            remote_prefix="/ergocubSim",
            local_prefix=f"/metacub_dashboard/{session_id}",
            control_boards=["head", "left_arm", "right_arm", "torso"],
            stream_name="robot_joints"
        ),
    }

    control_reader = ControlLoopReader(
        action_interface=action_interface,
        observation_interfaces=observation_interfaces,
        control_frequency=10.0,
        blocking=False,
    )

    # 2. Set up visualization
    print("🎨 Setting up visualization...")
    
    urdf_path = yarp.ResourceFinder().findFileByName("model.urdf")
    urdf = urdf_parser.URDF.from_xml_file(urdf_path)
    urdf.path = urdf_path

    # Build visualization blueprint
    camera_path = "/".join(
        urdf.get_chain(root=urdf.get_root(), tip="realsense_depth_frame")[0::2]
    )
    image_paths = [f"{camera_path}/cameras/agentview_rgb"]
    eef_paths = [
        "/".join(urdf.get_chain(root=urdf.get_root(), tip=eef)[0::2])
        for eef in ["l_hand_palm", "r_hand_palm"]
    ]

    blueprint = build_blueprint(
        image_paths=image_paths,
        eef_paths=eef_paths,
        poses=["target_poses", "robot_joints"],
    )

    visualizer = Visualizer(urdf=urdf, blueprint=blueprint, gradio=False)

    # 3. Set up data logging
    print("💾 Setting up data logging...")
    
    base_data_logger = DataLogger(
        path="assets/dataset.zarr.zip",
        flush_every=200,
        exist_ok=True
    )
    data_logger = PolarsDataLogger(base_data_logger)

    # 4. Main execution loop with DataFrame processing
    print("🔄 Starting control loop with DataFrame processing...")
    
    iteration = 0
    prev_time = time.perf_counter()
    
    try:
        while iteration < 50:
            # Read synchronized action/observation pair as pure DataFrames
            control_data = control_reader.read()
            
            if control_data is None:
                continue
                
            # ====== DATAFRAME PROCESSING ======
            
            # Extract DataFrames directly
            action_df = control_data.actions_df
            observation_df = control_data.observations_df
            
            # Step 1: Filter observation streams by type using native Polars
            camera_df = observation_df.filter(pl.col("stream_type") == "camera")
            encoder_df = observation_df.filter(pl.col("stream_type") == "encoders")
            
            # Step 2: Apply entity paths using Polars when/then/otherwise
            processed_encoder_df = encoder_df.with_columns([
                pl.when(pl.col("name").str.contains("left_arm"))
                .then(pl.lit(f"{eef_paths[0]}/fingers"))
                .when(pl.col("name").str.contains("right_arm"))  
                .then(pl.lit(f"{eef_paths[1]}/fingers"))
                .otherwise(pl.lit("joints"))
                .alias("entity_path")
            ])
            
            # Step 3: Combine processed data using pl.concat
            all_observation_df = pl.concat([camera_df, processed_encoder_df])
            
            # ====== PURE POLARS VISUALIZER ======
            
            # Log to visualizer with pure DataFrames
            visualizer.log_dataframes(
                poses_df=action_df,
                encoders_df=processed_encoder_df,
                camera_df=camera_df,
                timestamp=control_data.loop_timestamp,
                static=False,
            )
            
            # ====== END PURE POLARS VISUALIZATION ======
            
            # Data logging with pure DataFrames
            data_logger.log_dataframes(
                observations_df=all_observation_df,
                actions_df=action_df
            )
            
            # Print diagnostic info using pure Polars analytics
            current_time = time.perf_counter()
            if iteration % 10 == 0:
                collection_freq = 1 / (current_time - prev_time) if prev_time else 0
                print(f"📊 Iteration {iteration}: Collection freq: {collection_freq:.1f} Hz")
                
                # Pure Polars diagnostics
                print(f"   📷 Camera streams: {len(camera_df)}")
                print(f"   ⚙️  Encoder streams: {len(processed_encoder_df)}")
                print(f"   🎯 Pose streams: {len(action_df)}")
                
                # Show frequency statistics using Polars aggregation
                if len(all_observation_df) > 0:
                    freq_stats = all_observation_df.select([
                        pl.col("metadata").struct.field("frequency").mean().alias("avg_freq"),
                        pl.col("metadata").struct.field("frequency").max().alias("max_freq"),
                        pl.col("metadata").struct.field("read_delay").mean().alias("avg_delay")
                    ])
                    stats = freq_stats.row(0, named=True)
                    print(f"   📈 Avg freq: {stats['avg_freq']:.1f} Hz, Max: {stats['max_freq']:.1f} Hz")
                    print(f"   ⏱️  Avg read delay: {stats['avg_delay']*1000:.1f} ms")
            
            prev_time = current_time
            iteration += 1

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        print("🧹 Cleaning up...")
        
        # End episode with pure Polars diagnostics
        data_logger.end_episode()
        control_reader.close()
        yarp.Network.fini()
        print("✅ Pure Polars clean shutdown complete!")


if __name__ == "__main__":
    # Uncomment to test pure Polars operations
    # demo_pure_polars_operations()
    
    # Run main application with pure Polars
    main()
