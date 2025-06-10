
from rerun.blueprint import (
    Blueprint,
    BlueprintPanel,
    Horizontal,
    Vertical,
    SelectionPanel,
    Spatial3DView,
    TimePanel,
    TimeSeriesView,
    Tabs,
    BarChartView,
    Spatial2DView
)

def get_color(s: str) -> list[float]:
    """
    Convert a string to a RGB color.
    
    Args:
        s (str): Input string
    
    Returns:
        list[float]: RGB color values between 0 and 1
    """
    # Generate a hash of the string
    hash_value = hash(s)
    
    # Use the hash to generate RGB values
    r = ((hash_value & 0xFF0000) >> 16) / 255.0
    g = ((hash_value & 0x00FF00) >> 8) / 255.0
    b = (hash_value & 0x0000FF) / 255.0
    
    return [r, g, b, 1.0]


import rerun.blueprint as rrb
import rerun as rr


def build_blueprint(
    eef_paths: list[str] = [],
    image_paths: list[str] = [],
    poses: list[str] = [],
):
    """
    Build the blueprint for the visualizer.
    """

    blueprint = Blueprint(
        Horizontal(
            # Left Side
            Vertical(
                Spatial3DView(name="robot view", origin="/", contents=["/**"], overrides={path: rr.Transform3D.from_fields(axis_length=0.1) for path in eef_paths}),
                Horizontal(*(Spatial2DView(name=f'{img}', origin=img) for img in image_paths), name='Camera Views'),
                row_shares=[3, 1],
            ),
            # Right Side
            Tabs(
                # Poses
                Tabs(
                    Vertical(
                        TimeSeriesView(name='x', origin='poses', contents=[f'poses/{pose}/components/x/**' for pose in poses],         
                                     overrides={f'poses/{pose}/components/x': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        TimeSeriesView(name='y', origin='poses', contents=[f'poses/{pose}/components/y/**' for pose in poses],
                                     overrides={f'poses/{pose}/components/y': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        TimeSeriesView(name='z', origin='poses', contents=[f'poses/{pose}/components/z/**' for pose in poses],
                                     overrides={f'poses/{pose}/components/z': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        name='Position'
                    ),
                    Vertical(
                        TimeSeriesView(name='ax', origin='poses', contents=[f'poses/{pose}/components/ax/**' for pose in poses],
                                     overrides={f'poses/{pose}/components/ax': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        TimeSeriesView(name='ay', origin='poses', contents=[f'poses/{pose}/components/ay/**' for pose in poses],
                                     overrides={f'poses/{pose}/components/ay': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        TimeSeriesView(name='az', origin='poses', contents=[f'poses/{pose}/components/az/**' for pose in poses],
                                     overrides={f'poses/{pose}/components/az': rr.SeriesLines.from_fields(names=[pose], widths=[2.0], colors=[get_color(pose)]) for pose in poses}),
                        name='Orientation'
                    ),
                    name='Poses'
                ),
                # EEF View
                Vertical(
                    Spatial3DView(name='End Effector View Front', origin=eef_paths[0], contents=[f'{eef_paths[0]}/**', '/trajectories/**', f'- {eef_paths[0]}/right_camera']),
                    Spatial3DView(name='End Effector View Right', origin=eef_paths[1], contents=[f'{eef_paths[1]}/**', '/trajectories/**', f'- {eef_paths[1]}/front_camera']),
                    name='End Effector View',
                ),
                # Joints
                Vertical(
                    TimeSeriesView(name='Joints', origin='joints', contents=['joints/**']),	  
                    name='Joints',
                ),
                # Streams
                Vertical(
                    TimeSeriesView(name='Write Frequency', origin='streams', contents=['streams/write_frequency/**']),
                    TimeSeriesView(name='Write Timestamp', origin='streams', contents=['streams/write_timestamp/**']),
                    TimeSeriesView(name='Read Delay', origin='streams', contents=['streams/read_delay/**']),
                    TimeSeriesView(name='Read Attempts', origin='streams', contents=['streams/read_attempts/**']),
                    TimeSeriesView(name='Read Timestamp', origin='streams', contents=['streams/read_timestamp/**']),
                    TimeSeriesView(name='Missed Packets', origin='streams', contents=['streams/missed_packets/**']),
                    name='Streams',
                ),
            ),
            column_shares=[3, 2],
        ),
        BlueprintPanel(state='collapsed'),
        SelectionPanel(state='collapsed'),
        TimePanel(state='collapsed'),
        auto_views=False,
    )

    return blueprint
