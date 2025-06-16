# MetaCub Dashboard

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/platform-linux--64-lightgrey.svg)](https://github.com/your-org/metacub_dashboard)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![YARP](https://img.shields.io/badge/YARP-3.11.2-orange.svg)](https://www.yarp.it/)
[![Pixi](https://img.shields.io/badge/pixi-managed-purple.svg)](https://pixi.sh/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/your-org/metacub_dashboard)

A comprehensive data logging and visualization dashboard for the MetaCub humanoid robot. This project captures, stores, and visualizes robot data including joint encoders, camera feeds, and action commands in real-time during robot operation.

> [!IMPORTANT]
> You need to install **metaCub** as the dashboard only visualizes and logs the data produced by another project. Installation of metaCub is described in the [metaCub repository README](https://github.com/your-org/metaCub).

> [!NOTE]
> For now, only **Linux** is supported. Windows support will come in the future.

## Installation

### Prerequisites

1. **Install Pixi** following the [official installation guide](https://pixi.sh/latest/#installation)

### Setup

1. **Clone and enter the repository:**
   ```bash
   git clone https://github.com/your-org/metacub_dashboard.git
   cd metacub_dashboard
   ```

2. **Install dependencies:**
   ```bash
   pixi install
   ```

This will automatically install all required dependencies including YARP, Rerun SDK, Polars, and other necessary packages.

## Startup

### Prerequisites
- Ensure **metaCub** is properly installed and configured
- Start up metaCub first (refer to metaCub documentation)

### Running the Dashboard

1. **Activate the pixi environment:**
   ```bash
   pixi shell
   ```

2. **Start the dashboard:**
   ```bash
   python main.py
   ```

## Usage

### Data Collection Process

1. **Robot Reset**: The first thing the program will do is reset the robot and environment. This reset will be repeated at the beginning of each episode.

2. **Recording Trigger**: Data recording starts automatically when a non-None action arrives from the source process (e.g., metaCub).

3. **Episode Control**: While recording, you can control the episode using keyboard shortcuts:
   - **Spacebar**: End the episode and save the data
   - **Backspace**: End the episode and discard the data
   - **Ctrl+C**: Stop data collection entirely and exit the program

### Configuration

You can customize the dashboard behavior by modifying settings in the main script:

- **YARP Port Names**: Configure the names of YARP ports for communication
- **Dataset Name**: Set the name of the output dataset file
- **Episode Management**: If the dataset already exists, new episodes will be appended to it

### Data Format

The dashboard saves data in Zarr format (`.zarr.zip` files) containing:
- **Observation data**: Joint encoders, camera feeds, sensor readings
- **Action data**: Robot control commands
- **Episode metadata**: Timestamps, episode lengths, and metadata

## Visualizer

### Dataset Visualization Script

Use the `dataset_visualizer.py` script to visualize saved datasets and replay episodes.

#### Usage

```bash
python scripts/dataset_visualizer.py [dataset] [options]
```

#### Arguments

- **`dataset`** (optional): Path to the zarr.zip dataset file
  - Default: `assets/beer_data.zarr.zip`
  - Example: `python scripts/dataset_visualizer.py my_dataset.zarr.zip`

#### Options

- **`-e, --episode`** (int): Episode index to visualize
  - Default: `0` (first episode)
  - Example: `python scripts/dataset_visualizer.py -e 5`

- **`-s, --speed`** (float): Playback speed multiplier
  - Default: `1.0` (real-time speed)
  - Example: `python scripts/dataset_visualizer.py -s 2.0` (2x speed)

- **`-l, --list`**: List available episodes and exit
  - Example: `python scripts/dataset_visualizer.py -l`

#### Examples

```bash
# List all episodes in the default dataset
python scripts/dataset_visualizer.py --list

# Visualize episode 3 at normal speed
python scripts/dataset_visualizer.py -e 3

# Visualize episode 1 at half speed
python scripts/dataset_visualizer.py -e 1 -s 0.5

# Visualize a specific dataset file
python scripts/dataset_visualizer.py my_custom_dataset.zarr.zip -e 2
```

## Project Structure

```
metacub_dashboard/
├── metacub_dashboard/           # Main package
│   ├── data_logger/            # Data logging functionality
│   ├── interfaces/             # YARP and robot interfaces
│   ├── utils/                  # Utility functions
│   ├── visualizer/             # Real-time visualization
│   └── main.py                # Main application entry point
├── scripts/                    # Utility scripts
│   ├── dataset_visualizer.py   # Dataset replay and visualization
│   └── ...                    # Other utility scripts
├── assets/                     # Sample data and resources
├── pyproject.toml             # Project configuration
└── pixi.lock                  # Dependency lock file
```

## Dependencies

The project uses several key dependencies:
- **YARP**: Robot communication middleware
- **Rerun SDK**: Real-time visualization
- **Polars**: High-performance data processing
- **Zarr**: Efficient data storage format
- **PyTorch**: Machine learning framework support

## Troubleshooting

### Common Issues

1. **YARP Connection Issues**: Ensure metaCub is running and YARP ports are properly configured
2. **Graphics Issues**: On Linux, ensure proper OpenGL drivers are installed
3. **Permission Issues**: Make sure you have write permissions in the directory where datasets are saved

### Support

For issues related to:
- **MetaCub robot setup**: Refer to the metaCub repository
- **Dashboard functionality**: Open an issue in this repository
