# RosBot Autonomous Mapper - Execution Guide

## Overview
This project uses a unified Python controller to navigate, map, and solve 5 distinct maze environments. The robot utilizes LiDAR for mapping (SLAM), A* for path planning, and Camera vision for semantic object detection (Pillars, Poison, Red Walls).

## Prerequisites
Ensure the following Python libraries are installed in your Webots environment:

```bash
pip install numpy opencv-python pygame


How to Run
Launch Webots: Open the desired world file (e.g., Maze1.wbt, Maze2.wbt, etc.).

Select Controller:

Click on the RosBot robot in the scene tree.

In the controller field, select the corresponding python file (e.g., final_code2.py for Maze 2, final_code5.py for Maze 5).

Note: All controllers share the same core logic; only hyperparameters (speed, detection thresholds) are tuned for specific environment constraints.

Visualization: A Pygame window will open automatically, displaying:

Left: The real-time Occupancy Grid, Planned Path (Blue), and Frontiers (Red).

Right: The Live Camera feed with object detection overlays.

Simulation: Press the Play button in Webots"# RosBot-Autonomous-Mapping-Navigation-Webots" 
"# RosBot-Autonomous-Mapping-Navigation-Webots" 
