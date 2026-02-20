# 🤖 RosBot Autonomous Mapper & Navigator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Webots](https://img.shields.io/badge/Simulator-Webots-red)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green?logo=opencv)
![Pygame](https://img.shields.io/badge/UI-Pygame-yellow)

An autonomous control system developed for the RosBot mobile robot. This project features a unified, highly adaptable controller capable of exploring unknown maze environments, mapping structural geometry, identifying semantic landmarks, avoiding hazards, and executing high-speed optimal pathing—all without relying on hardcoded layouts.

## 🎥 Demonstrations

*(Add GIFs or YouTube links of your robot solving the mazes here. GIFs are highly recommended for GitHub!)*

| Maze 1 | Maze 3 | Maze 5 (Complex) |
| :---: | :---: | :---: |
| [![Maze 1](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) | [![Maze 3](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) | [![Maze 5](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID) |
*Click the images above to watch the full run videos.*

## ✨ Key Features

- **Unified Architecture:** A single Finite State Machine (FSM) controller solves multiple diverse maze layouts. Adaptation is achieved via minor hyperparameter tuning, not hardcoded logic.
- **Probabilistic SLAM:** Real-time 2D occupancy grid generation using 360° LiDAR and Bresenham’s Line Algorithm.
- **Dynamic Path Planning:** Utilizes the **A* (A-Star)** algorithm on a dynamically generated "Inflated Map" to ensure safe cornering and obstacle avoidance.
- **Semantic Computer Vision:** Uses HSV color thresholding and contour mapping to identify:
  - 🔵🟡 **Pillars (Goals):** Fuses 2D camera bearings with 3D LiDAR depth for precise target localization.
  - 🟢 **Poison Carpets (Hazards):** Utilizes projective geometry to map flat floor textures onto the 3D map.
  - 🔴 **Red Walls (Fatal Obstacles):** Triggers emergency braking and active avoidance.
- **Visual Homing & Dead-End Memory:** The robot "remembers" traps to prevent infinite loops and uses distant visual cues (target pixels) to bias frontier exploration down long corridors.

## 🛠️ System Architecture

### The "Sense-Plan-Act" Loop
The controller runs on a 32ms timestep, seamlessly integrating mapping, vision, and navigation. 

1. **Perception:** Lidar point clouds are ray-traced onto a probability grid. Camera feeds are processed via OpenCV to detect target colors.
2. **Cognition:** - *Exploration:* Identifies map "frontiers" (edges of known/unknown space) and scores them based on distance, size, and visual alignment.
   - *Safety:* Inflates the occupancy grid to account for the robot's physical radius, generating a safe configuration space.
3. **Action:** Calculates A* paths to the best frontier or target, utilizing a PID-like heading correction to drive smoothly.

## 🚀 How to Run

### Prerequisites
You need [Webots](https://cyberbotics.com/) installed on your machine and a Python environment.

Install the required Python packages:
```bash
pip install numpy opencv-python pygame
