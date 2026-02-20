# 🤖 RosBot Autonomous Mapper & Navigator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Webots](https://img.shields.io/badge/Simulator-Webots-red)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green?logo=opencv)
![Pygame](https://img.shields.io/badge/UI-Pygame-yellow)

An autonomous control system developed for the RosBot mobile robot. This project features a unified, highly adaptable controller capable of exploring unknown maze environments, mapping structural geometry, identifying semantic landmarks, avoiding hazards, and executing high-speed optimal pathing—all without relying on hardcoded layouts.

## 🎥Project Demonstration

| Maze 1 (incomplete) | Maze 2 | Maze 3 | Maze 4 | Maze 5 |
| :---: | :---: | :---: | :---: | :---: |
| [[Maze 1](https://drive.google.com/file/d/1QZHZpnhG1CD_ujp4BcvCTXvQayxwz7-O/view?usp=sharing) | [[Maze 2](https://drive.google.com/file/d/100V5A_OpzAN0D6-X1i9v3UarD0YugCVk/view?usp=sharing) | [[Maze 3](https://drive.google.com/file/d/1Atj4pOJkYmPBaaMFD5G_hTV-cq4udAjN/view?usp=sharing) | [[Maze 4](https://drive.google.com/file/d/1Z9a1RK9zhQ5Hqt7SH47c8B69scG74zuc/view?usp=sharing) | [[Maze 5](https://drive.google.com/file/d/1J421rTs8WrWG7BcnbvRVcDs3GCdNIcAc/view?usp=sharing) |


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

## 🧠 Engineering Solutions to Complex Problems

During development, we solved several critical edge cases:

* **The "Red Wall" Infinite Loop:** Standard planners get stuck turning back and forth at red walls. We engineered a **Dead-End Memory** that blacklists a 3.0m radius around red walls, forcing the robot to seek entirely new routes.
* **The "Ghost Poison" (Projective Geometry):** Flat floor hazards are invisible to LiDAR. We used homography to map the camera's 2D pixel contours to world coordinates, fitting bounding boxes (`cv2.minAreaRect`) to the floor and locking them into the A* cost-map.
* **Lost Mode Fallback:** If the robot fully maps an environment without finding the goal, it autonomously navigates to a random open space and performs a 360° sensor sweep to catch obscured targets.

---

# 📁 Project Structure

```bash
├── final_code.py         # Main Controller (Unified Logic)
├── Project_Report.pdf    # Detailed Description of the Project
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 👨‍💻 Contributors

This project was developed for the Autonomous Robotics module (Winter 2025-26) by **Team AR**:

* **Vamsi Ganesh**
* **Suresh Dangeti** 
* **Nagendra Mandapati**
* **Dharsan Morkar** 

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




