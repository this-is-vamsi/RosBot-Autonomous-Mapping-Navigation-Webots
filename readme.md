# 🤖 RosBot Autonomous Mapper & Navigator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Webots](https://img.shields.io/badge/Simulator-Webots-red)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-green?logo=opencv)
![Pygame](https://img.shields.io/badge/UI-Pygame-yellow)

An advanced autonomous navigation and semantic mapping system developed for the **RosBot** platform. This project features a unified, highly adaptable controller capable of exploring unknown maze environments, mapping structural geometry, and identifying semantic landmarks without hardcoded layout data.

The system utilizes a specialized **Sense → Plan → Act** pipeline to solve diverse maze layouts using the exact same core logic, adapted only via minor hyperparameter tuning.

---

## 📺 Project Demonstration

🎥 Watch the full project execution (Click the link below to view the robot solving complex environments)
| Maze 1 (incomplete) | Maze 2 | Maze 3 | Maze 4 | Maze 5 |
| :---: | :---: | :---: | :---: | :---: |
| [![Maze 1](https://drive.google.com/file/d/1QZHZpnhG1CD_ujp4BcvCTXvQayxwz7-O/view?usp=sharing) | [![Maze 2](https://drive.google.com/file/d/100V5A_OpzAN0D6-X1i9v3UarD0YugCVk/view?usp=sharing) | [![Maze 3](https://drive.google.com/file/d/1Atj4pOJkYmPBaaMFD5G_hTV-cq4udAjN/view?usp=sharing) | [![Maze 4](https://drive.google.com/file/d/1Z9a1RK9zhQ5Hqt7SH47c8B69scG74zuc/view?usp=sharing) | [![Maze 5](https://drive.google.com/file/d/1J421rTs8WrWG7BcnbvRVcDs3GCdNIcAc/view?usp=sharing) |

## 🚀 Key Features

- **Unified FSM Architecture**  
  A single Finite State Machine (FSM) controller solves multiple diverse maze layouts. Logic is generalized to handle variable corridor widths and wall densities.

- **Probabilistic SLAM & Mapping**  
  Real-time 2D occupancy grid generation using 360° LiDAR and Bresenham’s Line Algorithm. Includes dynamic map inflation for safe navigation buffers.

- **Dynamic A* Path Planning**  
  Optimal trajectory generation on an inflated cost-map. The pathfinder dynamically re-plans if a path becomes blocked or if a target's position is refined.

- **Semantic Computer Vision**  
  Fuses 2D camera bearings with 3D LiDAR depth for landmark localization. Uses OpenCV for HSV-based segmentation and projective geometry to map floor-level hazards.

- **Intelligent Exploration & Recovery**  
  Frontier-based exploration scoring, Visual Homing (biasing movement toward distant goals), and a "Lost Mode" random search fallback for closed environments.

---

# 🧠 Technical Deep Dive

---

## 1️⃣ Navigation & Cognition

The decision engine manages exploration through a multi-weighted scoring system:

### Frontier Clustering
- Groups unknown map edges  
- Scores frontiers based on distance and size  

### Visual Homing
- If Blue pixels are visible in the distance, the system applies a massive heuristic bonus  
- Frontiers in that direction are heavily prioritized  
- Pulls the robot toward distant goals

### Dead-End Memory
- Blacklists a 3.0m radius around Red Wall encounters  
- Prevents infinite pathing loops  
- Improves long-term exploration efficiency  

---

## 2️⃣ Mapping & Perception (The Eyes)

### Occupancy Grid
- Each cell stores a probability of occupancy (0.0 → 1.0)  
- Updated using LiDAR "hits" and "misses"  
- Enables probabilistic environment modeling  

### Projective Geometry
- LiDAR misses flat floor hazards (Green Poison)  
- Camera pixel contours are projected into 3D world coordinates  
- Rigid bounding boxes are fitted to the occupancy grid  

### Initial 360° Scan
- Slow rotation at startup  
- Immediately populates the map with visible features  
- Prevents driving away from goals located behind the robot  

---

## 3️⃣ Actuation & Control (The Hands)

### PID-like Heading Correction
- Smooth movement via turn-correction gains  
- Speed damping as waypoints are approached  
- Reduces oscillations in narrow corridors  

### Two-Phase Recovery
For collision handling:

1. Forced reverse phase to gain clearance  
2. 180° rotation to escape tight geometries  

Improves robustness in dense maze layouts.

---

# 📁 Project Structure

```bash
├── final_code.py         # Main Controller (Unified Logic)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

# 👥 Contributors

* **Satya Naga Vamsi Ganesh Manepalli**
* **Suresh Dangeti** 
* **Nagendra Mandapati**
* **Dharsan Morkar** 


**Institution:** OTH Amberg-Weiden  
**Module:** Autonomous Robotics (Winter 2025–26)
