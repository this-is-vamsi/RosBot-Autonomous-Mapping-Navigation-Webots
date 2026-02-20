"""
RosBot Final System - "Visual Homing Update"

NEW FEATURES:
1. VISUAL HOMING: If the camera sees Blue pixels in the distance, the robot 
   prioritizes frontiers in that direction.
2. VISUAL FALLBACK: If no frontiers are found, but Blue is visible, the robot 
   drives towards the Blue pixels instead of searching randomly.
3. PRESERVED: Collision Recovery, Red Wall Logic, Fixed Poison, Speed Run.
"""

from controller import Robot, Supervisor, Lidar, Camera, Keyboard
import math
import numpy as np
import heapq
import collections
import random
import colorsys
import json
import time

# --- Pygame Import ---
try:
    import pygame
except ImportError:
    print("Error: Pygame is not installed. Please run 'pip install pygame'")
    exit(1)

# --- OpenCV Import ---
try:
    import cv2
except ImportError:
    print("Error: OpenCV is not installed. Please run 'pip install opencv-python'")
    exit(1)

# --- Simulation Constants ---
TIME_STEP = 32
MAX_VELOCITY = 26.0
PLOT_UPDATE_RATE = 20 

# --- Robot Physical Parameters ---
ROBOT_WIDTH = 0.25
LIDAR_HEIGHT = 0.15
CAMERA_HEIGHT_METERS = 0.2
PILLAR_RADIUS_METERS = 0.1
HAZARD_BLOCK_RADIUS_METERS = 0.6 

# --- Map Parameters ---
MAP_SIZE = 500 
SCALE = 40

# Map grid: 0.5 = unknown, 0.0 = free, 1.0 = occupied, -1.0 = RED WALL
map_grid = np.full((MAP_SIZE, MAP_SIZE), 0.5, dtype=np.float32) 

# Inflated map grid (Backend only for A*)
inflated_map_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)

# --- Probabilistic Mapping Parameters ---
MAP_INITIAL_PROB = 0.5       
PROB_OCC_INCREMENT = 0.1    
PROB_FREE_DECREMENT = 0.01  
PROB_MAX_CONFIDENCE = 0.99   
PROB_MIN_CONFIDENCE = 0.01   
PROB_OCCUPIED_THRESHOLD = 0.7 
RED_WALL_GRID_VALUE = -1.0

# --- Robot Control Parameters ---
AUTONOMOUS_BASE_SPEED = 8.0
AUTONOMOUS_TURN_SPEED = 4.0 
INITIAL_SCAN_SPEED = 2.0      
FINAL_RUN_SPEED = 10.0 
TURN_CORRECTION_GAIN = 2.0
HEADING_TOLERANCE = 0.05 

# --- LiDAR Parameters ---
LIDAR_MIN_DISTANCE = 0.05 
LIDAR_MAX_DISTANCE = 10.0
LIDAR_MAX_MAPPING_DISTANCE = 5.0 
LIDAR_INFLATION_RADIUS_MAP = 1 
LIDAR_FOV = 2 * math.pi 
LIDAR_RESOLUTION = 360 

# --- Autonomous Exploration States ---
STATE_EXPLORING = 0
STATE_FOLLOWING_PATH = 1
STATE_MAPPING_COMPLETE = 3
STATE_STUCK_RECOVERY = 4
STATE_AVOIDING_HAZARD = 5
STATE_CENTERING_ON_PILLAR = 6 
STATE_RETURNING_TO_START = 7 
STATE_FINAL_RUN = 8          
STATE_FINISHED = 9
STATE_AVOIDING_RED_WALL = 10 
STATE_LOST_SEARCH_ROTATING = 11 
STATE_INITIAL_SPIN = 12       

# --- Path Planning Parameters ---
WAYPOINT_REACH_THRESHOLD_METERS = 0.05 
FRONTIER_SEARCH_RADIUS_METERS = 15.0 
STUCK_ROTATION_ANGLE = math.pi / 2 
PILLAR_RING_RADIUS_METERS = 0.30 

# --- Red Wall / Dead End Parameters ---
RED_WALL_PIXEL_THRESHOLD = 0.2  
RED_WALL_REVERSE_TIME = 1.0      
RED_WALL_TURN_ANGLE = math.pi    
DEAD_END_RADIUS_METERS = 2.0     
RED_WALL_PROXIMITY_CHECK_RADIUS = 2.0 

# --- Stuck Recovery Specific Parameters ---
RECOVERY_REVERSE_DURATION = 1.0 
RED_WALL_AVOID_THRESHOLD_METERS = 0.75 
COLLISION_THRESHOLD_METERS = 0.20 

# --- Frontier Selection Parameters ---
MIN_FRONTIER_CLUSTER_SIZE = 1 
MAX_FRONTIER_CLUSTERS_TO_CONSIDER = 50 
FRONTIER_SIZE_WEIGHT = 0.1

# --- Memory & Anti-Loop Parameters ---
UNREACHABLE_FRONTIER_MEMORY = {}
BLACKLISTED_FRONTIERS = [] 
DEAD_END_FRONTIERS = [] 
TEMPORARY_OBSTACLES = [] 
TEMP_OBSTACLE_RADIUS = 0.8 
TEMP_OBSTACLE_DURATION = 45.0 
BLACKLIST_RADIUS_METERS = 2.0 
MEMORY_DURATION = 30.0 

# ** Directional Memory **
BAD_DIRECTION_MEMORY = None 
BAD_DIRECTION_DURATION = 30.0 
BAD_DIRECTION_CONE = 2 * math.pi / 3 

# ** Hazard Recovery Cooldown **
HAZARD_RECOVERY_COOLDOWN_TIME = None 

# --- Poison Zone Parameters ---
POISON_GRID_VALUE = -2.0  
POISON_SPATIAL_DEBOUNCE_METERS = 2.0 

# --- Robot Inflation & Safety Cost ---
ROBOT_INFLATION_RADIUS_CELLS = int((ROBOT_WIDTH / 2.0 + 0.01) * SCALE) 
SAFETY_BUFFER_CELLS = ROBOT_INFLATION_RADIUS_CELLS + 4 

# A* Costs
COST_MOVE_SAFE = 1
COST_MOVE_CAUTION = 25 

# --- Pillar Detection Parameters ---
DISTANCE_DISCREPANCY_THRESHOLD = 0.3 
MAX_PILLAR_WIDTH = 0.25 
CENTERING_TOLERANCE_RAD = 0.1 
PILLAR_MAX_DETECTION_RANGE = 3.0

# Color Ranges
BLUE_HSV_LOWER = (0.55, 0.5, 0.5)
BLUE_HSV_UPPER = (0.75, 1.0, 1.0)
YELLOW_HSV_LOWER = (0.1, 0.5, 0.5)
YELLOW_HSV_UPPER = (0.25, 1.0, 1.0)
GREEN_HSV_LOWER = (0.25, 0.4, 0.3)  
GREEN_HSV_UPPER = (0.45, 1.0, 1.0)

detected_pillars = [] 
mission_pillars = {
    'blue': {'x': 0, 'y': 0, 'min_dist': 999.0, 'found': False}, 
    'yellow': {'x': 0, 'y': 0, 'min_dist': 999.0, 'found': False}, 
    'poison_list': []
} 

# --- Global Control Variables ---
mapping_enabled = True
final_run_start_time = 0.0
final_run_active = False
is_random_searching = False 

# --- Initialize Robot ---
robot = Supervisor()
time_step = int(robot.getBasicTimeStep())
rosbot_node = robot.getFromDef("ROSBot")
if not rosbot_node:
    print("CRITICAL ERROR: Robot node with DEF 'ROSBot' not found!")
    exit(1)

# --- Motor Devices ---
motors = [robot.getDevice(n) for n in ["fl_wheel_joint", "fr_wheel_joint", "rl_wheel_joint", "rr_wheel_joint"]]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

def motors_set_velocity(left_speed, right_speed):
    left_speed = max(-MAX_VELOCITY, min(left_speed, MAX_VELOCITY))
    right_speed = max(-MAX_VELOCITY, min(right_speed, MAX_VELOCITY))
    motors[0].setVelocity(left_speed)
    motors[2].setVelocity(left_speed)
    motors[1].setVelocity(right_speed)
    motors[3].setVelocity(right_speed)

def motors_stop():
    for m in motors: m.setVelocity(0.0)

motors_stop()
for _ in range(10): robot.step(TIME_STEP)

# --- Sensors ---
lidar = robot.getDevice("laser")
lidar.enable(time_step)
lidar.enablePointCloud()

distance_sensors = [robot.getDevice("fl_range"), robot.getDevice("fr_range")]
for ds in distance_sensors: ds.enable(time_step)

camera_rgb = robot.getDevice("camera rgb")
camera_rgb.enable(time_step)
camera_width = camera_rgb.getWidth()
camera_height = camera_rgb.getHeight()
camera_fov = camera_rgb.getFov()
camera_fov_vertical = 2 * math.atan(math.tan(camera_fov / 2) * (camera_height / camera_width))

keyboard = robot.getKeyboard()
keyboard.enable(time_step)

# --- Mapping Variables ---
robot_x_map_global = 0.0
robot_y_map_global = 0.0
robot_theta_global = 0.0
last_theta_global = 0.0 

# --- START IN INITIAL SPIN ---
current_robot_state = STATE_INITIAL_SPIN
total_accumulated_spin = 0.0
# -----------------------------

current_path = []
path_waypoint_index = 0
stuck_timer_start = 0.0
target_avoidance_heading = 0.0 
centering_target_color = None 

# --- Movement Monitor Variables ---
last_stuck_check_time = 0.0
last_stuck_check_pos = (0.0, 0.0)
STUCK_CHECK_INTERVAL = 5.0 
MIN_MOVEMENT_DISTANCE = 0.1
# --- Target Tracking ---
current_target_type = None 
current_pillar_target_coords = None

# --- Visualization Variables ---
selected_frontier_for_viz = None 
planned_path_for_viz = []
solution_path_for_viz = [] 

# --- Pygame Setup ---
pygame.init()
window_size = (MAP_SIZE, MAP_SIZE)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("RosBot Final System")

# Color Definitions
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (127, 127, 127)
COLOR_ROBOT = (0, 255, 0) 
COLOR_PATH = (0, 0, 255) 
COLOR_FRONTIER = (255, 0, 0) 
COLOR_PILLAR_BLUE = (0, 0, 200)
COLOR_PILLAR_YELLOW = (200, 200, 0)
COLOR_BLACKLIST = (100, 0, 0) 
COLOR_SOLUTION = (255, 0, 255) 
COLOR_TEMP_OBSTACLE = (255, 100, 0)
COLOR_POISON = (0, 255, 0)  

# --- Helper Functions ---
def get_robot_global_pose():
    position = rosbot_node.getPosition()
    rotation = rosbot_node.getOrientation()
    return position, rotation

def rotation_to_yaw(rotation_matrix):
    yaw = math.atan2(rotation_matrix[3], rotation_matrix[0])
    if yaw > math.pi: yaw -= 2 * math.pi
    elif yaw < -math.pi: yaw += 2 * math.pi
    return yaw

def world_to_map(wx, wy):
    mx = int(wx * SCALE + MAP_SIZE / 2)
    my = int(wy * SCALE + MAP_SIZE / 2)
    return mx, my

def map_to_world(mx, my):
    wx = (mx - MAP_SIZE / 2) / SCALE
    wy = (my - MAP_SIZE / 2) / SCALE
    return wx, wy

def map_to_screen(mx, my):
    sx = mx
    sy = MAP_SIZE - 1 - my
    return sx, sy

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def transform_lidar_point(point, robot_theta):
    dx = (point.x * math.cos(robot_theta)) - (point.y * math.sin(robot_theta))
    dy = (point.x * math.sin(robot_theta)) + (point.y * math.cos(robot_theta))
    return dx, dy

def check_surround_collision():
    lidar_points = lidar.getPointCloud()
    for point in lidar_points:
        if not (math.isfinite(point.x) and math.isfinite(point.y)): continue
        if point.x > 0: 
            dist = math.sqrt(point.x**2 + point.y**2)
            if dist < COLLISION_THRESHOLD_METERS:
                return True
    return False

def add_area_to_memory(target_mx, target_my):
    radius = 3 
    t = robot.getTime()
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            if dx**2 + dy**2 <= radius**2:
                UNREACHABLE_FRONTIER_MEMORY[(target_mx+dx, target_my+dy)] = t

def mark_nearby_frontiers_as_dead_end(robot_x, robot_y):
    global DEAD_END_FRONTIERS
    DEAD_END_FRONTIERS.append((robot_x, robot_y))
    print(f"🚫 MARKED DEAD END at ({robot_x:.2f}, {robot_y:.2f})")

def is_frontier_near_dead_end(frontier_wx, frontier_wy):
    for dead_x, dead_y in DEAD_END_FRONTIERS:
        dist = math.sqrt((frontier_wx - dead_x)**2 + (frontier_wy - dead_y)**2)
        if dist < DEAD_END_RADIUS_METERS:
            return True
    return False

def is_frontier_surrounded_by_red_walls(frontier_wx, frontier_wy):
    check_radius_cells = int(RED_WALL_PROXIMITY_CHECK_RADIUS * SCALE)
    angles = [i * (2 * math.pi / 8) for i in range(8)]
    red_wall_hits = 0
    
    for angle in angles:
        dx = math.cos(angle)
        dy = math.sin(angle)
        for step in range(1, check_radius_cells):
            check_x = frontier_wx + (step / SCALE) * dx
            check_y = frontier_wy + (step / SCALE) * dy
            check_mx, check_my = world_to_map(check_x, check_y)
            
            if not (0 <= check_mx < MAP_SIZE and 0 <= check_my < MAP_SIZE): break
                
            if map_grid[check_my][check_mx] == RED_WALL_GRID_VALUE:
                red_wall_hits += 1
                break
            if map_grid[check_my][check_mx] >= PROB_OCCUPIED_THRESHOLD:
                break
                
    return red_wall_hits >= 5 

def project_pixel_to_world(u, v, rx, ry, rt):
    cx = camera_width / 2
    cy = camera_height / 2
    angle_v = ((v - cy) / cy) * (camera_fov_vertical / 2.0)
    angle_h = ((cx - u) / cx) * (camera_fov / 2.0)
    
    if angle_v < 0.05: return None 
    
    dist = CAMERA_HEIGHT_METERS / math.tan(angle_v)
    if dist > 3.0: return None 
    
    wx = rx + dist * math.cos(rt + angle_h)
    wy = ry + dist * math.sin(rt + angle_h)
    return wx, wy

# --- NEW: VISUAL BEARING HELPER ---
def get_visual_bearing(image, target_color='blue'):
    """
    Returns the angle (radians) to the largest blob of the specified color.
    Returns None if not found or too small.
    """
    img_np = np.frombuffer(image, np.uint8).reshape((camera_height, camera_width, 4))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    lower = np.array([100, 150, 50]) if target_color == 'blue' else np.array([20, 100, 100])
    upper = np.array([140, 255, 255]) if target_color == 'blue' else np.array([40, 255, 255])
    
    if target_color == 'blue':
        lower = np.array([100, 150, 50])
        upper = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower, upper)
    
    # Clean noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None, 0.0
    
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: return None, 0.0 # Too small/far noise
    
    M = cv2.moments(c)
    if M["m00"] == 0: return None, 0.0
    
    cX = int(M["m10"] / M["m00"])
    
    # Calculate angle relative to robot center
    # 0 is center, negative left, positive right
    angle_h = (0.5 - (cX / camera_width)) * camera_fov
    
    return angle_h, cv2.contourArea(c)

def update_map_lidar(robot_x_world, robot_y_world, robot_theta):
    global map_grid 
    if not mapping_enabled: return 

    lidar_points = lidar.getPointCloud()
    robot_mx, robot_my = world_to_map(robot_x_world, robot_y_world)

    if 0 <= robot_mx < MAP_SIZE and 0 <= robot_my < MAP_SIZE:
        if map_grid[robot_my][robot_mx] != RED_WALL_GRID_VALUE and map_grid[robot_my][robot_mx] != POISON_GRID_VALUE: 
            map_grid[robot_my][robot_mx] = PROB_MIN_CONFIDENCE 

    for point in lidar_points:
        if not (math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)): continue
        distance = math.sqrt(point.x**2 + point.y**2 + point.z**2) 
        if (distance < LIDAR_MIN_DISTANCE or distance > LIDAR_MAX_DISTANCE or abs(point.z) > 0.1): continue
        
        if distance > LIDAR_MAX_MAPPING_DISTANCE: continue 

        dx, dy = transform_lidar_point(point, robot_theta)
        ox = robot_x_world + dx
        oy = robot_y_world + dy
        obstacle_mx, obstacle_my = world_to_map(ox, oy)
        
        if (0 <= robot_mx < MAP_SIZE and 0 <= robot_my < MAP_SIZE and
           0 <= obstacle_mx < MAP_SIZE and 0 <= obstacle_my < MAP_SIZE):
            line_points = bresenham_line(robot_mx, robot_my, obstacle_mx, obstacle_my)
            for mx, my in line_points[:-1]: 
                if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                    if map_grid[my][mx] != RED_WALL_GRID_VALUE and map_grid[my][mx] != POISON_GRID_VALUE:
                        if map_grid[my][mx] > PROB_MIN_CONFIDENCE:
                            map_grid[my][mx] = max(PROB_MIN_CONFIDENCE, map_grid[my][mx] - PROB_FREE_DECREMENT)
        
        if 0 <= obstacle_mx < MAP_SIZE and 0 <= obstacle_my < MAP_SIZE:
            if map_grid[obstacle_my][obstacle_mx] != RED_WALL_GRID_VALUE and map_grid[obstacle_my][obstacle_mx] != POISON_GRID_VALUE:
                if map_grid[obstacle_my][obstacle_mx] < PROB_MAX_CONFIDENCE:
                    map_grid[obstacle_my][obstacle_mx] = min(PROB_MAX_CONFIDENCE, map_grid[obstacle_my][obstacle_mx] + PROB_OCC_INCREMENT)

def update_inflated_map():
    global inflated_map_grid, TEMPORARY_OBSTACLES
    if not mapping_enabled and inflated_map_grid[0][0] != 0: return 

    inflated_map_grid.fill(1) 
    
    pillar_cells = set()
    rad_pixels = int(PILLAR_RADIUS_METERS * SCALE) + 1
    
    for p_color in ['blue', 'yellow']:
        if mission_pillars[p_color]['found']:
            px, py = mission_pillars[p_color]['x'], mission_pillars[p_color]['y']
            pmx, pmy = world_to_map(px, py)
            for dy in range(-rad_pixels, rad_pixels + 1):
                for dx in range(-rad_pixels, rad_pixels + 1):
                    if dx**2 + dy**2 <= rad_pixels**2:
                        pillar_cells.add((pmx + dx, pmy + dy))

    occupied_cells = np.argwhere(
        (map_grid >= PROB_OCCUPIED_THRESHOLD) | 
        (map_grid == RED_WALL_GRID_VALUE) | 
        (map_grid == POISON_GRID_VALUE)
    )

    inflation_sources = []
    for y, x in occupied_cells:
        if (x, y) not in pillar_cells:
            inflation_sources.append((y, x))

    for y_occ, x_occ in inflation_sources:
        for i in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
            for j in range(-SAFETY_BUFFER_CELLS, SAFETY_BUFFER_CELLS + 1):
                imx, imy = x_occ + j, y_occ + i 
                if 0 <= imx < MAP_SIZE and 0 <= imy < MAP_SIZE:
                    inflated_map_grid[imy][imx] = 3
    for y_occ, x_occ in inflation_sources:
        for i in range(-ROBOT_INFLATION_RADIUS_CELLS, ROBOT_INFLATION_RADIUS_CELLS + 1):
            for j in range(-ROBOT_INFLATION_RADIUS_CELLS, ROBOT_INFLATION_RADIUS_CELLS + 1):
                imx, imy = x_occ + j, y_occ + i 
                if 0 <= imx < MAP_SIZE and 0 <= imy < MAP_SIZE:
                    inflated_map_grid[imy][imx] = 2 

    for px, py in pillar_cells:
        if 0 <= px < MAP_SIZE and 0 <= py < MAP_SIZE:
            inflated_map_grid[py][px] = 2

    current_time = robot.getTime()
    TEMPORARY_OBSTACLES = [obs for obs in TEMPORARY_OBSTACLES if current_time - obs[2] < TEMP_OBSTACLE_DURATION]
    rad_cells = int(TEMP_OBSTACLE_RADIUS * SCALE)
    for ox, oy, _ in TEMPORARY_OBSTACLES:
        omx, omy = world_to_map(ox, oy)
        for i in range(-rad_cells, rad_cells + 1):
            for j in range(-rad_cells, rad_cells + 1):
                if i**2 + j**2 <= rad_cells**2:
                     imx, imy = omx + j, omy + i
                     if 0 <= imx < MAP_SIZE and 0 <= imy < MAP_SIZE:
                         inflated_map_grid[imy][imx] = 2 
                         
    position = rosbot_node.getPosition()
    robot_mx, robot_my = world_to_map(position[0], position[1])
    if 0 <= robot_mx < MAP_SIZE and 0 <= robot_my < MAP_SIZE:
        inflated_map_grid[robot_my][robot_mx] = 1

def update_display(robot_x, robot_y, robot_theta):
    pygame.event.pump()
    safe_grid = np.maximum(0, map_grid) 
    gray_array = ((1.0 - safe_grid.T) * 255).astype(np.uint8)
    rgb_array = np.dstack((gray_array, gray_array, gray_array))
    
    red_indices = np.where(map_grid.T == RED_WALL_GRID_VALUE)
    rgb_array[red_indices[0], red_indices[1], :] = [255, 0, 0]
    
    poison_indices = np.where(map_grid.T == POISON_GRID_VALUE)
    rgb_array[poison_indices[0], poison_indices[1], :] = COLOR_POISON

    map_surface = pygame.surfarray.make_surface(rgb_array)
    map_surface = pygame.transform.flip(map_surface, False, True)
    screen.blit(map_surface, (0, 0))
    
    rmx, rmy = world_to_map(robot_x, robot_y)
    sx, sy = map_to_screen(rmx, rmy)
    pygame.draw.circle(screen, COLOR_ROBOT, (sx, sy), 4)
    end_x = sx + 10 * math.cos(-robot_theta)
    end_y = sy + 10 * math.sin(-robot_theta)
    pygame.draw.line(screen, COLOR_ROBOT, (sx, sy), (end_x, end_y), 2)
    
    if planned_path_for_viz and len(planned_path_for_viz) > 1:
        points = []
        for mx, my in planned_path_for_viz:
            points.append(map_to_screen(mx, my))
        pygame.draw.lines(screen, COLOR_PATH, False, points, 2)
        
    if selected_frontier_for_viz:
        fmx, fmy = selected_frontier_for_viz
        fsx, fsy = map_to_screen(fmx, fmy)
        pygame.draw.circle(screen, COLOR_FRONTIER, (fsx, fsy), 5)

    if solution_path_for_viz and len(solution_path_for_viz) > 1:
        spoints = []
        for mx, my in solution_path_for_viz:
            spoints.append(map_to_screen(mx, my))
        pygame.draw.lines(screen, COLOR_SOLUTION, False, spoints, 4)
        
    for pillar in detected_pillars:
        pmx, pmy = world_to_map(pillar['x'], pillar['y'])
        psx, psy = map_to_screen(pmx, pmy)
        col = COLOR_PILLAR_BLUE if pillar['color'] == 'blue' else COLOR_PILLAR_YELLOW
        pygame.draw.circle(screen, col, (psx, psy), 6)
        
    for ox, oy, _ in TEMPORARY_OBSTACLES:
        omx, omy = world_to_map(ox, oy)
        osx, osy = map_to_screen(omx, omy)
        pygame.draw.circle(screen, COLOR_TEMP_OBSTACLE, (osx, osy), 5) 

    pygame.display.flip()

def find_frontiers(robot_mx, robot_my):
    frontiers = []
    visited = np.zeros_like(map_grid, dtype=bool)
    search_radius_cells = int(FRONTIER_SEARCH_RADIUS_METERS * SCALE)
    min_x_search = max(0, robot_mx - search_radius_cells)
    max_x_search = min(MAP_SIZE, robot_mx + search_radius_cells)
    min_y_search = max(0, robot_my - search_radius_cells)
    max_y_search = min(MAP_SIZE, robot_my + search_radius_cells)
    neighbors_dx = [-1, 1, 0, 0]
    neighbors_dy = [0, 0, -1, 1]

    for y in range(min_y_search, max_y_search):
        for x in range(min_x_search, max_x_search):
            val = map_grid[y][x]
            if val > 0.45 or val == RED_WALL_GRID_VALUE or val == POISON_GRID_VALUE or visited[y][x]: 
                continue
            is_frontier = False
            for dy_nb in [-1, 0, 1]:
                for dx_nb in [-1, 0, 1]:
                    if dx_nb == 0 and dy_nb == 0: continue
                    nx, ny = x + dx_nb, y + dy_nb
                    if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                        if 0.45 <= map_grid[ny][nx] <= 0.55:
                            is_frontier = True; break
                if is_frontier: break
            if is_frontier:
                current_cluster = []
                q = collections.deque([(x, y)])
                visited[y][x] = True
                while q:
                    cx, cy = q.popleft()
                    current_cluster.append((cx, cy))
                    for i in range(4):
                        nx = cx + neighbors_dx[i]
                        ny = cy + neighbors_dy[i]
                        if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE) or visited[ny][nx]: continue
                        if map_grid[ny][nx] < 0.45 and map_grid[ny][nx] != RED_WALL_GRID_VALUE and map_grid[ny][nx] != POISON_GRID_VALUE:
                            for dy_nb2 in [-1, 0, 1]:
                                for dx_nb2 in [-1, 0, 1]:
                                    if dx_nb2 == 0 and dy_nb2 == 0: continue
                                    nnx, nny = nx + dx_nb2, ny + dy_nb2
                                    if 0 <= nnx < MAP_SIZE and 0 <= nny < MAP_SIZE:
                                        if 0.45 <= map_grid[nny][nnx] <= 0.55:
                                            visited[ny][nx] = True
                                            q.append((nx, ny))
                                            break
                if current_cluster and len(current_cluster) >= MIN_FRONTIER_CLUSTER_SIZE:
                    frontiers.append(current_cluster)
    frontiers.sort(key=len, reverse=True)
    return frontiers[:MAX_FRONTIER_CLUSTERS_TO_CONSIDER]

def select_target_frontier(robot_mx, robot_my, robot_theta, frontier_clusters):
    global BAD_DIRECTION_MEMORY
    if not frontier_clusters: return None
    best_target, best_score = None, float('inf')
    
    # --- VISUAL HOMING: Get blue bearing ---
    blue_bearing, blue_size = get_visual_bearing(camera_rgb.getImage(), 'blue')
    visual_bias_angle = None
    
    if blue_bearing is not None:
        visual_bias_angle = robot_theta + blue_bearing
        print(f"VISUAL HOMING ACTIVE: Blue detected at relative angle {blue_bearing:.2f}")

    avoid_angle = None
    if BAD_DIRECTION_MEMORY:
        if robot.getTime() - BAD_DIRECTION_MEMORY['time'] < BAD_DIRECTION_DURATION:
            avoid_angle = BAD_DIRECTION_MEMORY['heading']
        else:
            BAD_DIRECTION_MEMORY = None

    for cluster in frontier_clusters:
        sum_x = sum(p[0] for p in cluster)
        sum_y = sum(p[1] for p in cluster)
        centroid_mx, centroid_my = int(sum_x / len(cluster)), int(sum_y / len(cluster))
        cx, cy = map_to_world(centroid_mx, centroid_my)
        
        # Hazard Avoidance
        if avoid_angle is not None:
            rx, ry = map_to_world(robot_mx, robot_my)
            dx = cx - rx
            dy = cy - ry
            angle_to_frontier = math.atan2(dy, dx)
            diff = angle_to_frontier - avoid_angle
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            if abs(diff) < (BAD_DIRECTION_CONE / 2):
                continue

        if is_frontier_near_dead_end(cx, cy): continue
        if is_frontier_surrounded_by_red_walls(cx, cy): continue

        if (0 <= centroid_mx < MAP_SIZE and 0 <= centroid_my < MAP_SIZE and 
            inflated_map_grid[centroid_my][centroid_mx] != 2):
            target = (centroid_mx, centroid_my)
        else:
            target = None
            min_d = float('inf')
            for fx, fy in cluster:
                if inflated_map_grid[fy][fx] != 2:
                    d = (fx - robot_mx)**2 + (fy - robot_my)**2
                    if d < min_d: min_d = d; target = (fx, fy)
        
        if target:
            if target in UNREACHABLE_FRONTIER_MEMORY:
                if robot.getTime() - UNREACHABLE_FRONTIER_MEMORY[target] < MEMORY_DURATION: continue
                
            dist = math.sqrt((target[0] - robot_mx)**2 + (target[1] - robot_my)**2)
            
            # --- SCORING ---
            score = dist - (FRONTIER_SIZE_WEIGHT * len(cluster))
            
            # Apply Visual Homing Bonus
            if visual_bias_angle is not None:
                rx, ry = map_to_world(robot_mx, robot_my)
                tx, ty = map_to_world(target[0], target[1])
                angle_to_target = math.atan2(ty - ry, tx - rx)
                
                ang_diff = abs(angle_to_target - visual_bias_angle)
                if ang_diff > math.pi: ang_diff = 2*math.pi - ang_diff
                
                if ang_diff < 0.5: # Approx 30 degrees
                    score -= 50.0 # Massive bonus to prioritize this direction
                    
            if score < best_score: best_score = score; best_target = target
    return best_target

def find_nearest_free_cell(target_mx, target_my):
    q = collections.deque([(target_mx, target_my)])
    visited = set([(target_mx, target_my)])
    max_search = 40 
    
    while q:
        cx, cy = q.popleft()
        if abs(cx - target_mx) > max_search or abs(cy - target_my) > max_search: continue
        if 0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE:
            if inflated_map_grid[cy][cx] != 2 and map_grid[cy][cx] < 0.45: return cx, cy
        
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited: visited.add((nx, ny)); q.append((nx, ny))
    return None

def find_random_open_point():
    for _ in range(50):
        mx = random.randint(0, MAP_SIZE-1)
        my = random.randint(0, MAP_SIZE-1)
        if map_grid[my][mx] < 0.2 and inflated_map_grid[my][mx] != 2:
            pos = rosbot_node.getPosition()
            start_mx, start_my = world_to_map(pos[0], pos[1])
            path = a_star_search(start_mx, start_my, mx, my)
            if path:
                print(f"Random Search Target Found at ({mx}, {my})")
                return path
    return []

def is_path_safe_and_known(path):
    if not path: return False
    for mx, my in path:
        if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
            if map_grid[my][mx] > 0.45 and map_grid[my][mx] != RED_WALL_GRID_VALUE: return False
            if inflated_map_grid[my][mx] == 2: return False
    return True

def a_star_search(start_mx, start_my, goal_mx, goal_my):
    if not (0 <= goal_mx < MAP_SIZE and 0 <= goal_my < MAP_SIZE): return []
    if inflated_map_grid[goal_my][goal_mx] == 2: return []

    if inflated_map_grid[start_my][start_mx] == 2:
        q = collections.deque([(start_mx, start_my)])
        visited = set()
        found_start = None
        while q:
            cx, cy = q.popleft()
            if inflated_map_grid[cy][cx] != 2:
                found_start = (cx, cy)
                break
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                 nx, ny = cx+dx, cy+dy
                 if 0<=nx<MAP_SIZE and 0<=ny<MAP_SIZE and (nx,ny) not in visited:
                     visited.add((nx,ny))
                     q.append((nx,ny))
            if len(visited) > 100: break 
        if found_start:
            start_mx, start_my = found_start
        else:
            return []

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    open_set = []
    heapq.heappush(open_set, (0, (start_mx, start_my)))
    g_score = {(start_mx, start_my): 0}
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (goal_mx, goal_my):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append((start_mx, start_my))
            return path[::-1]

        cx, cy = current
        for i in range(4):
            nx, ny = cx + dx[i], cy + dy[i]
            if not (0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE): continue
            cell_val = inflated_map_grid[ny][nx]
            if cell_val == 2: continue 
            step_cost = COST_MOVE_CAUTION if cell_val == 3 else COST_MOVE_SAFE
            tentative_g = g_score[current] + step_cost
            if tentative_g < g_score.get((nx, ny), float('inf')):
                came_from[(nx, ny)] = current
                g_score[(nx, ny)] = tentative_g
                h = math.sqrt((nx - goal_mx)**2 + (ny - goal_my)**2)
                heapq.heappush(open_set, (tentative_g + h, (nx, ny)))
    return []

def smooth_path(path):
    if len(path) < 3: return path
    smoothed_path = [path[0]]
    index = 0
    while index < len(path) - 1:
        next_index = index + 1 
        for i in range(len(path)-1, index, -1):
            p1 = path[index]
            p2 = path[i]
            is_safe = True
            line = bresenham_line(p1[0], p1[1], p2[0], p2[1])
            for lx, ly in line:
                 cell_val = inflated_map_grid[ly][lx]
                 if cell_val == 2: 
                     is_safe = False
                     break
            if is_safe:
                next_index = i
                break
        smoothed_path.append(path[next_index])
        index = next_index
    return smoothed_path

def calculate_angle_to_point(robot_x, robot_y, robot_theta, target_x, target_y):
    dx = target_x - robot_x
    dy = target_y - robot_y
    angle = math.atan2(dy, dx) - robot_theta
    return math.atan2(math.sin(angle), math.cos(angle))

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    return colorsys.rgb_to_hsv(r, g, b)

def is_color_in_range(hsv, lower, upper):
    return (lower[0] <= hsv[0] <= upper[0] and 
            lower[1] <= hsv[1] <= upper[1] and 
            lower[2] <= hsv[2] <= upper[2])

# --- FIXED GEOMETRY + LOCKING + ADAPTIVE SIZING ---
def detect_and_map_poison_shape(rx, ry, rt):
    global mission_pillars, map_grid
    
    if current_robot_state == STATE_FINAL_RUN: return

    image = camera_rgb.getImage()
    if not image: return
    
    img_np = np.frombuffer(image, np.uint8).reshape((camera_height, camera_width, 4))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_img, np.array([40, 100, 50]), np.array([80, 255, 255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 2000: continue 
        
        floor_points = []
        for point in cnt:
            u, v = point[0]
            if random.random() > 0.1: continue 
            res = project_pixel_to_world(u, v, rx, ry, rt)
            if res: floor_points.append(res)
            
        if len(floor_points) < 10: continue
        
        pts_np = np.array(floor_points, dtype=np.float32)
        rect = cv2.minAreaRect(pts_np) 
        
        center, size, angle = rect
        cx, cy = center
        w, h = size
        
        # --- LOCKING LOGIC: Check duplicates first ---
        is_new = True
        for p in mission_pillars['poison_list']:
            dist = math.sqrt((cx - p['x'])**2 + (cy - p['y'])**2)
            if dist < POISON_SPATIAL_DEBOUNCE_METERS:
                is_new = False
                break
        
        if not is_new: continue # Skip if already mapped
        
        # --- SAFETY SIZING (Adaptive but safe) ---
        final_w = max(min(w, 2.5), 0.5)
        final_h = max(min(h, 2.5), 0.5)
        
        # Draw ONCE
        box = cv2.boxPoints(((cx, cy), (final_w, final_h), angle))
        box_map = []
        for bx, by in box:
            bmx, bmy = world_to_map(bx, by)
            box_map.append([bmx, bmy])
            
        pts_draw = np.array(box_map, np.int32).reshape((-1, 1, 2))
        mask_layer = np.zeros_like(map_grid, dtype=np.uint8)
        cv2.fillPoly(mask_layer, [pts_draw], 255)
        
        map_grid[mask_layer == 255] = POISON_GRID_VALUE
        
        mission_pillars['poison_list'].append({'x': cx, 'y': cy})
        print(f"LOCKED POISON ZONE at ({cx:.2f}, {cy:.2f}) - Size: {final_w:.1f}x{final_h:.1f}")

def detect_and_mark_features(camera_image, robot_x, robot_y, robot_theta, is_rotating=False):
    global mission_pillars, detected_pillars
    
    img_np = np.frombuffer(camera_image, np.uint8).reshape((camera_height, camera_width, 4))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # --- ROBUST RED WALL DETECTION (Wide Center 70%) ---
    mask1 = cv2.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    h, w = red_mask.shape
    start_col = int(w * 0.15)
    end_col = int(w * 0.85)
    center_region = red_mask[:, start_col:end_col]
    
    red_ratio = np.count_nonzero(center_region) / center_region.size
    
    if red_ratio > RED_WALL_PIXEL_THRESHOLD:
        print(f"!!! RED WALL DETECTED (Ratio: {red_ratio:.2f}) - AVOID + MARK DEAD END !!!")
        mark_nearby_frontiers_as_dead_end(robot_x, robot_y)
        
        wx = robot_x + 1.0 * math.cos(robot_theta)
        wy = robot_y + 1.0 * math.sin(robot_theta)
        mx, my = world_to_map(wx, wy)
        block_rad = 3 
        for dy in range(-block_rad, block_rad+1):
            for dx in range(-block_rad, block_rad+1):
                if 0 <= mx+dx < MAP_SIZE and 0 <= my+dy < MAP_SIZE:
                    map_grid[my+dy][mx+dx] = RED_WALL_GRID_VALUE
        
        return 'red_wall_trigger', 0.0

    # --- Pillar Colors ---
    blue_pixels, yellow_pixels = [], []
    step = 5
    for x in range(0, camera_width, step):
        for y in range(0, camera_height, step):
            r = Camera.imageGetRed(camera_image, camera_width, x, y)
            g = Camera.imageGetGreen(camera_image, camera_width, x, y)
            b = Camera.imageGetBlue(camera_image, camera_width, x, y)
            hsv = rgb_to_hsv(r, g, b)
            if is_color_in_range(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER): blue_pixels.append((x,y))
            elif is_color_in_range(hsv, YELLOW_HSV_LOWER, YELLOW_HSV_UPPER): yellow_pixels.append((x,y))

    detected_color, pixels = None, []
    if len(blue_pixels) > len(yellow_pixels) and len(blue_pixels) > 20: detected_color, pixels = 'blue', blue_pixels
    elif len(yellow_pixels) > len(blue_pixels) and len(yellow_pixels) > 20: detected_color, pixels = 'yellow', yellow_pixels
    else: return None, 0.0

    current_best_dist = mission_pillars[detected_color]['min_dist']
    base_pixel = max(pixels, key=lambda p: p[1])
    angle_down = ((base_pixel[1] - camera_height/2) / (camera_height/2)) * (camera_fov_vertical/2)
    
    if angle_down <= 0.01: return None, 0.0
    cam_dist = CAMERA_HEIGHT_METERS / math.tan(angle_down)

    if cam_dist > PILLAR_MAX_DETECTION_RANGE: return None, 0.0

    if mission_pillars[detected_color]['found']:
        if cam_dist >= (current_best_dist - 0.20):
             return None, 0.0

    avg_x = sum(p[0] for p in pixels) / len(pixels)
    angle_h = (0.5 - (avg_x / camera_width)) * camera_fov
    lidar_pts = lidar.getPointCloud()

    if abs(angle_h) > CENTERING_TOLERANCE_RAD and not mission_pillars[detected_color]['found']:
        return 'centering', angle_h 

    candidates = []
    for i, p in enumerate(lidar_pts):
        if not (math.isfinite(p.x) and math.isfinite(p.y)): continue
        if abs(math.atan2(p.y, p.x) - angle_h) < 0.035: candidates.append(i)
    
    if not candidates: return None, 0.0
    
    best_idx = min(candidates, key=lambda i: math.sqrt(lidar_pts[i].x**2 + lidar_pts[i].y**2))
    lidar_dist = math.sqrt(lidar_pts[best_idx].x**2 + lidar_pts[best_idx].y**2)

    if abs(cam_dist - lidar_dist) > DISTANCE_DISCREPANCY_THRESHOLD: return None, 0.0
    
    num_pts = len(lidar_pts)
    left_idx, right_idx = best_idx, best_idx
    DEPTH_JUMP_THRESHOLD = 0.2
    
    for _ in range(int(num_pts / 8)): 
        prev = (left_idx - 1) % num_pts
        d_curr, d_prev = math.sqrt(lidar_pts[left_idx].x**2 + lidar_pts[left_idx].y**2), math.sqrt(lidar_pts[prev].x**2 + lidar_pts[prev].y**2)
        if abs(d_curr - d_prev) > DEPTH_JUMP_THRESHOLD: break 
        left_idx = prev
    
    for _ in range(int(num_pts / 8)):
        nxt = (right_idx + 1) % num_pts
        d_curr, d_nxt = math.sqrt(lidar_pts[right_idx].x**2 + lidar_pts[right_idx].y**2), math.sqrt(lidar_pts[nxt].x**2 + lidar_pts[nxt].y**2)
        if abs(d_curr - d_nxt) > DEPTH_JUMP_THRESHOLD: break 
        right_idx = nxt

    p_l, p_r = lidar_pts[left_idx], lidar_pts[right_idx]
    obj_width = math.sqrt((p_l.x - p_r.x)**2 + (p_l.y - p_r.y)**2)
    
    if obj_width > MAX_PILLAR_WIDTH: return None, 0.0

    feat_angle = robot_theta + angle_h
    fx = robot_x + (lidar_dist + PILLAR_RADIUS_METERS) * math.cos(feat_angle)
    fy = robot_y + (lidar_dist + PILLAR_RADIUS_METERS) * math.sin(feat_angle)
    mx, my = world_to_map(fx, fy)
    
    print(f"CONFIRMED {detected_color.upper()} at ({fx:.2f}, {fy:.2f}) [Dist: {lidar_dist:.2f}]")
    
    if mission_pillars[detected_color]['found']:
        old_x, old_y = mission_pillars[detected_color]['x'], mission_pillars[detected_color]['y']
        old_mx, old_my = world_to_map(old_x, old_y)
        rad = int(PILLAR_RADIUS_METERS * SCALE)
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if dx**2+dy**2 <= rad**2 and 0 <= old_mx+dx < MAP_SIZE and 0 <= old_my+dy < MAP_SIZE:
                    map_grid[old_my+dy][old_mx+dx] = PROB_MIN_CONFIDENCE

    mission_pillars[detected_color]['x'] = fx
    mission_pillars[detected_color]['y'] = fy
    mission_pillars[detected_color]['min_dist'] = lidar_dist
    mission_pillars[detected_color]['found'] = True
    
    detected_pillars[:] = [p for p in detected_pillars if p['color'] != detected_color]
    detected_pillars.append({'color': detected_color, 'x': fx, 'y': fy})
    
    rad = int(PILLAR_RADIUS_METERS * SCALE)
    if mapping_enabled:
        for dy in range(-rad, rad+1):
            for dx in range(-rad, rad+1):
                if dx**2+dy**2 <= rad**2 and 0 <= mx+dx < MAP_SIZE and 0 <= my+dy < MAP_SIZE:
                    map_grid[my+dy][mx+dx] = PROB_MAX_CONFIDENCE + 0.2 

    return detected_color, lidar_dist

def get_path_to_pillar_ring_robust(start_mx, start_my, px, py):
    for r_mult in [1.0, 1.5, 2.0]: 
        ring_rad = PILLAR_RING_RADIUS_METERS * r_mult
        candidates = []
        step_rad = (2 * math.pi) / 16
        for i in range(16):
            angle = i * step_rad
            wx = px + ring_rad * math.cos(angle)
            wy = py + ring_rad * math.sin(angle)
            mx, my = world_to_map(wx, wy)
            
            if (mx, my) in UNREACHABLE_FRONTIER_MEMORY:
                if robot.getTime() - UNREACHABLE_FRONTIER_MEMORY[(mx, my)] < MEMORY_DURATION: continue

            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                 cell_val = inflated_map_grid[my][mx]
                 if cell_val != 2 and map_grid[my][mx] < 0.2: 
                     candidates.append((mx, my))
                     
        candidates.sort(key=lambda p: (p[0]-start_mx)**2 + (p[1]-start_my)**2)
        for tgt in candidates:
            path = a_star_search(start_mx, start_my, tgt[0], tgt[1])
            if path: return path

    print(f"Standard rings unreachable for pillar. Attempting fallback...")
    pillar_mx, pillar_my = world_to_map(px, py)
    q = collections.deque([(pillar_mx, pillar_my)])
    visited = set([(pillar_mx, pillar_my)])
    search_limit_cells = int(2.0 * SCALE)

    while q:
        cx, cy = q.popleft()
        if not (0 <= cx < MAP_SIZE and 0 <= cy < MAP_SIZE): continue
        if max(abs(cx - pillar_mx), abs(cy - pillar_my)) > search_limit_cells: continue

        if inflated_map_grid[cy][cx] != 2 and map_grid[cy][cx] != RED_WALL_GRID_VALUE and map_grid[cy][cx] < 0.2:
            path = a_star_search(start_mx, start_my, cx, cy)
            if path:
                print(f"Fallback path found.")
                return path

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return []

# --- Main Loop ---
print("System Initialized: Slow 360 Spin -> Explore (Visual Homing) -> Speed Run.")

motors_stop()
for _ in range(20): robot.step(TIME_STEP)

position, rotation = get_robot_global_pose()
robot_x, robot_y, robot_theta = position[0], position[1], rotation_to_yaw(rotation)
robot_mx, robot_my = world_to_map(robot_x, robot_y)
map_grid[robot_my][robot_mx] = PROB_MIN_CONFIDENCE
last_theta_global = robot_theta

frontier_timer = 0
plot_timer = 0

while robot.step(TIME_STEP) != -1:
    position, rotation = get_robot_global_pose()
    robot_x, robot_y, robot_theta = position[0], position[1], rotation_to_yaw(rotation)
    current_robot_mx, current_robot_my = world_to_map(robot_x, robot_y)
    
    delta_theta = abs(robot_theta - last_theta_global)
    if delta_theta > math.pi: delta_theta = 2*math.pi - delta_theta
    is_rotating_fast = delta_theta > 0.005 
    
    # --- MAP AND DETECT FEATURES ---
    if not is_rotating_fast:
        update_map_lidar(robot_x, robot_y, robot_theta)
        
    update_inflated_map()
    last_theta_global = robot_theta
    
    cam_img = camera_rgb.getImage()
    
    # Run detections even during spin to catch pillars early
    feature, feature_dist = detect_and_mark_features(cam_img, robot_x, robot_y, robot_theta, is_rotating_fast)
    
    if not is_rotating_fast:
        detect_and_map_poison_shape(robot_x, robot_y, robot_theta)
    
    if plot_timer % PLOT_UPDATE_RATE == 0:
        update_display(robot_x, robot_y, robot_theta)
    plot_timer += 1
    
    # --- Mission Control Checks ---
    if mission_pillars['blue']['found'] and mission_pillars['yellow']['found'] and current_robot_state < STATE_RETURNING_TO_START:
        
        # Don't interrupt spin or critical states
        if current_robot_state not in [STATE_INITIAL_SPIN, STATE_CENTERING_ON_PILLAR, STATE_STUCK_RECOVERY, STATE_AVOIDING_HAZARD, STATE_AVOIDING_RED_WALL]:
            
            bx, by = mission_pillars['blue']['x'], mission_pillars['blue']['y']
            yx, yy = mission_pillars['yellow']['x'], mission_pillars['yellow']['y']
            
            path_to_start = get_path_to_pillar_ring_robust(current_robot_mx, current_robot_my, bx, by)
            
            if path_to_start:
                dist_to_blue = math.sqrt((robot_x - bx)**2 + (robot_y - by)**2)
                
                if dist_to_blue > 1.0:
                    print("Both Pillars Found. Repositioning to START (Blue Pillar)...")
                    current_path = smooth_path(path_to_start)
                    planned_path_for_viz = current_path
                    selected_frontier_for_viz = None 
                    current_robot_state = STATE_RETURNING_TO_START
                    path_waypoint_index = 1
                    current_target_type = 'pillar'
                    current_pillar_target_coords = (bx, by)
                else:
                    print("Both Pillars Found. Robot is at Start. Initiating SPEED RUN!")
                    current_robot_state = STATE_FINAL_RUN
                    mapping_enabled = False 
                    raw_run = get_path_to_pillar_ring_robust(current_robot_mx, current_robot_my, yx, yy)
                    current_path = smooth_path(raw_run)
                    planned_path_for_viz = current_path
                    selected_frontier_for_viz = None 
                    solution_path_for_viz = current_path
                    path_waypoint_index = 1
                    final_run_start_time = robot.getTime()
                    final_run_active = True
    
    # --- State Machine ---
    
    # 1. INITIAL SPIN STATE
    if current_robot_state == STATE_INITIAL_SPIN:
        total_accumulated_spin += delta_theta
        
        if total_accumulated_spin >= 2 * math.pi:
            print("Initial 360 Scan Complete. Starting Exploration.")
            current_robot_state = STATE_EXPLORING
            motors_stop()
        else:
            motors_set_velocity(INITIAL_SCAN_SPEED, -INITIAL_SCAN_SPEED)
        continue # Skip other states

    if feature == 'red_wall_trigger' and current_robot_state != STATE_AVOIDING_RED_WALL:
        motors_stop()
        current_robot_state = STATE_AVOIDING_RED_WALL
        stuck_timer_start = robot.getTime()
        target_avoidance_heading = robot_theta + RED_WALL_TURN_ANGLE
        if target_avoidance_heading > math.pi: target_avoidance_heading -= 2*math.pi
        elif target_avoidance_heading < -math.pi: target_avoidance_heading += 2*math.pi
    
    if feature == 'centering' and current_robot_state < STATE_RETURNING_TO_START:
        if current_robot_state not in [STATE_CENTERING_ON_PILLAR, STATE_AVOIDING_RED_WALL]:
            current_robot_state = STATE_CENTERING_ON_PILLAR
    
    if current_robot_state == STATE_EXPLORING:
        motors_stop()
        selected_frontier_for_viz = None
        
        if mission_pillars['blue']['found'] and mission_pillars['yellow']['found']:
             robot.step(TIME_STEP)
             continue

        if 0 <= current_robot_mx < MAP_SIZE and 0 <= current_robot_my < MAP_SIZE:
             val = map_grid[current_robot_my][current_robot_mx]
             if val != RED_WALL_GRID_VALUE and val != POISON_GRID_VALUE:
                map_grid[current_robot_my][current_robot_mx] = PROB_MIN_CONFIDENCE

        if frontier_timer % 5 == 0:
            clusters = find_frontiers(current_robot_mx, current_robot_my)
            target = None 
            
            if clusters:
                target = select_target_frontier(current_robot_mx, current_robot_my, robot_theta, clusters)
                selected_frontier_for_viz = target 
            
            if target:
                raw_path = a_star_search(current_robot_mx, current_robot_my, target[0], target[1])
                current_path = smooth_path(raw_path) 
                planned_path_for_viz = current_path
                if current_path and len(current_path) > 1:
                    path_waypoint_index = 1
                    current_robot_state = STATE_FOLLOWING_PATH
                    stuck_timer_start = robot.getTime() 
                    last_stuck_check_time = robot.getTime()
                    last_stuck_check_pos = (robot_x, robot_y)
                    current_target_type = 'frontier'
                    current_pillar_target_coords = None
                else:
                    UNREACHABLE_FRONTIER_MEMORY[target] = robot.getTime()
            
            else:
                # --- VISUAL FALLBACK: See Blue? Drive to it! ---
                blue_ang, blue_sz = get_visual_bearing(camera_rgb.getImage(), 'blue')
                if blue_ang is not None and not mission_pillars['blue']['found']:
                    print("No frontiers but BLUE visible! Driving towards visual target...")
                    # Project point 2m ahead in that direction
                    tx = robot_x + 2.0 * math.cos(robot_theta + blue_ang)
                    ty = robot_y + 2.0 * math.sin(robot_theta + blue_ang)
                    tmx, tmy = world_to_map(tx, ty)
                    
                    # Plan path to that "ghost" target
                    if 0 <= tmx < MAP_SIZE and 0 <= tmy < MAP_SIZE:
                        # Find nearest free cell to that point to avoid hitting walls
                        valid_t = find_nearest_free_cell(tmx, tmy)
                        if valid_t:
                            new_path = a_star_search(current_robot_mx, current_robot_my, valid_t[0], valid_t[1])
                            if new_path:
                                current_path = smooth_path(new_path)
                                path_waypoint_index = 1
                                current_robot_state = STATE_FOLLOWING_PATH
                                stuck_timer_start = robot.getTime()
                                last_stuck_check_time = robot.getTime()
                                last_stuck_check_pos = (robot_x, robot_y)
                                print("Visual Homing Path Generated!")
                                # Skip the rest of this loop
                                continue
                
                # --- STANDARD LOST MODE ---
                if not (mission_pillars['blue']['found'] and mission_pillars['yellow']['found']):
                    print("No Valid Frontiers & Pillars Missing. LOST MODE ACTIVE.")
                    new_path = find_random_open_point()
                    if new_path:
                        current_path = smooth_path(new_path)
                        path_waypoint_index = 1
                        current_robot_state = STATE_FOLLOWING_PATH
                        is_random_searching = True 
                        stuck_timer_start = robot.getTime() 
                        last_stuck_check_time = robot.getTime()
                        last_stuck_check_pos = (robot_x, robot_y)
                    else:
                        motors_set_velocity(-AUTONOMOUS_TURN_SPEED, AUTONOMOUS_TURN_SPEED)
                        robot.step(TIME_STEP * 20)
                        motors_stop()
                else:
                    motors_set_velocity(-AUTONOMOUS_TURN_SPEED, AUTONOMOUS_TURN_SPEED)
                    robot.step(TIME_STEP * 20)
                    motors_stop()
        frontier_timer += 1

    elif current_robot_state in [STATE_FOLLOWING_PATH, STATE_RETURNING_TO_START, STATE_FINAL_RUN]:
        
        if current_target_type == 'pillar' and current_pillar_target_coords:
            target_color = 'blue' if current_robot_state == STATE_RETURNING_TO_START else 'yellow'
            if current_robot_state == STATE_FINAL_RUN: target_color = 'yellow'
            
            latest_x = mission_pillars[target_color]['x']
            latest_y = mission_pillars[target_color]['y']
            old_x, old_y = current_pillar_target_coords
            
            drift = math.sqrt((latest_x - old_x)**2 + (latest_y - old_y)**2)
            
            if drift > 0.2: 
                print(f"Target Pillar ({target_color.upper()}) Refined! Updating Path...")
                current_pillar_target_coords = (latest_x, latest_y)
                new_raw_path = get_path_to_pillar_ring_robust(current_robot_mx, current_robot_my, latest_x, latest_y)
                if new_raw_path:
                    current_path = smooth_path(new_raw_path)
                    path_waypoint_index = 1
                    planned_path_for_viz = current_path
        
        path_blocked = False
        lookahead = min(path_waypoint_index + 50, len(current_path))
        
        if current_robot_state != STATE_FINAL_RUN:
            for i in range(path_waypoint_index, lookahead):
                wx, wy = current_path[i]
                if inflated_map_grid[wy][wx] == 2: 
                    path_blocked = True
                    break
            
            if path_blocked:
                print("Path Blocked. Replanning...")
                motors_stop()
                new_path_raw = []
                
                if current_target_type == 'pillar' and current_pillar_target_coords:
                    print(" >> Re-scanning Pillar Ring for new safe approach point...")
                    px, py = current_pillar_target_coords
                    new_path_raw = get_path_to_pillar_ring_robust(current_robot_mx, current_robot_my, px, py)
                else:
                    print(" >> Attempting local replan to same frontier...")
                    target_mx, target_my = current_path[-1]
                    new_path_raw = a_star_search(current_robot_mx, current_robot_my, target_mx, target_my)

                if new_path_raw and is_path_safe_and_known(new_path_raw):
                     print("Replan successful.")
                     current_path = smooth_path(new_path_raw)
                     path_waypoint_index = 1 
                     planned_path_for_viz = current_path
                else:
                     print("Target unreachable. Aborting.")
                     current_path = []
                     current_robot_state = STATE_EXPLORING 
                continue 
            
        if check_surround_collision():
             print("CRITICAL: Collision detected! Recovery.")
             motors_stop()
             current_robot_state = STATE_STUCK_RECOVERY
             stuck_timer_start = robot.getTime()
             target_avoidance_heading = robot_theta + math.pi
             if target_avoidance_heading > math.pi: target_avoidance_heading -= 2*math.pi
             elif target_avoidance_heading < -math.pi: target_avoidance_heading += 2*math.pi
             
             TEMPORARY_OBSTACLES.append((robot_x + 0.5*math.cos(robot_theta), robot_y + 0.5*math.sin(robot_theta), robot.getTime()))
             continue

        if not current_path or path_waypoint_index >= len(current_path):
            motors_stop()
            
            if is_random_searching:
                print("Reached Random Search Point. Scanning Area...")
                is_random_searching = False
                current_robot_state = STATE_LOST_SEARCH_ROTATING
                stuck_timer_start = robot.getTime()
                target_avoidance_heading = robot_theta + (2 * math.pi * 0.95)
                continue
            
            if current_robot_state == STATE_RETURNING_TO_START:
                 print("Arrived at Blue Pillar (Start Line). Initializing Final Run...")
                 current_robot_state = STATE_FINAL_RUN
                 mapping_enabled = False 
                 yx, yy = mission_pillars['yellow']['x'], mission_pillars['yellow']['y']
                 
                 raw_run = get_path_to_pillar_ring_robust(current_robot_mx, current_robot_my, yx, yy)
                 current_path = smooth_path(raw_run)
                 solution_path_for_viz = current_path
                 planned_path_for_viz = current_path
                 path_waypoint_index = 1
                 final_run_start_time = robot.getTime()
                 final_run_active = True
                 current_target_type = 'pillar'
                 current_pillar_target_coords = (yx, yy)

            elif current_robot_state == STATE_FINAL_RUN:
                 duration = robot.getTime() - final_run_start_time
                 print(f"==========================================")
                 print(f"MISSION COMPLETE! Reached Yellow Pillar.")
                 print(f"FINAL RUN TIME: {duration:.2f} seconds")
                 print(f"==========================================")
                 current_robot_state = STATE_FINISHED
            else:
                current_robot_state = STATE_EXPLORING
            continue

        tx, ty = current_path[path_waypoint_index]
        wtx, wty = map_to_world(tx, ty)
        dist = math.sqrt((wtx-robot_x)**2 + (wty-robot_y)**2)
        angle = calculate_angle_to_point(robot_x, robot_y, robot_theta, wtx, wty)

        if current_robot_state != STATE_FINAL_RUN and robot.getTime() - last_stuck_check_time > STUCK_CHECK_INTERVAL:
            move_dist = math.sqrt((robot_x - last_stuck_check_pos[0])**2 + (robot_y - last_stuck_check_pos[1])**2)
            if move_dist < MIN_MOVEMENT_DISTANCE:
                print("STUCK DETECTED. Recovering...")
                current_robot_state = STATE_STUCK_RECOVERY
                stuck_timer_start = robot.getTime()
                target_avoidance_heading = robot_theta + math.pi
                continue
            last_stuck_check_time = robot.getTime()
            last_stuck_check_pos = (robot_x, robot_y)

        if dist < WAYPOINT_REACH_THRESHOLD_METERS:
            path_waypoint_index += 1
            last_stuck_check_time = robot.getTime()
            last_stuck_check_pos = (robot_x, robot_y)
            continue

        current_speed_setting = FINAL_RUN_SPEED if current_robot_state == STATE_FINAL_RUN else AUTONOMOUS_BASE_SPEED

        if abs(angle) > HEADING_TOLERANCE:
            left_speed = -AUTONOMOUS_TURN_SPEED if angle > 0 else AUTONOMOUS_TURN_SPEED
            right_speed = -left_speed
            motors_set_velocity(left_speed, right_speed)
        else:
            base = current_speed_setting * min(1.0, dist*2)
            corr = TURN_CORRECTION_GAIN * angle
            corr = max(-AUTONOMOUS_TURN_SPEED, min(corr, AUTONOMOUS_TURN_SPEED))
            left_speed = base - corr
            right_speed = base + corr
            motors_set_velocity(left_speed, right_speed)

    elif current_robot_state == STATE_CENTERING_ON_PILLAR:
        selected_frontier_for_viz = None
        target_angle = 0.0
        
        if feature == 'centering':
             target_angle = feature_dist
        elif feature in ['blue', 'yellow']:
             print("Pillar Centered & Mapped! Resuming...")
             motors_stop()
             current_robot_state = STATE_EXPLORING 
             continue
        else:
             current_robot_state = STATE_EXPLORING
             continue

        rot_speed = AUTONOMOUS_TURN_SPEED * 0.8
        if target_angle > 0:
            motors_set_velocity(-rot_speed, rot_speed) 
        else:
            motors_set_velocity(rot_speed, -rot_speed) 

    elif current_robot_state == STATE_LOST_SEARCH_ROTATING:
        diff = target_avoidance_heading - robot_theta
        if diff > math.pi: diff -= 2*math.pi
        elif diff < -math.pi: diff += 2*math.pi
        
        if abs(diff) < 0.1:
            print("Lost Mode Scan Complete. Resuming Exploration.")
            current_robot_state = STATE_EXPLORING
            motors_stop()
        else:
            motors_set_velocity(AUTONOMOUS_TURN_SPEED, -AUTONOMOUS_TURN_SPEED)

    elif current_robot_state == STATE_AVOIDING_RED_WALL:
        elapsed = robot.getTime() - stuck_timer_start
        
        if elapsed < RED_WALL_REVERSE_TIME:
            motors_set_velocity(-AUTONOMOUS_BASE_SPEED * 0.8, -AUTONOMOUS_BASE_SPEED * 0.8)
        else:
            diff = target_avoidance_heading - robot_theta
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            
            if abs(diff) < 0.1: 
                print("Red Wall Avoidance Complete. Resuming Exploration.")
                current_robot_state = STATE_EXPLORING
                motors_stop()
                current_path = []
            else:
                speed = AUTONOMOUS_TURN_SPEED
                if diff > 0:
                    motors_set_velocity(-speed, speed) 
                else:
                    motors_set_velocity(speed, -speed) 

    elif current_robot_state == STATE_STUCK_RECOVERY:
        elapsed = robot.getTime() - stuck_timer_start
        
        # --- PHASE 1: BACK UP ---
        if elapsed < RECOVERY_REVERSE_DURATION:
            motors_set_velocity(-AUTONOMOUS_BASE_SPEED * 0.8, -AUTONOMOUS_BASE_SPEED * 0.8)
        
        # --- PHASE 2: ROTATE ---
        else:
            diff = target_avoidance_heading - robot_theta
            if diff > math.pi: diff -= 2*math.pi
            elif diff < -math.pi: diff += 2*math.pi
            
            if abs(diff) < 0.05: 
                print("Recovery Turn Complete.")
                current_robot_state = STATE_EXPLORING
                motors_stop()
                stuck_timer_start = robot.getTime() 
                current_path = [] 
            else:
                speed = AUTONOMOUS_TURN_SPEED
                if diff > 0:
                    motors_set_velocity(-speed, speed) 
                else:
                    motors_set_velocity(speed, -speed) 

    elif current_robot_state == STATE_AVOIDING_HAZARD:
        diff = target_avoidance_heading - robot_theta
        if diff > math.pi: diff -= 2*math.pi
        elif diff < -math.pi: diff += 2*math.pi
        
        if abs(diff) < 0.05: 
            print("Hazard Avoidance Turn Complete.")
            current_robot_state = STATE_EXPLORING
            motors_stop()
        else:
            speed = AUTONOMOUS_TURN_SPEED
            if diff > 0:
                motors_set_velocity(-speed, speed) 
            else:
                motors_set_velocity(speed, -speed) 
            
    elif current_robot_state == STATE_FINISHED:
        motors_stop()
            
    if keyboard.getKey() == ord('Q'):
        pygame.quit()
        break

motors_stop()
pygame.quit()