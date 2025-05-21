"""
Simplified version of the Robot Soccer Simulation that doesn't rely on module imports.
This file combines all the necessary components into a single file for ease of execution.
"""

import sys
import math
import random
import time

# Check Python version
py_version = sys.version.split()[0]
print(f"Python version: {py_version}")
if py_version.startswith("3.13"):
    print("✓ Python 3.13.0 detected - compatible version")
else:
    print(f"Note: Running with Python {py_version}. This simulation was tested with Python 3.13.0")
    print("If you encounter any issues, consider switching to Python 3.13.0")

try:
    import pygame
    import numpy as np
    print(f"✓ pygame {pygame.__version__} is installed")
    print(f"✓ numpy {np.__version__} is installed")
except ImportError:
    print("Error: This program requires pygame and numpy.")
    print("Please install them with: pip install pygame numpy")
    sys.exit(1)

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Soccer Simulation (Simplified)")

# Set up clock
clock = pygame.time.Clock()
FPS = 60

#########################################
# Ball Class
#########################################
class Ball:
    """Class representing the soccer ball with physics"""
    def __init__(self, x, y, radius=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = (255, 255, 255)  # White ball
        
        # Physics properties
        self.velocity_x = 0
        self.velocity_y = 0
        self.deceleration = 0.96  # Ball slows down over time (friction)
        self.max_speed = 15
    
    def update(self, field_width, field_height):
        """Update ball position based on velocity and handle collisions"""
        # Update position based on velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Apply deceleration
        self.velocity_x *= self.deceleration
        self.velocity_y *= self.deceleration
        
        # Stop ball if velocity is very small
        if abs(self.velocity_x) < 0.1:
            self.velocity_x = 0
        if abs(self.velocity_y) < 0.1:
            self.velocity_y = 0
        
        # Handle collisions with field boundaries
        if self.x - self.radius < 0:
            self.x = self.radius
            self.velocity_x = -self.velocity_x * 0.8
        elif self.x + self.radius > field_width:
            self.x = field_width - self.radius
            self.velocity_x = -self.velocity_x * 0.8
        
        if self.y - self.radius < 0:
            self.y = self.radius
            self.velocity_y = -self.velocity_y * 0.8
        elif self.y + self.radius > field_height:
            self.y = field_height - self.radius
            self.velocity_y = -self.velocity_y * 0.8
    
    def kick(self, power, direction):
        """Apply force to ball in a given direction"""
        power = min(1.0, max(0.0, power)) * self.max_speed
        self.velocity_x += power * math.cos(direction)
        self.velocity_y += power * math.sin(direction)
        
        # Cap velocity at max_speed
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
    
    def is_moving(self):
        """Check if ball is still in motion"""
        return abs(self.velocity_x) > 0.1 or abs(self.velocity_y) > 0.1
    
    def get_position(self):
        """Get current position as a tuple"""
        return (self.x, self.y)
    
    def draw(self, screen):
        """Draw the ball on the screen"""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 1)
        pygame.draw.line(screen, (0, 0, 0), (int(self.x - self.radius/2), int(self.y)), 
                        (int(self.x + self.radius/2), int(self.y)), 1)

#########################################
# Field Class
#########################################
class Field:
    """Class representing the soccer field"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        # Field colors
        self.field_color = (0, 128, 0)      # Green field
        self.line_color = (255, 255, 255)   # White lines
          # Goal properties
        self.goal_width = 100
        self.goal_depth = 20
        self.goal_color = (200, 200, 200)
        self.blue_goal_highlight_color = (100, 150, 255, 128)  # Blue highlight with transparency
        self.red_goal_highlight_color = (255, 150, 150, 128)   # Red highlight with transparency
        self.goal_post_width = 8  # Width of the goal post
        
        # Calculate goal positions
        self.goal_top = (height - self.goal_width) // 2
        self.goal_bottom = self.goal_top + self.goal_width
        
        # Field markings
        self.center_circle_radius = 70
        self.center_spot_radius = 5
        
        # Scoreboard
        self.blue_score = 0
        self.red_score = 0
    
    def check_goal(self, ball):
        """Check if ball has entered either goal"""
        ball_pos = ball.get_position()
        ball_x, ball_y = ball_pos
        
        if self.goal_top <= ball_y <= self.goal_bottom:
            # Check blue goal (left side)
            if ball_x - ball.radius <= 0 and ball_x + ball.radius > -self.goal_depth:
                return 'red'  # Red team scored
            
            # Check red goal (right side)
            if ball_x + ball.radius >= self.width and ball_x - ball.radius < self.width + self.goal_depth:
                return 'blue'  # Blue team scored
        
        return None
    
    def update_score(self, scoring_team):
        """Update score based on which team scored"""
        if scoring_team == 'blue':
            self.blue_score += 1
        elif scoring_team == 'red':
            self.red_score += 1
    
    def reset_ball(self, ball):
        """Reset ball to center of field after a goal"""
        ball.x = self.width // 2
        ball.y = self.height // 2
        ball.velocity_x = 0
        ball.velocity_y = 0
    
    def draw(self, screen):
        """Draw the field on the screen"""        # Draw the field
        screen.fill(self.field_color)
        
        # Create goal area highlights (semi-transparent)
        # Blue goal area highlight
        blue_goal_area = pygame.Surface((self.width // 5, self.height // 3), pygame.SRCALPHA)
        blue_goal_area.fill(self.blue_goal_highlight_color)
        screen.blit(blue_goal_area, (0, self.height // 3))
        
        # Red goal area highlight
        red_goal_area = pygame.Surface((self.width // 5, self.height // 3), pygame.SRCALPHA)
        red_goal_area.fill(self.red_goal_highlight_color)
        screen.blit(red_goal_area, (self.width - self.width // 5, self.height // 3))
        
        # Draw center line
        pygame.draw.line(screen, self.line_color, (self.width // 2, 0), 
                        (self.width // 2, self.height), 2)
        
        # Draw center circle
        pygame.draw.circle(screen, self.line_color, (self.width // 2, self.height // 2), 
                          self.center_circle_radius, 2)
          # Draw center spot
        pygame.draw.circle(screen, self.line_color, (self.width // 2, self.height // 2), 
                          self.center_spot_radius)
        
        # Draw blue goal (left) with highlighted posts
        # Goal area
        pygame.draw.rect(screen, self.goal_color,
                        pygame.Rect(-self.goal_depth, self.goal_top, self.goal_depth, self.goal_width))
        # Top goal post
        pygame.draw.rect(screen, (0, 0, 200),
                        pygame.Rect(-self.goal_depth, self.goal_top - self.goal_post_width, 
                                   self.goal_depth + self.goal_post_width, self.goal_post_width))
        # Bottom goal post
        pygame.draw.rect(screen, (0, 0, 200),
                        pygame.Rect(-self.goal_depth, self.goal_bottom, 
                                   self.goal_depth + self.goal_post_width, self.goal_post_width))
        # Back post
        pygame.draw.rect(screen, (0, 0, 200),
                        pygame.Rect(-self.goal_depth - self.goal_post_width, self.goal_top - self.goal_post_width, 
                                   self.goal_post_width, self.goal_width + self.goal_post_width * 2))
        
        # Draw red goal (right) with highlighted posts
        # Goal area
        pygame.draw.rect(screen, self.goal_color,
                        pygame.Rect(self.width, self.goal_top, self.goal_depth, self.goal_width))
        # Top goal post
        pygame.draw.rect(screen, (200, 0, 0),
                        pygame.Rect(self.width - self.goal_post_width, self.goal_top - self.goal_post_width, 
                                   self.goal_depth + self.goal_post_width, self.goal_post_width))
        # Bottom goal post
        pygame.draw.rect(screen, (200, 0, 0),
                        pygame.Rect(self.width - self.goal_post_width, self.goal_bottom, 
                                   self.goal_depth + self.goal_post_width, self.goal_post_width))
        # Back post
        pygame.draw.rect(screen, (200, 0, 0),
                        pygame.Rect(self.width + self.goal_depth, self.goal_top - self.goal_post_width, 
                                   self.goal_post_width, self.goal_width + self.goal_post_width * 2))
        
        # Draw field outline
        pygame.draw.rect(screen, self.line_color, pygame.Rect(0, 0, self.width, self.height), 2)
        
        # Draw score
        font = pygame.font.Font(None, 36)
        blue_text = font.render(f"Blue: {self.blue_score}", True, (0, 0, 255))
        red_text = font.render(f"Red: {self.red_score}", True, (255, 0, 0))
        
        screen.blit(blue_text, (20, 20))
        screen.blit(red_text, (self.width - 120, 20))

#########################################
# A* Pathfinding Algorithm
#########################################
class AStar:
    """A* pathfinding algorithm implementation"""
    def __init__(self, field_size, grid_size=20):
        self.field_size = field_size
        self.grid_size = grid_size
        
        # Calculate grid dimensions
        self.grid_width = int(field_size[0] / grid_size)
        self.grid_height = int(field_size[1] / grid_size)
        
        # Create grid representing the field
        self.grid = np.zeros((self.grid_width, self.grid_height))
        
        # For visualization
        self.search_space = set()      # Grid cells explored during search
        self.optimal_path = []         # The optimal path found by A*
        self.alternative_paths = []    # Alternative paths for comparison
    
    def update_obstacles(self, obstacles):
        """Update the grid with obstacles (opponents, other robots)"""
        # Reset grid
        self.grid = np.zeros((self.grid_width, self.grid_height))
        
        # Mark obstacles on the grid
        for x, y, radius in obstacles:
            # Convert from pixel coordinates to grid coordinates
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            
            # Mark cells within the radius of the obstacle
            radius_in_cells = int(radius / self.grid_size) + 1
            
            for i in range(max(0, grid_x - radius_in_cells), min(self.grid_width, grid_x + radius_in_cells + 1)):
                for j in range(max(0, grid_y - radius_in_cells), min(self.grid_height, grid_y + radius_in_cells + 1)):                    # Check if this cell is within the obstacle radius
                    if ((i - grid_x) ** 2 + (j - grid_y) ** 2) <= (radius_in_cells ** 2):
                        self.grid[i, j] = 1  # Mark as obstacle
    
    def get_path(self, start, end):
        """Find a path from start to end using A* algorithm"""
        import heapq
        
        # Convert from pixel coordinates to grid coordinates
        start_x, start_y = int(start[0] / self.grid_size), int(start[1] / self.grid_size)
        end_x, end_y = int(end[0] / self.grid_size), int(end[1] / self.grid_size)
        
        # Ensure coordinates are within grid bounds
        start_x = max(0, min(start_x, self.grid_width - 1))
        start_y = max(0, min(start_y, self.grid_height - 1))
        end_x = max(0, min(end_x, self.grid_width - 1))
        end_y = max(0, min(end_y, self.grid_height - 1))
        
        # Check if start or end is an obstacle
        if self.grid[start_x, start_y] == 1 or self.grid[end_x, end_y] == 1:
            self.search_space = set()
            self.optimal_path = []
            self.alternative_paths = []
            return []  # No path if start or end is an obstacle
        
        # Reset the visualizations
        self.search_space = set()
        self.alternative_paths = []
        
        # Initialize A* algorithm
        open_set = []
        closed_set = set()
        came_from = {}
        
        # g_score[node] is the cost from start to node
        g_score = {(start_x, start_y): 0}
        
        # f_score[node] = g_score[node] + heuristic(node, goal)
        f_score = {(start_x, start_y): self._heuristic((start_x, start_y), (end_x, end_y))}
        
        # Add start node to open set
        heapq.heappush(open_set, (f_score[(start_x, start_y)], (start_x, start_y)))
        
        # Possible movements (8 directions)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # Add to search space for visualization
            self.search_space.add(current)
            
            # If we've reached the goal, reconstruct and return the path
            if current == (end_x, end_y):
                optimal_path = self._reconstruct_path(came_from, current)
                self.optimal_path = optimal_path
                
                # Generate alternative paths
                self._generate_alternative_paths(start, end, optimal_path)
                
                return optimal_path
            
            closed_set.add(current)
            
            # Check all neighbors
            for dx, dy in directions:
                neighbor = current[0] + dx, current[1] + dy
                
                # Skip if out of bounds
                if not (0 <= neighbor[0] < self.grid_width and 0 <= neighbor[1] < self.grid_height):
                    continue
                
                # Skip if obstacle or already evaluated
                if self.grid[neighbor[0], neighbor[1]] == 1 or neighbor in closed_set:
                    continue
                
                # Calculate the tentative g_score
                movement_cost = 1.4 if dx != 0 and dy != 0 else 1
                tentative_g_score = g_score[current] + movement_cost
                
                # If this path to neighbor is better than any previous one, record it
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, (end_x, end_y))
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # No path found
        self.optimal_path = []
        self.alternative_paths = []
        return []
    
    def _heuristic(self, a, b):
        """Calculates the Euclidean distance between points a and b"""
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def _reconstruct_path(self, came_from, current):
        """Reconstruct path from A* result"""
        path = []
        while current in came_from:
            # Convert grid coordinates back to pixel coordinates (center of the cell)
            path.append((
                (current[0] + 0.5) * self.grid_size,
                (current[1] + 0.5) * self.grid_size
            ))
            current = came_from[current]
        
        # Add the start position
        path.append((
            (current[0] + 0.5) * self.grid_size,
            (current[1] + 0.5) * self.grid_size
        ))
        
        # Return the path in reverse (from start to end)
        path.reverse()
        return path

    def _generate_alternative_paths(self, start, end, optimal_path):
        """Generate alternative paths to demonstrate A* optimality"""
        self.alternative_paths = []
        
        # Convert pixel coordinates to grid
        start_grid = (int(start[0] / self.grid_size), int(start[1] / self.grid_size))
        end_grid = (int(end[0] / self.grid_size), int(end[1] / self.grid_size))
        
        # Try to generate 3 alternative paths
        for i in range(3):
            # Use different strategies for each path
            if i == 0:
                # Path with significant detour
                alt_path = self._generate_detour_path(start_grid, end_grid)
            elif i == 1:
                # Zigzag path
                alt_path = self._generate_zigzag_path(start_grid, end_grid)
            else:
                # Path along the edges
                alt_path = self._generate_edge_path(start_grid, end_grid)
                
            if alt_path and len(alt_path) > 0:
                # Convert to pixel coordinates
                pixel_path = [(p[0] * self.grid_size + self.grid_size/2, 
                             p[1] * self.grid_size + self.grid_size/2) for p in alt_path]
                self.alternative_paths.append(pixel_path)

    def _generate_detour_path(self, start_grid, end_grid):
        """Generate path with significant detour"""
        # Choose a random midpoint that's not on the direct path
        mid_x = (start_grid[0] + end_grid[0]) // 2
        mid_y = (start_grid[1] + end_grid[1]) // 2
        
        # Add a detour offset (perpendicular to direct path)
        dx = end_grid[0] - start_grid[0]
        dy = end_grid[1] - start_grid[1]
        
        # Create perpendicular vector
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            perpendicular_x = -dy / length
            perpendicular_y = dx / length
            
            # Random detour strength
            detour_strength = random.randint(3, 6)
            mid_x = int(mid_x + perpendicular_x * detour_strength)
            mid_y = int(mid_y + perpendicular_y * detour_strength)
            
            # Keep within grid
            mid_x = max(0, min(mid_x, self.grid_width - 1))
            mid_y = max(0, min(mid_y, self.grid_height - 1))
        
        # Draw line through midpoint
        path = []
        
        # First half - start to mid
        steps_1 = max(abs(mid_x - start_grid[0]), abs(mid_y - start_grid[1]))
        for i in range(steps_1 + 1):
            t = i / steps_1 if steps_1 > 0 else 0
            x = int(start_grid[0] + t * (mid_x - start_grid[0]))
            y = int(start_grid[1] + t * (mid_y - start_grid[1]))
            
            # Skip if obstacle
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if self.grid[x, y] == 1:
                    continue
                path.append((x, y))
        
        # Second half - mid to end
        steps_2 = max(abs(end_grid[0] - mid_x), abs(end_grid[1] - mid_y))
        for i in range(1, steps_2 + 1):  # Start from 1 to avoid duplicating midpoint
            t = i / steps_2 if steps_2 > 0 else 0
            x = int(mid_x + t * (end_grid[0] - mid_x))
            y = int(mid_y + t * (end_grid[1] - mid_y))
            
            # Skip if obstacle
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if self.grid[x, y] == 1:
                    continue
                path.append((x, y))
        
        return path
        
    def _generate_zigzag_path(self, start_grid, end_grid):
        """Generate path with zigzag pattern"""
        path = []
        
        # Calculate direct vector
        dx = end_grid[0] - start_grid[0]
        dy = end_grid[1] - start_grid[1]
        
        # Number of zigzags
        zigzags = 4
        
        # Calculate each segment
        for i in range(zigzags + 1):
            t1 = i / zigzags
            t2 = (i + 0.5) / zigzags if i < zigzags else 1.0
            
            # Points along direct path
            x1 = int(start_grid[0] + t1 * dx)
            y1 = int(start_grid[1] + t1 * dy)
            
            # Add point along direct path
            if self._is_valid_point(x1, y1):
                path.append((x1, y1))
            
            # Add zigzag point if not at end
            if i < zigzags:
                # Perpendicular offset, alternating sides
                offset = 4 if i % 2 == 0 else -4
                
                if abs(dx) > abs(dy):
                    # Horizontal dominant, so zigzag vertically
                    x2 = int(start_grid[0] + t2 * dx)
                    y2 = int(start_grid[1] + t2 * dy + offset)
                else:
                    # Vertical dominant, so zigzag horizontally
                    x2 = int(start_grid[0] + t2 * dx + offset)
                    y2 = int(start_grid[1] + t2 * dy)
                
                if self._is_valid_point(x2, y2):
                    path.append((x2, y2))
        
        return path
    
    def _generate_edge_path(self, start_grid, end_grid):
        """Generate path that follows field edges"""
        path = []
        
        # Choose which edge to follow
        use_top_edge = random.choice([True, False])
        
        # Start point
        path.append(start_grid)
        
        if use_top_edge:
            # Go to top edge first
            edge_y = 1  # Near top edge
            
            # Add point at top edge
            top_x = start_grid[0]
            if self._is_valid_point(top_x, edge_y):
                path.append((top_x, edge_y))
            
            # Move along top edge toward end's x-coordinate
            steps = abs(end_grid[0] - top_x)
            for i in range(1, steps + 1):
                x = top_x + (1 if end_grid[0] > top_x else -1) * i
                if self._is_valid_point(x, edge_y):
                    path.append((x, edge_y))
            
            # Move from edge to end point
            if self._is_valid_point(end_grid[0], edge_y):
                path.append((end_grid[0], edge_y))
        else:
            # Go to side edge first
            edge_x = 1 if start_grid[0] > self.grid_width // 2 else self.grid_width - 2
            
            # Add point at side edge
            side_y = start_grid[1]
            if self._is_valid_point(edge_x, side_y):
                path.append((edge_x, side_y))
            
            # Move along side edge toward end's y-coordinate
            steps = abs(end_grid[1] - side_y)
            for i in range(1, steps + 1):
                y = side_y + (1 if end_grid[1] > side_y else -1) * i
                if self._is_valid_point(edge_x, y):
                    path.append((edge_x, y))
            
            # Move from edge to end point
            if self._is_valid_point(edge_x, end_grid[1]):
                path.append((edge_x, end_grid[1]))
        
        # End point
        path.append(end_grid)
        
        return path
    
    def _is_valid_point(self, x, y):
        """Check if a grid point is valid (in bounds and not an obstacle)"""
        return (0 <= x < self.grid_width and 
                0 <= y < self.grid_height and 
                self.grid[x, y] != 1)

#########################################
# Robot Class
#########################################
class Robot:
    """Class representing a robot on the soccer field"""
    def __init__(self, x, y, radius=15, team='blue', id=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.team = team
        self.id = id
        
        # Set color based on team
        self.color = (0, 0, 255) if team == 'blue' else (255, 0, 0)  # Blue or Red
        
        # Movement properties
        self.max_speed = 5.0
        self.speed = 0
        self.direction = 0
        self.target_x = x
        self.target_y = y
        
        # Pathfinding
        self.path = []
        self.current_path_index = 0
        
        # State
        self.has_ball = False
        self.state = 'idle'  # idle, moving_to_ball, moving_to_position, with_ball
    
    def update(self, ball, field_width, field_height, obstacles, pathfinder):
        """Update robot's position and state"""
        # Update the path if needed
        if self.state == 'moving_to_ball':
            # If ball moved significantly, recalculate path
            ball_pos = ball.get_position()
            if self._distance((self.target_x, self.target_y), ball_pos) > 20:
                self._set_target(ball_pos[0], ball_pos[1])
                self._calculate_path(pathfinder, obstacles)
        
        # Follow current path if we have one
        if self.path and self.current_path_index < len(self.path):
            # Get next waypoint
            waypoint = self.path[self.current_path_index]
            
            # Move towards waypoint
            dx = waypoint[0] - self.x
            dy = waypoint[1] - self.y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 5:  # Reached waypoint
                self.current_path_index += 1
            else:
                # Calculate direction to waypoint
                self.direction = math.atan2(dy, dx)
                
                # Move in that direction
                self.speed = min(self.max_speed, distance)
                self.x += self.speed * math.cos(self.direction)
                self.y += self.speed * math.sin(self.direction)
        
        # If no path or at end of path
        elif self.state != 'idle':
            # If target is ball, try to catch it
            if self.state == 'moving_to_ball':
                ball_pos = ball.get_position()
                dx = ball_pos[0] - self.x
                dy = ball_pos[1] - self.y
                distance = math.sqrt(dx**2 + dy**2)
                
                if distance < self.radius + ball.radius:
                    # Reached the ball
                    self.has_ball = True
                    self.state = 'with_ball'
                else:
                    # Move directly to ball
                    self.direction = math.atan2(dy, dx)
                    self.speed = min(self.max_speed, distance)
                    self.x += self.speed * math.cos(self.direction)
                    self.y += self.speed * math.sin(self.direction)
        
        # Handle ball possession
        if self.has_ball:
            # Position ball at robot's edge in direction of movement
            ball_offset = self.radius + ball.radius
            ball.x = self.x + ball_offset * math.cos(self.direction)
            ball.y = self.y + ball_offset * math.sin(self.direction)
              # Reset ball velocity
            ball.velocity_x = 0
            ball.velocity_y = 0
        
        # Ensure robot stays within field boundaries
        self.x = max(self.radius, min(field_width - self.radius, self.x))
        self.y = max(self.radius, min(field_height - self.radius, self.y))
    
    def move_to_ball(self, ball, pathfinder, obstacles):
        """Set the robot to move towards the ball"""
        ball_pos = ball.get_position()
        self._set_target(ball_pos[0], ball_pos[1])
        self.state = 'moving_to_ball'
        self._calculate_path(pathfinder, obstacles)
        # Store pathfinder for visualization
        self.pathfinder = pathfinder
    
    def move_to_position(self, x, y, pathfinder, obstacles):
        """Set the robot to move to a specific position"""
        self._set_target(x, y)
        self.state = 'moving_to_position'
        self._calculate_path(pathfinder, obstacles)
        # Store pathfinder for visualization
        self.pathfinder = pathfinder
    
    def kick_ball(self, ball, power, direction=None):
        """Kick the ball if robot has possession"""
        if self.has_ball:
            if direction is None:
                direction = self.direction
            
            ball.kick(power, direction)
            self.has_ball = False
            self.state = 'idle'
    
    def decide_action(self, ball, teammate_positions, opponent_positions, field_width, field_height):
        """Rule-based decision making for robot with ball"""
        if not self.has_ball:
            return 'get_ball', ball.get_position()
        
        # Check if we're in shooting position (close to opponent's goal)
        if self.team == 'blue':
            # Blue team shoots toward right goal
            goal_x = field_width
            distance_to_goal = abs(goal_x - self.x)
            shot_threshold = field_width * 0.3  # Shoot if within 30% of field width to goal
        else:
            # Red team shoots toward left goal
            goal_x = 0
            distance_to_goal = abs(self.x - goal_x)
            shot_threshold = field_width * 0.3
        
        # Calculate goal center Y position
        goal_center_y = field_height / 2
        
        # Check if path to goal is clear
        path_blocked = False
        goal_direction = math.atan2(goal_center_y - self.y, goal_x - self.x)
        
        for opponent_pos in opponent_positions:
            op_x, op_y = opponent_pos
            
            # Check if opponent is between robot and goal
            if self.team == 'blue' and op_x < self.x:
                continue  # Opponent is behind us
            if self.team == 'red' and op_x > self.x:
                continue  # Opponent is behind us
                
            # Calculate distance from opponent to line from robot to goal
            # Using point-line distance formula
            a = math.tan(goal_direction)  # Slope
            b = -1
            c = self.y - a * self.x  # y-intercept
            
            distance = abs(a * op_x + b * op_y + c) / math.sqrt(a**2 + b**2)
            
            if distance < 30:  # If opponent is within 30 pixels of shot line
                path_blocked = True
                break
        
        # If close to goal and path not blocked, shoot
        if distance_to_goal < shot_threshold and not path_blocked:
            return 'shoot', goal_direction
        
        # Find closest teammate for potential pass
        closest_teammate = None
        closest_distance = float('inf')
        
        for teammate_pos in teammate_positions:
            if teammate_pos == (self.x, self.y):
                continue  # Skip self
            
            distance = self._distance((self.x, self.y), teammate_pos)
            
            # Check if path to teammate is clear
            teammate_direction = math.atan2(teammate_pos[1] - self.y, teammate_pos[0] - self.x)
            teammate_path_blocked = False
            
            for opponent_pos in opponent_positions:
                # Similar check as above, but for teammate
                op_x, op_y = opponent_pos
                a = math.tan(teammate_direction)
                b = -1
                c = self.y - a * self.x
                
                distance_to_line = abs(a * op_x + b * op_y + c) / math.sqrt(a**2 + b**2)
                
                if distance_to_line < 20:  # Tighter threshold for passing
                    teammate_path_blocked = True
                    break
            
            if not teammate_path_blocked and distance < closest_distance:
                closest_teammate = teammate_pos
                closest_distance = distance
        
        # If we have an open teammate, pass
        if closest_teammate:
            pass_direction = math.atan2(closest_teammate[1] - self.y, closest_teammate[0] - self.x)
            return 'pass', pass_direction
        
        # Otherwise, dribble toward goal
        return 'dribble', goal_direction
    
    def _set_target(self, x, y):
        """Set target position"""
        self.target_x = x
        self.target_y = y
    
    def _calculate_path(self, pathfinder, obstacles):
        """Calculate path to target using A* pathfinding"""
        # Update obstacles
        pathfinder.update_obstacles(obstacles)
          # Get path
        self.path = pathfinder.get_path((self.x, self.y), (self.target_x, self.target_y))
        self.current_path_index = 0
    
    def _distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def draw(self, screen):
        """Draw the robot on the screen"""
        # Draw alternative paths first if available
        if hasattr(self, 'pathfinder') and self.pathfinder and self.state == 'moving_to_ball':
            # Draw explored cells from search space (softly)
            for cell in self.pathfinder.search_space:
                x = int(cell[0] * self.pathfinder.grid_size + self.pathfinder.grid_size/2)
                y = int(cell[1] * self.pathfinder.grid_size + self.pathfinder.grid_size/2)
                pygame.draw.rect(
                    screen,
                    (50, 50, 50),  # Dark gray
                    pygame.Rect(
                        x - self.pathfinder.grid_size/2,
                        y - self.pathfinder.grid_size/2,
                        self.pathfinder.grid_size,
                        self.pathfinder.grid_size
                    ),
                    1  # Just the outline
                )
            
            # Draw alternative paths
            for i, alt_path in enumerate(self.pathfinder.alternative_paths):
                if not alt_path:
                    continue
                    
                # Different colors for different alternative paths
                alt_colors = [(255, 100, 100), (100, 255, 100), (255, 255, 100)]
                alt_color = alt_colors[i % len(alt_colors)]
                
                # Draw path with dashed lines
                for j in range(len(alt_path) - 1):
                    # Make dashed line effect
                    if j % 3 != 0:  # Skip every third segment
                        pygame.draw.line(
                            screen, 
                            alt_color, 
                            (int(alt_path[j][0]), int(alt_path[j][1])),
                            (int(alt_path[j+1][0]), int(alt_path[j+1][1])),
                            1  # Thin line
                        )
                
                # Add length indicator
                if len(alt_path) > 1:
                    mid_index = len(alt_path) // 2
                    length_text = f"L: {len(alt_path)}"
                    alt_font = pygame.font.Font(None, 16)
                    label = alt_font.render(length_text, True, alt_color)
                    screen.blit(label, (int(alt_path[mid_index][0]), int(alt_path[mid_index][1])))
        
        # Draw robot body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw direction indicator
        end_x = self.x + self.radius * math.cos(self.direction)
        end_y = self.y + self.radius * math.sin(self.direction)
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)
        
        # Draw robot ID
        font = pygame.font.Font(None, 20)
        text = font.render(str(self.id), True, (255, 255, 255))
        screen.blit(text, (int(self.x) - 5, int(self.y) - 8))
        
        # Draw path with enhanced visibility
        if self.path:
            # Choose color based on team and state
            if self.team == 'blue':
                if self.state == 'moving_to_ball':
                    path_color = (0, 255, 255)  # Cyan for blue team moving to ball
                else:
                    path_color = (50, 150, 255)  # Light blue for other movements
            else:
                if self.state == 'moving_to_ball':
                    path_color = (255, 165, 0)  # Orange for red team moving to ball
                else:
                    path_color = (255, 100, 100)  # Light red for other movements
            
            # Draw path with line segments
            for i in range(len(self.path) - 1):
                pygame.draw.line(
                    screen, 
                    path_color, 
                    (int(self.path[i][0]), int(self.path[i][1])),
                    (int(self.path[i+1][0]), int(self.path[i+1][1])),
                    3  # Increased thickness
                )
                
            # Draw dots at waypoints for better visibility
            for point in self.path:
                pygame.draw.circle(
                    screen,
                    path_color,
                    (int(point[0]), int(point[1])),
                    4  # Small dot at each waypoint
                )
                
        # Draw status indicator around active robot
        if self.state == 'with_ball':
            pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius + 5, 2)

#########################################
# Main Game Loop
#########################################
def main():
    # Create field
    field = Field(WIDTH, HEIGHT)

    # Create ball
    ball = Ball(WIDTH // 2, HEIGHT // 2)

    # Create pathfinder
    pathfinder = AStar((WIDTH, HEIGHT), grid_size=20)    # Create teams
    blue_team = []
    red_team = []
    
    # Create blue team robots (left side)
    for i in range(5):  # Increased from 3 to 5 robots
        x = random.randint(50, WIDTH // 2 - 50)
        y = random.randint(50, HEIGHT - 50)
        robot = Robot(x, y, team='blue', id=i)
        blue_team.append(robot)

    # Create red team robots (right side)
    for i in range(5):  # Increased from 3 to 5 robots
        x = random.randint(WIDTH // 2 + 50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        robot = Robot(x, y, team='red', id=i)
        red_team.append(robot)
        
    # Active robot (controlled by player)
    active_robot = blue_team[0]
    active_robot.color = (100, 100, 255)  # Highlight active robot
    
    # Active team (blue or red)
    active_team = 'blue'

    # Game state
    game_state = "playing"  # "playing", "goal", "reset"
    goal_timer = 0    # Main game loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Switch between teams with T key
                if event.key == pygame.K_t:
                    # Reset color of current active robot
                    if active_team == 'blue':
                        active_robot.color = (0, 0, 255)  # Reset to blue
                        active_team = 'red'
                        active_robot = red_team[0]
                        active_robot.color = (255, 100, 100)  # Highlight active red robot
                    else:
                        active_robot.color = (255, 0, 0)  # Reset to red
                        active_team = 'blue'
                        active_robot = blue_team[0]
                        active_robot.color = (100, 100, 255)  # Highlight active blue robot
                
                # Change active robot with number keys (1-5)
                elif event.key == pygame.K_1:
                    if active_team == 'blue':
                        active_robot.color = (0, 0, 255)  # Reset previous active robot color
                        active_robot = blue_team[0]
                        active_robot.color = (100, 100, 255)  # Highlight new active robot
                    else:
                        active_robot.color = (255, 0, 0)
                        active_robot = red_team[0]
                        active_robot.color = (255, 100, 100)
                elif event.key == pygame.K_2:
                    if active_team == 'blue' and len(blue_team) >= 2:
                        active_robot.color = (0, 0, 255)
                        active_robot = blue_team[1]
                        active_robot.color = (100, 100, 255)
                    elif active_team == 'red' and len(red_team) >= 2:
                        active_robot.color = (255, 0, 0)
                        active_robot = red_team[1]
                        active_robot.color = (255, 100, 100)
                elif event.key == pygame.K_3:
                    if active_team == 'blue' and len(blue_team) >= 3:
                        active_robot.color = (0, 0, 255)
                        active_robot = blue_team[2]
                        active_robot.color = (100, 100, 255)
                    elif active_team == 'red' and len(red_team) >= 3:
                        active_robot.color = (255, 0, 0)
                        active_robot = red_team[2]
                        active_robot.color = (255, 100, 100)
                elif event.key == pygame.K_4:
                    if active_team == 'blue' and len(blue_team) >= 4:
                        active_robot.color = (0, 0, 255)
                        active_robot = blue_team[3]
                        active_robot.color = (100, 100, 255)
                    elif active_team == 'red' and len(red_team) >= 4:
                        active_robot.color = (255, 0, 0)
                        active_robot = red_team[3]
                        active_robot.color = (255, 100, 100)
                elif event.key == pygame.K_5:
                    if active_team == 'blue' and len(blue_team) >= 5:
                        active_robot.color = (0, 0, 255)
                        active_robot = blue_team[4]
                        active_robot.color = (100, 100, 255)
                    elif active_team == 'red' and len(red_team) >= 5:
                        active_robot.color = (255, 0, 0)
                        active_robot = red_team[4]
                        active_robot.color = (255, 100, 100)
                
                # Kick the ball
                elif event.key == pygame.K_SPACE:
                    if active_robot.has_ball:
                        active_robot.kick_ball(ball, 0.8)  # Kick with 80% power
                
                # Move to ball
                elif event.key == pygame.K_b:
                    # Create list of obstacles (all robots except active one)
                    obstacles = []
                    for robot in blue_team + red_team:
                        if robot != active_robot:
                            obstacles.append((robot.x, robot.y, robot.radius))
                    
                    active_robot.move_to_ball(ball, pathfinder, obstacles)
                
                # Auto decision
                elif event.key == pygame.K_a:
                    # Collect teammate and opponent positions
                    teammate_positions = []
                    opponent_positions = []
                    
                    for robot in blue_team + red_team:
                        if robot.team == active_robot.team and robot != active_robot:
                            teammate_positions.append((robot.x, robot.y))
                        elif robot.team != active_robot.team:
                            opponent_positions.append((robot.x, robot.y))
                    
                    # Get decision
                    action, target = active_robot.decide_action(
                        ball, 
                        teammate_positions, 
                        opponent_positions, 
                        WIDTH, 
                        HEIGHT
                    )
                    
                    # Execute decision
                    if action == 'shoot':
                        active_robot.kick_ball(ball, 1.0, target)  # Full power shot
                    elif action == 'pass':
                        active_robot.kick_ball(ball, 0.6, target)  # Medium power pass
                    elif action == 'dribble':
                        # Create obstacles
                        obstacles = []
                        for robot in blue_team + red_team:
                            if robot != active_robot:
                                obstacles.append((robot.x, robot.y, robot.radius))
                        
                        # Calculate target position in direction of goal
                        target_x = active_robot.x + 100 * math.cos(target)
                        target_y = active_robot.y + 100 * math.sin(target)
                        
                        # Ensure target is within field
                        target_x = max(active_robot.radius, min(WIDTH - active_robot.radius, target_x))
                        target_y = max(active_robot.radius, min(HEIGHT - active_robot.radius, target_y))
                        
                        active_robot.move_to_position(target_x, target_y, pathfinder, obstacles)
                    elif action == 'get_ball':
                        obstacles = []
                        for robot in blue_team + red_team:
                            if robot != active_robot:
                                obstacles.append((robot.x, robot.y, robot.radius))
                        active_robot.move_to_ball(ball, pathfinder, obstacles)                # Reset
                elif event.key == pygame.K_r:
                    field.reset_ball(ball)
                    
                    # Reset blue team
                    for i, robot in enumerate(blue_team):
                        robot.x = random.randint(50, WIDTH // 2 - 50)
                        robot.y = random.randint(50, HEIGHT - 50)
                        robot.has_ball = False
                        robot.state = 'idle'
                        # Reset color if it was active
                        if robot != active_robot or active_team != 'blue':
                            robot.color = (0, 0, 255)
                    
                    # Reset red team
                    for i, robot in enumerate(red_team):
                        robot.x = random.randint(WIDTH // 2 + 50, WIDTH - 50)
                        robot.y = random.randint(50, HEIGHT - 50)
                        robot.has_ball = False
                        robot.state = 'idle'                        # Reset color if it was active
                        if robot != active_robot or active_team != 'red':
                            robot.color = (255, 0, 0)
        
        # Create list of obstacles (all robots except active one)
        obstacles = []
        for robot in blue_team + red_team:
            if robot != active_robot:
                obstacles.append((robot.x, robot.y, robot.radius))
        
        # Update game state
        if game_state == "playing":
            # Update ball
            ball.update(WIDTH, HEIGHT)
            
            # Update robots
            for robot in blue_team + red_team:
                robot.update(ball, WIDTH, HEIGHT, obstacles, pathfinder)
            
            # Check for goals
            scoring_team = field.check_goal(ball)
            if scoring_team:
                field.update_score(scoring_team)
                game_state = "goal"
                goal_timer = FPS * 2  # 2 seconds before reset
        
        elif game_state == "goal":
            # Wait for timer to expire
            goal_timer -= 1
            if goal_timer <= 0:
                # Reset ball and game state
                field.reset_ball(ball)
                game_state = "playing"
        
        # Draw everything
        field.draw(screen)
        ball.draw(screen)
          # Draw all robots
        for robot in blue_team + red_team:
            robot.draw(screen)
        
        # Display instructions
        font = pygame.font.Font(None, 24)
        instructions = [
            "Keys:",
            "T: Switch team (Blue/Red)",
            "1-5: Select robot",
            "B: Move to ball",
            "A: Make AI decision",
            "SPACE: Kick ball",
            "R: Reset game"
        ]
        
        # Show current team
        team_text = font.render(f"Active Team: {active_team.upper()}", True, 
                               (100, 100, 255) if active_team == 'blue' else (255, 100, 100))
        screen.blit(team_text, (WIDTH - 240, HEIGHT - 180))
        
        for i, line in enumerate(instructions):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (WIDTH - 180, HEIGHT - 150 + i * 20))
        
        # Update display
        pygame.display.flip()
        
        # Cap framerate
        clock.tick(FPS)

    # Clean up
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    print("Starting Robot Soccer Simulation...")
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)  # Keep console window open for a few seconds to see the error
