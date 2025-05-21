import heapq
import numpy as np
import random
import math

class AStar:
    """
    A* pathfinding algorithm implementation for robot soccer
    """
    def __init__(self, field_size, grid_size=10):
        self.field_size = field_size  # (width, height) of the field in pixels
        self.grid_size = grid_size    # Size of each grid cell for pathfinding
        
        # Calculate grid dimensions
        self.grid_width = int(field_size[0] / grid_size)
        self.grid_height = int(field_size[1] / grid_size)
        
        # Create grid representing the field
        self.grid = np.zeros((self.grid_width, self.grid_height))
        
        # Store the last search results for visualization
        self.alternative_paths = []
        self.optimal_path = []
        self.search_space = set()  # Grid cells explored during search
    
    def update_obstacles(self, obstacles):
        """
        Update the grid with obstacles (opponents, other robots)
        
        Args:
            obstacles: List of (x, y, radius) tuples representing obstacles
        """
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
                for j in range(max(0, grid_y - radius_in_cells), min(self.grid_height, grid_y + radius_in_cells + 1)):
                    # Check if this cell is within the obstacle radius
                    if ((i - grid_x) ** 2 + (j - grid_y) ** 2) <= (radius_in_cells ** 2):
                        self.grid[i, j] = 1  # Mark as obstacle
    def get_path(self, start, end):
        """
        Find a path from start to end using A* algorithm
        
        Args:
            start: (x, y) tuple of start position in pixels
            end: (x, y) tuple of end position in pixels
            
        Returns:
            List of (x, y) coordinates forming a path, or empty list if no path found
        """
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
            self.alternative_paths = []
            self.optimal_path = []
            self.search_space = set()
            return []  # No path if start or end is an obstacle
        
        # Reset the visualizations
        self.alternative_paths = []
        self.search_space = set()
        
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
                # Diagonal movement costs √2, orthogonal costs 1
                movement_cost = 1.4 if dx != 0 and dy != 0 else 1
                tentative_g_score = g_score[current] + movement_cost
                
                # If this path to neighbor is better than any previous one, record it
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, (end_x, end_y))
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        self.alternative_paths = []
        self.optimal_path = []
        return []  # No path found
    
    def _generate_alternative_paths(self, start, end, optimal_path):
        """Generate alternative paths to demonstrate A* optimality"""
        self.alternative_paths = []
        
        # Convert pixel coordinates to grid
        start_grid = (int(start[0] / self.grid_size), int(start[1] / self.grid_size))
        end_grid = (int(end[0] / self.grid_size), int(end[1] / self.grid_size))
        
        # Generate 3 alternative paths
        for i in range(3):
            alt_path = self._generate_suboptimal_path(start_grid, end_grid)
            if alt_path and len(alt_path) > 0:
                # Convert to pixel coordinates
                pixel_path = [(p[0] * self.grid_size + self.grid_size/2, 
                               p[1] * self.grid_size + self.grid_size/2) for p in alt_path]
                self.alternative_paths.append(pixel_path)
        
        # If we couldn't generate all 3 alternative paths, try with more variations
        attempts = 0
        while len(self.alternative_paths) < 3 and attempts < 5:
            # Use different detour strategies
            alt_path = self._generate_detour_path(start_grid, end_grid, detour_factor=3 + attempts)
            if alt_path and len(alt_path) > 0:
                # Check that this path is different from existing paths
                pixel_path = [(p[0] * self.grid_size + self.grid_size/2, 
                               p[1] * self.grid_size + self.grid_size/2) for p in alt_path]
                
                # Only add if path is sufficiently different (at least 20% different)
                if not self._is_similar_to_existing_paths(pixel_path):
                    self.alternative_paths.append(pixel_path)
            attempts += 1
    
    def _generate_suboptimal_path(self, start_grid, end_grid):
        """Generate a suboptimal path by adding random detours"""
        if random.random() < 0.3:
            # Straight line with slight variations
            path = []
            current = start_grid
            steps = max(abs(end_grid[0] - start_grid[0]), abs(end_grid[1] - start_grid[1]))
            
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                ideal_x = start_grid[0] + t * (end_grid[0] - start_grid[0])
                ideal_y = start_grid[1] + t * (end_grid[1] - start_grid[1])
                
                # Add some randomness
                x = int(ideal_x + random.uniform(-2, 2))
                y = int(ideal_y + random.uniform(-2, 2))
                
                # Keep within grid
                x = max(0, min(x, self.grid_width - 1))
                y = max(0, min(y, self.grid_height - 1))
                
                # Skip if obstacle
                if self.grid[x, y] == 1:
                    continue
                    
                path.append((x, y))
            
            return path
        else:
            # Path with significant detour
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
                detour_strength = random.randint(3, 8)
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
    
    def _heuristic(self, a, b):
        """
        Calculates the Euclidean distance between points a and b
        """
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    
    def _reconstruct_path(self, came_from, current):
        """
        Reconstruct path from A* result
        """
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
    
    def _is_similar_to_existing_paths(self, new_path):
        """Check if a path is too similar to existing paths"""
        if not self.alternative_paths:
            return False
            
        for existing_path in self.alternative_paths:
            # Skip if lengths are very different
            if abs(len(existing_path) - len(new_path)) > 0.5 * max(len(existing_path), len(new_path)):
                continue
                
            # Check similarity by sampling points and measuring distances
            similarity_count = 0
            sample_count = min(10, len(new_path), len(existing_path))
            
            for i in range(sample_count):
                # Sample points from both paths
                idx_new = int(i * (len(new_path) - 1) / (sample_count - 1)) if sample_count > 1 else 0
                idx_existing = int(i * (len(existing_path) - 1) / (sample_count - 1)) if sample_count > 1 else 0
                
                # Check distance between points
                dist = math.sqrt((new_path[idx_new][0] - existing_path[idx_existing][0])**2 + 
                                 (new_path[idx_new][1] - existing_path[idx_existing][1])**2)
                
                if dist < 2 * self.grid_size:  # Points are close
                    similarity_count += 1
            
            # If more than 70% of points are similar, paths are too similar
            if similarity_count / sample_count > 0.7:
                return True
                
        return False
        
    def _generate_detour_path(self, start_grid, end_grid, detour_factor=3):
        """Generate path with detours around obstacles or along field edges"""
        # Choose a detour strategy:
        # 1. Go along field edges
        # 2. Make zigzag pattern
        # 3. Choose random intermediate points
        
        detour_type = random.randint(0, 2)
        
        if detour_type == 0:  # Field edge detour
            # Decide which edges to follow (top, right, bottom, left)
            edge_x = random.choice([0, self.grid_width - 1])
            edge_y = random.choice([0, self.grid_height - 1])
            
            # Create waypoints
            waypoints = [
                start_grid,
                (edge_x, start_grid[1]),  # Move horizontally to edge
                (edge_x, edge_y),         # Move vertically along edge
                (end_grid[0], edge_y),    # Move horizontally to align with goal
                end_grid
            ]
            
        elif detour_type == 1:  # Zigzag pattern
            # Create a zigzag path with multiple steps
            num_steps = random.randint(3, 6)
            dx = (end_grid[0] - start_grid[0]) / (num_steps + 1)
            waypoints = [start_grid]
            
            for i in range(1, num_steps + 1):
                # Alternate moving vertically
                zigzag_factor = detour_factor * (-1 if i % 2 == 0 else 1)
                waypoints.append((
                    int(start_grid[0] + i * dx),
                    int(start_grid[1] + zigzag_factor)
                ))
            
            waypoints.append(end_grid)
            
        else:  # Random intermediate points
            # Pick 2-4 random intermediate points
            num_points = random.randint(2, 4)
            waypoints = [start_grid]
            
            for _ in range(num_points):
                # Generate random point within field bounds
                rand_x = random.randint(0, self.grid_width - 1)
                rand_y = random.randint(0, self.grid_height - 1)
                
                # Skip if it's an obstacle
                if self.grid[rand_x, rand_y] == 1:
                    continue
                    
                waypoints.append((rand_x, rand_y))
                
            waypoints.append(end_grid)
        
        # Connect all waypoints with straight lines
        path = []
        for i in range(len(waypoints) - 1):
            path.extend(self._connect_points(waypoints[i], waypoints[i+1]))
            
        return path
        
    def _connect_points(self, p1, p2):
        """Connect two points with a straight line, avoiding obstacles"""
        points = []
        steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
        
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            
            # Keep within grid bounds
            x = max(0, min(x, self.grid_width - 1))
            y = max(0, min(y, self.grid_height - 1))
            
            # Skip obstacles
            if self.grid[x, y] == 1:
                continue
                
            points.append((x, y))
            
        return points