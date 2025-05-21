import pygame
import math
# Use regular import when modules are in the same directory
from pathfinding import AStar

class Robot:
    """
    Class representing a robot on the soccer field
    """
    def __init__(self, x, y, radius=15, team='blue', id=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.team = team
        self.id = id  # Robot identifier
        
        # Set color based on team
        self.color = (0, 0, 255) if team == 'blue' else (255, 0, 0)  # Blue or Red
        
        # Movement properties
        self.max_speed = 5.0
        self.speed = 0
        self.direction = 0  # Angle in radians
        self.target_x = x
        self.target_y = y
        
        # Pathfinding
        self.path = []
        self.current_path_index = 0
        
        # State
        self.has_ball = False
        self.state = 'idle'  # idle, moving_to_ball, moving_to_position, with_ball
    
    def update(self, ball, field_width, field_height, obstacles, pathfinder):
        """
        Update robot's position and state
        
        Args:
            ball: The soccer ball object
            field_width, field_height: Dimensions of the field
            obstacles: List of obstacles (other robots)
            pathfinder: AStar pathfinding object
        """
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
        """
        Set the robot to move towards the ball
        """
        ball_pos = ball.get_position()
        self._set_target(ball_pos[0], ball_pos[1])
        self.state = 'moving_to_ball'
        self._calculate_path(pathfinder, obstacles)
    
    def move_to_position(self, x, y, pathfinder, obstacles):
        """
        Set the robot to move to a specific position
        """
        self._set_target(x, y)
        self.state = 'moving_to_position'
        self._calculate_path(pathfinder, obstacles)
      def kick_ball(self, ball, power, direction=None):
        """
        Kick the ball if robot has possession
        
        Args:
            ball: The ball object
            power: Kick power (0-1)
            direction: Kick direction (if None, use robot's current direction)
        """
        if self.has_ball:
            if direction is None:
                direction = self.direction
            
            ball.kick(power, direction)
            self.has_ball = False
            self.state = 'idle'
    
    def decide_action(self, ball, teammate_positions, opponent_positions, field_width, field_height):
        """
        Rule-based decision making for robot with ball
        
        Args:
            ball: The ball object
            teammate_positions: List of (x, y) positions of teammates
            opponent_positions: List of (x, y) positions of opponents
            field_width, field_height: Field dimensions
            
        Returns:
            action: String describing action ('shoot', 'pass', 'dribble')
            target: Target position or direction for action
        """
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
        """
        Set target position
        """
        self.target_x = x
        self.target_y = y
        
    def _calculate_path(self, pathfinder, obstacles):
        """
        Calculate path to target using A* pathfinding
        """
        # Update obstacles
        pathfinder.update_obstacles(obstacles)
        
        # Get path
        self.path = pathfinder.get_path((self.x, self.y), (self.target_x, self.target_y))
        self.current_path_index = 0
    
    def _distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two positions
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
      def draw(self, screen):
        """
        Draw the robot on the screen
        """
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
        
        # Draw path for debugging
        if self.path:
            for i in range(len(self.path) - 1):
                pygame.draw.line(
                    screen, 
                    (100, 100, 100), 
                    (int(self.path[i][0]), int(self.path[i][1])),
                    (int(self.path[i+1][0]), int(self.path[i+1][1])),
                    1
                )
                
        # Draw status indicator around active robot
        if self.state == 'with_ball':
            pygame.draw.circle(screen, (255, 255, 0), (int(self.x), int(self.y)), self.radius + 5, 2)