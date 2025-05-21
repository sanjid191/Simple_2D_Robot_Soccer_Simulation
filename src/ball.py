import pygame
import math

class Ball:
    """
    Class representing the soccer ball with physics
    """
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
        """
        Update ball position based on velocity and handle collisions with field boundaries
        """
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
        # Left and right boundaries
        if self.x - self.radius < 0:
            self.x = self.radius
            self.velocity_x = -self.velocity_x * 0.8  # Lose some energy in collision
        elif self.x + self.radius > field_width:
            self.x = field_width - self.radius
            self.velocity_x = -self.velocity_x * 0.8
        
        # Top and bottom boundaries
        if self.y - self.radius < 0:
            self.y = self.radius
            self.velocity_y = -self.velocity_y * 0.8
        elif self.y + self.radius > field_height:
            self.y = field_height - self.radius
            self.velocity_y = -self.velocity_y * 0.8
    
    def kick(self, power, direction):
        """
        Apply force to ball in a given direction
        
        Args:
            power: Force magnitude (0-1)
            direction: Angle in radians
        """
        # Scale power to be between 0 and max_speed
        power = min(1.0, max(0.0, power)) * self.max_speed
        
        # Convert angle to velocity components
        self.velocity_x += power * math.cos(direction)
        self.velocity_y += power * math.sin(direction)
        
        # Cap velocity at max_speed
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
    
    def is_moving(self):
        """
        Check if ball is still in motion
        """
        return abs(self.velocity_x) > 0.1 or abs(self.velocity_y) > 0.1
    
    def get_position(self):
        """
        Get current position as a tuple
        """
        return (self.x, self.y)
    
    def draw(self, screen):
        """
        Draw the ball on the screen
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        
        # Draw a pattern on the ball to show rotation
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 1)
        pygame.draw.line(
            screen, 
            (0, 0, 0), 
            (int(self.x - self.radius/2), int(self.y)), 
            (int(self.x + self.radius/2), int(self.y)), 
            1
        )