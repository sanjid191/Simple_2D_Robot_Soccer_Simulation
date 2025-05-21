import pygame

class Field:
    """
    Class representing the soccer field
    """
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
        """
        Check if ball has entered either goal
        
        Returns:
            'blue' if blue team scored, 'red' if red team scored, None otherwise
        """
        ball_pos = ball.get_position()
        ball_x, ball_y = ball_pos
        
        # Check if ball is within vertical bounds of goal
        if self.goal_top <= ball_y <= self.goal_bottom:
            # Check if ball entered blue goal (left side)
            if ball_x - ball.radius <= 0 and ball_x + ball.radius > -self.goal_depth:
                return 'red'  # Red team scored in blue goal
            
            # Check if ball entered red goal (right side)
            if ball_x + ball.radius >= self.width and ball_x - ball.radius < self.width + self.goal_depth:
                return 'blue'  # Blue team scored in red goal
        
        return None
    
    def update_score(self, scoring_team):
        """
        Update score based on which team scored
        """
        if scoring_team == 'blue':
            self.blue_score += 1
        elif scoring_team == 'red':
            self.red_score += 1
    
    def reset_ball(self, ball):
        """
        Reset ball to center of field after a goal
        """
        ball.x = self.width // 2
        ball.y = self.height // 2
        ball.velocity_x = 0
        ball.velocity_y = 0
    
    def draw(self, screen):
        """
        Draw the field on the screen
        """
        # Draw the field
        screen.fill(self.field_color)
        
        # Draw center line
        pygame.draw.line(
            screen, 
            self.line_color, 
            (self.width // 2, 0), 
            (self.width // 2, self.height), 
            2
        )
        
        # Draw center circle
        pygame.draw.circle(
            screen, 
            self.line_color, 
            (self.width // 2, self.height // 2), 
            self.center_circle_radius, 
            2
        )
        
        # Draw center spot
        pygame.draw.circle(
            screen, 
            self.line_color, 
            (self.width // 2, self.height // 2), 
            self.center_spot_radius
        )
        
        # Draw blue goal (left)
        pygame.draw.rect(
            screen,
            self.goal_color,
            pygame.Rect(-self.goal_depth, self.goal_top, self.goal_depth, self.goal_width)
        )
        
        # Draw red goal (right)
        pygame.draw.rect(
            screen,
            self.goal_color,
            pygame.Rect(self.width, self.goal_top, self.goal_depth, self.goal_width)
        )
        
        # Draw field outline
        pygame.draw.rect(
            screen, 
            self.line_color, 
            pygame.Rect(0, 0, self.width, self.height), 
            2
        )
        
        # Draw score
        font = pygame.font.Font(None, 36)
        blue_text = font.render(f"Blue: {self.blue_score}", True, (0, 0, 255))
        red_text = font.render(f"Red: {self.red_score}", True, (255, 0, 0))
        
        screen.blit(blue_text, (20, 20))
        screen.blit(red_text, (self.width - 120, 20))