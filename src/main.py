import pygame
import sys
import math
import random

# Use regular imports when modules are in the same directory
from field import Field
from ball import Ball
from robot import Robot
from pathfinding import AStar

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Robot Soccer Simulation")

# Set up clock
clock = pygame.time.Clock()
FPS = 60

# Create field
field = Field(WIDTH, HEIGHT)

# Create ball
ball = Ball(WIDTH // 2, HEIGHT // 2)

# Create pathfinder
pathfinder = AStar((WIDTH, HEIGHT), grid_size=20)

# Create teams
blue_team = []
red_team = []

# Create blue team robots (left side)
for i in range(3):
    x = random.randint(50, WIDTH // 2 - 50)
    y = random.randint(50, HEIGHT - 50)
    robot = Robot(x, y, team='blue', id=i)
    blue_team.append(robot)

# Create red team robots (right side)
for i in range(3):
    x = random.randint(WIDTH // 2 + 50, WIDTH - 50)
    y = random.randint(50, HEIGHT - 50)
    robot = Robot(x, y, team='red', id=i)
    red_team.append(robot)

# Active robot (controlled by AI)
active_robot = blue_team[0]
active_robot.color = (100, 100, 255)  # Highlight active robot

# Game state
game_state = "playing"  # "playing", "goal", "reset"
goal_timer = 0

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            # Change active robot with number keys
            if event.key == pygame.K_1 and len(blue_team) >= 1:
                active_robot.color = (0, 0, 255)  # Reset previous active robot color
                active_robot = blue_team[0]
                active_robot.color = (100, 100, 255)  # Highlight new active robot
            elif event.key == pygame.K_2 and len(blue_team) >= 2:
                active_robot.color = (0, 0, 255)
                active_robot = blue_team[1]
                active_robot.color = (100, 100, 255)
            elif event.key == pygame.K_3 and len(blue_team) >= 3:
                active_robot.color = (0, 0, 255)
                active_robot = blue_team[2]
                active_robot.color = (100, 100, 255)
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
                    active_robot.move_to_ball(ball, pathfinder, obstacles)
            
            # Reset
            elif event.key == pygame.K_r:
                field.reset_ball(ball)
                
                # Reset blue team
                for i, robot in enumerate(blue_team):
                    robot.x = random.randint(50, WIDTH // 2 - 50)
                    robot.y = random.randint(50, HEIGHT - 50)
                    robot.has_ball = False
                    robot.state = 'idle'
                
                # Reset red team
                for i, robot in enumerate(red_team):
                    robot.x = random.randint(WIDTH // 2 + 50, WIDTH - 50)
                    robot.y = random.randint(50, HEIGHT - 50)
                    robot.has_ball = False
                    robot.state = 'idle'
    
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
        "1-3: Select robot",
        "B: Move to ball",
        "A: Make AI decision",
        "SPACE: Kick ball",
        "R: Reset game"
    ]
    
    for i, line in enumerate(instructions):
        text = font.render(line, True, (255, 255, 255))
        screen.blit(text, (WIDTH - 150, HEIGHT - 140 + i * 20))
    
    # Update display
    pygame.display.flip()
    
    # Cap framerate
    clock.tick(FPS)

# Clean up
pygame.quit()
sys.exit()