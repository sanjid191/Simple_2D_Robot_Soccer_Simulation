# Robot Soccer Simulation

A 2D Robot Soccer Simulation implementing A\* pathfinding for robots to navigate around obstacles and approach the ball.

## Description

This project simulates a simplified 2D soccer game with robots. Each team has 3 robots, with one robot being actively controlled at a time. The robots use the A\* pathfinding algorithm to navigate around obstacles (other robots) when approaching the ball.

## Features

- A\* pathfinding for robot movement
- Basic physics for ball movement
- Collision detection and handling
- Multiple robots per team
- Simple scoring system
- Interactive controls
- Rule-based decision making for robots

## Requirements

- Python 3.6+
- pygame
- numpy

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Simulation

There are several ways to run the simulation:

1. **Using Python directly**:

   ```
   cd "path_to_project"
   python src/main.py
   ```

2. **Using the batch file** (Windows):

   ```
   run_simulation.bat
   ```

3. **Using the PowerShell script** (Windows):

   ```
   .\run_simulation.ps1
   ```

4. **Using the alternative Python launcher**:

   ```
   python run.py
   ```

5. **Using the simplified single-file version**:

   ```
   python simple_run.py
   ```

6. **Troubleshooting mode** (recommended if you're having issues):
   ```
   troubleshoot_run.bat
   ```

The simplified version (`simple_run.py`) combines all the code into a single file, which can be useful if you're having import issues or want to quickly run the simulation without setting up a full project.

## Controls

- **1, 2, 3**: Select different robots from the blue team
- **B**: Command the active robot to move to the ball
- **A**: AI makes automatic decision (shoot, pass, or dribble)
- **SPACE**: Kick the ball (when the robot has possession)
- **R**: Reset the game

## Project Structure

- `main.py`: Main game loop and event handling
- `field.py`: Soccer field representation and drawing
- `ball.py`: Ball physics and rendering
- `robot.py`: Robot behavior, movement, and A\* integration
- `pathfinding.py`: A\* pathfinding algorithm implementation

### Implementation Details

### A\* Pathfinding

The A\* algorithm is implemented in `pathfinding.py`. It creates a grid representation of the field and finds the shortest path while avoiding obstacles (other robots).

### Robot AI

Robots can be in different states:

- Idle
- Moving to ball
- Moving to position
- With ball

When commanded to approach the ball, robots use A\* to find a path around obstacles.

### Decision Making Algorithm

The simulation includes a rule-based decision-making system that allows robots to:

1. **Shoot**: When the robot is close to the opponent's goal and has a clear shot
2. **Pass**: When the path to the goal is blocked but a teammate is open
3. **Dribble**: When the robot can't shoot or pass effectively

The decision algorithm evaluates:

- Distance to the opponent's goal
- Whether the path to the goal is blocked by opponents
- The positions of teammates and whether they are open for passes

### Physics

Simple physics are implemented for ball movement, including:

- Velocity and deceleration
- Collision with field boundaries
- Kicking mechanics

## Future Improvements

- Add more sophisticated team strategies
- Implement passing between robots
- Add different robot roles (defender, attacker)
- Improve the physics simulation
- Add AI-controlled robots that play automatically

## Troubleshooting

If you're having issues running the simulation, try these steps:

1. **Python not installed or not in PATH**:

   - Install Python 3.6+ from [python.org](https://www.python.org/downloads/) or the Microsoft Store
   - Make sure to check "Add Python to PATH" during installation
   - Restart your computer after installation

2. **Missing dependencies**:

   - Install required libraries: `pip install pygame numpy`
   - Or use: `pip install -r requirements.txt`

3. **Import errors**:

   - Try running with the troubleshooting batch file: `troubleshoot_run.bat`
   - Or use the simplified version: `python simple_run.py`

4. **Other errors**:
   - If pygame is not installing properly, you might need to install Visual C++ Build Tools
   - For more advanced troubleshooting, check the pygame documentation
