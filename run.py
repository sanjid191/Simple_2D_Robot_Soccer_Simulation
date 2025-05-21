"""
Run script for Robot Soccer Simulation.
This script provides an alternative way to run the simulation if there are issues with import paths.
"""
import os
import sys
import traceback

def main():
    print("=== Robot Soccer Simulation ===")
    print("Troubleshooting mode: Attempting to run the simulation...")
    
    # Check Python version
    py_version = sys.version.split()[0]
    print(f"Python version: {py_version}")
    
    # Get the directory where this file is located
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Add the parent directory to the Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        # Try importing pygame and numpy first to check if they're installed
        print("Checking for required libraries...")
        try:
            import pygame
            print("✓ pygame is installed")
        except ImportError:
            print("✗ pygame is not installed. Please install it with: pip install pygame")
            return False
            
        try:
            import numpy as np
            print("✓ numpy is installed")
        except ImportError:
            print("✗ numpy is not installed. Please install it with: pip install numpy")
            return False
            
        # Now try to import our modules
        print("\nAttempting to import simulation modules...")
        try:
            from src.field import Field
            from src.ball import Ball
            from src.robot import Robot
            from src.pathfinding import AStar
            print("✓ All modules imported successfully")
        except ImportError as e:
            print(f"✗ Error importing modules: {e}")
            print("\nTrying alternative import method...")
            
            # Try with relative imports instead
            try:
                import src.main
                print("✓ Imported using package structure")
                return True
            except ImportError as e:
                print(f"✗ Alternative import failed: {e}")
                print("\nTrying to run main.py directly...")
                
                # Last resort - try running main.py directly
                try:
                    os.chdir(os.path.join(current_dir, 'src'))
                    import main
                    print("✓ Ran main.py directly")
                    return True
                except Exception as e:
                    print(f"✗ Direct execution failed: {e}")
                    return False
        
        # If we got here, all imports worked, so we can run the main module
        print("\nStarting the simulation...")
        import src.main
        return True
        
    except Exception as e:
        print(f"\n✗ Error running simulation: {e}")
        print("\nDetailed error traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n===== Troubleshooting Tips =====")
        print("1. Make sure Python 3.6+ is installed and in your PATH")
        print("2. Install required packages: pip install pygame numpy")
        print("3. Try running the simplified version: python simple_run.py")
        print("4. If all else fails, try installing Python from python.org")
    
    print("\nPress Enter to exit...")
    input()
