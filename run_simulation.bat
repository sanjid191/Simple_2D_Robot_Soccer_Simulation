@echo off
echo Running Robot Soccer Simulation...
python -m src.main
if %ERRORLEVEL% NEQ 0 (
    echo Python not found or error occurred.
    echo Trying python3 instead...
    python3 -m src.main
    if %ERRORLEVEL% NEQ 0 (
        echo Error: Could not run the simulation.
        echo Please ensure Python is installed and in your PATH.
        pause
    )
)
pause
