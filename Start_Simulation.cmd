@echo off
echo Robot Soccer Simulation - Python 3.13 Launcher
echo ==============================================
echo.

echo Checking Python version...
python --version

echo.
echo Installing required packages if needed...
python -m pip install pygame numpy

echo.
echo Starting the simulation...
python simple_run.py

echo.
echo If the simulation didn't start correctly, check the error messages above.
pause
