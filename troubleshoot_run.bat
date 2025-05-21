@echo off
echo Robot Soccer Simulation - Troubleshooting Runner
echo ================================================
echo.

REM Try Python command
echo Trying 'python src\main.py'...
python src\main.py
if %ERRORLEVEL% EQU 0 goto :END

REM Try Python3 command
echo.
echo Trying 'python3 src\main.py'...
python3 src\main.py
if %ERRORLEVEL% EQU 0 goto :END

REM Try py command
echo.
echo Trying 'py src\main.py'...
py src\main.py
if %ERRORLEVEL% EQU 0 goto :END

REM Try simplified version
echo.
echo Trying 'python simple_run.py'...
python simple_run.py
if %ERRORLEVEL% EQU 0 goto :END

echo.
echo Python could not be found or there were errors running the simulation.
echo.
echo Troubleshooting tips:
echo 1. Make sure Python is installed and in your PATH
echo 2. Try installing Python from the Microsoft Store or python.org
echo 3. Install required packages: pip install pygame numpy
echo.

:END
pause
