# PowerShell script to run the Robot Soccer Simulation
Write-Host "Running Robot Soccer Simulation..." -ForegroundColor Green

# Try to run with python command
try {
    python -m src.main
}
catch {
    Write-Host "Python command failed. Trying python3..." -ForegroundColor Yellow
    
    # Try with python3 command
    try {
        python3 -m src.main
    }
    catch {
        Write-Host "Python3 command failed. Trying with run.py..." -ForegroundColor Yellow
        
        # Try with the alternative run.py file
        try {
            python run.py
        }
        catch {
            Write-Host "Error: Could not run the simulation with any method." -ForegroundColor Red
            Write-Host "Please ensure Python is installed and in your PATH." -ForegroundColor Red
            Write-Host "You can also try running 'python -m pip install -r requirements.txt' to install dependencies." -ForegroundColor Yellow
        }
    }
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
