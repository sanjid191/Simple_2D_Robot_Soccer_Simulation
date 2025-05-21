# PowerShell script to run the Robot Soccer Simulation with Python 3.13.0
Write-Host "Robot Soccer Simulation - Python 3.13.0 Runner" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host

# Check Python version
Write-Host "Checking Python version..."
$pythonVersion = python --version
Write-Host "Detected: $pythonVersion" -ForegroundColor Cyan

# Check for required packages
Write-Host "`nChecking for required packages..."
try {
    python -c "import pygame; print(f'pygame {pygame.__version__} is installed')"
}
catch {
    Write-Host "pygame is not installed. Installing now..." -ForegroundColor Yellow
    python -m pip install pygame
}

try {
    python -c "import numpy; print(f'numpy {numpy.__version__} is installed')"
}
catch {
    Write-Host "numpy is not installed. Installing now..." -ForegroundColor Yellow
    python -m pip install numpy
}

# Try running the simulation
Write-Host "`nStarting Robot Soccer Simulation..." -ForegroundColor Green
Write-Host "If the window doesn't appear, check the console for error messages`n" -ForegroundColor Yellow

# Try running using various methods until one works
$methods = @(
    @{Name = "Direct module"; Command = "python -m src.main" },
    @{Name = "Run script"; Command = "python run.py" },
    @{Name = "Main directly"; Command = "cd src; python main.py; cd .." },
    @{Name = "Simplified version"; Command = "python simple_run.py" }
)

$success = $false
foreach ($method in $methods) {
    if (-not $success) {
        Write-Host "Trying method: $($method.Name)..." -ForegroundColor Cyan
        try {
            Invoke-Expression $method.Command
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Success! The simulation ran using $($method.Name)" -ForegroundColor Green
                $success = $true
                break
            }
        }
        catch {
            Write-Host "Method failed: $($method.Name)" -ForegroundColor Red
            Write-Host "Error: $_" -ForegroundColor Red
        }
        Write-Host
    }
}

if (-not $success) {
    Write-Host "All methods failed. Please check the error messages above." -ForegroundColor Red
    Write-Host "Try running the simplified version manually: python simple_run.py" -ForegroundColor Yellow
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
