@echo off
:: ============================================================
:: run_all_scenarios.bat
:: One-click launcher for all traffic optimization experiments.
::
:: USAGE:
::   Double-click this file, OR run from cmd:
::
::   run_all_scenarios.bat                   -- run all scenarios, both methods
::   run_all_scenarios.bat --only 1A 2C      -- run specific scenarios
::   run_all_scenarios.bat --skip 4D 1E      -- skip specific scenarios
::   run_all_scenarios.bat --methods pso     -- run only PSO
::   run_all_scenarios.bat --list            -- list all scenarios and exit
::
:: Results will be saved to:  experiment_results\<scenario_name>\<pso|aco>\
:: ============================================================

:: --- Configuration ---
:: If python is not on your PATH, replace "python" with the full path,
:: e.g.: set PYTHON=C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe
set PYTHON=.venv\Scripts\python.exe

:: Root folder where all scenario archives will be stored
set ARCHIVE_DIR=experiment_results

:: The results folder that main.py writes to (must match OUTPUT_DIR in config.py)
set RESULTS_DIR=results

:: Path to config.py (usually same folder as this .bat)
set CONFIG=config.py

:: ---------------------------------------------------------------
:: Print header
echo.
echo ============================================================
echo   Traffic Signal Optimization - Scenario Runner
echo   %DATE% %TIME%
echo ============================================================
echo.

:: ---------------------------------------------------------------
:: Run the Python orchestrator, forwarding any arguments passed
:: to this .bat file (e.g. --only 1A --methods pso)
%PYTHON% run_scenarios.py ^
    --config %CONFIG% ^
    --results-dir %RESULTS_DIR% ^
    --archive-dir %ARCHIVE_DIR% ^
    --python %PYTHON% ^
    %*

:: ---------------------------------------------------------------
:: Check for errors
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] run_scenarios.py exited with error code %ERRORLEVEL%
    echo Check the output above for details.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ============================================================
echo   Done! Results saved to: %ARCHIVE_DIR%\
echo ============================================================
echo.

:: Keep window open so you can read the final summary
pause
