@echo off
:: whisp-carrier launcher
:: Calls whisp_carrier.py via Python with all passed arguments

setlocal
set "SCRIPT_DIR=%~dp0"

:: Prefer an explicit interpreter, but fall back to whatever is on PATH so the
:: launcher keeps working on other machines and after a Python reinstall.
set "PYTHON=%WHISP_CARRIER_PYTHON%"
if not defined PYTHON set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not exist "%PYTHON%" set "PYTHON=py -3.11"
if "%PYTHON%"=="py -3.11" (
    where py >nul 2>nul || set "PYTHON=python"
)

%PYTHON% "%SCRIPT_DIR%whisp_carrier.py" %*
endlocal & exit /b %ERRORLEVEL%
