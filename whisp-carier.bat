@echo off
:: whisp-carier launcher
:: Calls whisp_carier.py via Python with all passed arguments

set "SCRIPT_DIR=%~dp0"
set "PYTHON=C:\Users\Owner1\AppData\Local\Programs\Python\Python311\python.exe"
"%PYTHON%" "%SCRIPT_DIR%whisp_carier.py" %*
exit /b %ERRORLEVEL%
