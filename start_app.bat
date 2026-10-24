@echo off
cd /d "%~dp0"
start "AgriPredict AI Server" cmd /k "python backend\app.py"
timeout /t 4 /nobreak >nul 2>nul || ping 127.0.0.1 -n 5 >nul
start http://127.0.0.1:5000