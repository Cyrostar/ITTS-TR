@echo off

Title WEBUI - %~dp0

cd %~dp0\bat\

call paths.bat

call site.bat

cd %ARTHA_HOME_DIR%

start /b cmd /c "ping localhost -n 10 >nul & start http://127.0.0.1:7860/?__theme=dark"

%PYTHON% -s app.py

cmd /k

:exit
echo/
echo Press a button to exit...
pause >nul
exit




