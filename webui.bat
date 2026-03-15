@echo off

Title WEBUI - %~dp0

cd %~dp0\bat\

call paths.bat

call site.bat

cd %ARTHA_HOME_DIR%

%PYTHON% -s app.py

cmd /k

:exit
echo/
echo Press a button to exit...
pause >nul
exit




