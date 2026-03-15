@echo off
Title WEBUI SETUP - %~dp0

call paths.bat

call site.bat

%PYTHON% -m pip freeze > frozen.txt

:check
if %ERRORLEVEL% == 0 (
    goto success
) else (
    goto exit
)


:success
echo.
echo .............
echo ...DONE.....
echo .............
echo.
cmd /k
exit


:exit
echo.
echo Operation failed. Press any key to exit...
pause >nul
exit



