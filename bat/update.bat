@echo off
Title WEBUI SETUP - %~dp0

call paths.bat

call site.bat

:choice
echo.
echo ===============================
echo   WEBUI PYTHON DEPENDENCIES
echo ===============================
echo.
echo [1] Install (no upgrade)
echo [2] Update (upgrade existing)
echo.
set /p choice=Select an option (1 or 2): 

if "%choice%"=="1" goto install
if "%choice%"=="2" goto update

echo Invalid choice.
goto choice

:install
echo.
echo Installing dependencies...
%PYTHON% -m pip install pip setuptools wheel cython
%PYTHON% -m pip install -r requirments.txt --target %ARTHA_ENV_DIR%Lib\site-packages
goto check

:update
echo.
echo Updating dependencies...
%PYTHON% -m pip install --upgrade pip setuptools wheel cython
%PYTHON% -m pip install -U -r requirments.txt --target %ARTHA_ENV_DIR%Lib\site-packages
goto check

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