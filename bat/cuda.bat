@echo off

Title WEBUI INSTALL - %~dp0

call paths.bat

call site.bat

echo/
echo DETECTING TORCH VERSION...
set TORCH_VER=2.8.0
for /f "delims=" %%I in ('%PYTHON% -c "from importlib.metadata import version; print(version('torch'))" 2^>nul') do set TORCH_VER=%%I
echo/
echo DETECTED / RECOMMENDED TORCH VERSION: %TORCH_VER%
echo/

set istorch=
set /p istorch=INSTALL TORCH %TORCH_VER% WITH CUDA?[y/n]: 

if /i not "%istorch%"=="y" goto command

:CUDAMENU
echo/
echo SELECT CUDA VERSION:
echo --------------------
echo [1] CUDA 12.6
echo [2] CUDA 12.8
echo [3] CUDA 13.0
echo --------------------
set /p choice="Enter number [1-3]: "

if "%choice%"=="1" (
	set CUDA_TAG=cu126
) else if "%choice%"=="2" (
	set CUDA_TAG=cu128
) else if "%choice%"=="3" (
	set CUDA_TAG=cu130
) else (
	echo Invalid choice. Please try again.
	goto CUDAMENU
)

echo/
echo ...INSTALLING TORCH %TORCH_VER% WITH CUDA %CUDA_TAG%...
echo/
%PYTHON% -m pip install -U torch==%TORCH_VER% torchaudio==%TORCH_VER% --index-url https://download.pytorch.org/whl/%CUDA_TAG% --target %ARTHA_ENV_DIR%Lib\site-packages
echo/

:command
echo/
echo ..........
echo ...DONE...
echo ..........
echo/
cmd /k

:exit
echo/
echo Install failed. Press a button to exit...
pause >nul
exit