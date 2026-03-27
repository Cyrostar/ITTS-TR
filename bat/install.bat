@echo off

Title WEBUI INSTALL - %~dp0

call paths.bat

echo/
set isgit=
set /p isgit=INSTALL GITHUB?[y/n]:

if "%isgit%"=="y" (

if not exist %ARTHA_GITHUB_DIR% (
echo/
echo ...CREATING GITHUB DIRECTORY...
echo/
md %ARTHA_GITHUB_DIR%
) else (
echo/
echo ...DELETING EXISTING GITHUB DIRECTORY...
echo/
rmdir /s /q %ARTHA_GITHUB_DIR%
md %ARTHA_GITHUB_DIR%
)

echo ...DOWNLOADING GITHUB...
echo/
curl -L -o %ARTHA_GITHUB_DIR%PortableGit.7z.exe %ARTHA_GITHUB_URL%

echo/
echo ...INSTALLING GITHUB...
echo/
%ARTHA_GITHUB_DIR%PortableGit.7z.exe

echo ...DELETING 7Z.EXE FILE...
echo/
del %ARTHA_GITHUB_DIR%PortableGit.7z.exe

echo ...MOVING GITHUB FILES...
echo/
xcopy %ARTHA_GITHUB_DIR%PortableGit %ARTHA_GITHUB_DIR% /s

echo/
echo ...DELETING TEMPORARY FOLDER...
echo/
rmdir /s /q %ARTHA_GITHUB_DIR%PortableGit
)

echo/
set /p roPython=Enter Python version (Recommended 3.11.9):
echo/

set roPythonDigits=%roPython:.=%
set roPythonDigits=%roPythonDigits:~0,-1%

if not exist %ARTHA_BIN_DIR% (
md %ARTHA_BIN_DIR%
)

if not exist %ARTHA_PYTHON_DIR% (
echo ...CREATING PYTHON DIRECTORY...
echo/
md %ARTHA_PYTHON_DIR%
) else (
echo/
echo ...DELETING EXISTING PYTHON DIRECTORY...
echo/
rmdir /s /q %ARTHA_PYTHON_DIR%
md %ARTHA_PYTHON_DIR%
)

echo ...DOWNLOADING PYTHON...
echo/
curl -o %ARTHA_PYTHON_DIR%python%roPython%.zip %ARTHA_PYTHON_URL%%roPython%/python-%roPython%-embed-amd64.zip

echo/
echo ...EXTRACTING PYTHON...
echo/

tar -xf %ARTHA_PYTHON_DIR%python%roPython%.zip -C %ARTHA_PYTHON_DIR%

echo ...DELETING ZIP FILE...
echo/

del %ARTHA_PYTHON_DIR%python%roPython%.zip

if not exist %ARTHA_PYTHON_DIR%Lib (
echo ...CREATING PYTHON SUB DIRECTORIES...
echo/
md %ARTHA_PYTHON_DIR%Lib
md %ARTHA_PYTHON_DIR%DLLs
tar -xf %ARTHA_PYTHON_DIR%python%roPythonDigits%.zip -C %ARTHA_PYTHON_DIR%Lib
xcopy %ARTHA_PYTHON_DIR%*.dll %ARTHA_PYTHON_DIR%DLLs /y
xcopy %ARTHA_PYTHON_DIR%*.pyd %ARTHA_PYTHON_DIR%DLLs /y
)

echo/
echo ...DOWNLOADING NUGET...
echo/
set ARTHA_NUGET_ZIP=%ARTHA_BIN_DIR%python_nuget.zip
set ARTHA_NUGET_DIR=%ARTHA_BIN_DIR%python_nuget_extracted
curl -L -o "%ARTHA_NUGET_ZIP%" "https://www.nuget.org/api/v2/package/python/%roPython%"

echo/
echo ...EXTRACTING ARCHIVE...
echo/
if not exist "%ARTHA_NUGET_DIR%" md "%ARTHA_NUGET_DIR%"
tar -xf "%ARTHA_NUGET_ZIP%" -C "%ARTHA_NUGET_DIR%"

echo ...COPYING INCLUDE AND LIBS...
echo/
xcopy "%ARTHA_NUGET_DIR%\tools\include" "%ARTHA_PYTHON_DIR%include" /E /I /Y
xcopy "%ARTHA_NUGET_DIR%\tools\libs" "%ARTHA_PYTHON_DIR%libs" /E /I /Y

echo ...CLEANING UP TEMPORARY FILES...
echo/
del "%ARTHA_NUGET_ZIP%"
rmdir /s /q "%ARTHA_NUGET_DIR%"

echo/
echo ...DEFINING PYTHON EXECUTABLE...
echo/
set roPython=%ARTHA_PYTHON_DIR%python.exe

echo ...CONFIGURING PYTHON...
echo/

echo .\Lib>%ARTHA_PYTHON_DIR%python%roPythonDigits%._pth
echo .\Scripts>>%ARTHA_PYTHON_DIR%python%roPythonDigits%._pth
echo .>>%ARTHA_PYTHON_DIR%python%roPythonDigits%._pth
echo import site>>%ARTHA_PYTHON_DIR%python%roPythonDigits%._pth
echo import sys>%ARTHA_PYTHON_DIR%sitecustomize.py

echo import site>%ARTHA_PYTHON_DIR%Lib\sitecustomize.py
echo site.addsitedir(r"%ARTHA_ENV_DIR%Lib\site-packages")>>%ARTHA_PYTHON_DIR%Lib\sitecustomize.py

if not exist %ARTHA_ENV_DIR% (
echo/
echo ...CREATING ENV DIRECTORY...
echo/
md %ARTHA_ENV_DIR%Lib\site-packages
md %ARTHA_ENV_DIR%Scripts
)

echo ...DOWNLOADING PIP...
echo/
curl -o %ARTHA_PYTHON_DIR%get-pip.py %ARTHA_PIP_URL%

echo/
echo ...INSTALLING PIP...
echo/
%PYTHON% %ARTHA_PYTHON_DIR%get-pip.py --no-warn-script-location

echo/
echo ...UPGRADING PIP...
echo/
%PYTHON% -m pip install --upgrade pip wheel cython pybind11 ninja

echo/
echo ...INSTALLING UV...
echo/
%PYTHON% -m pip install uv

echo/
echo ...INSTALLING SETUPTOOLS...
echo/
%PYTHON% -m pip install setuptools==69.5.1

echo/
echo ...PURGING CACHE...
echo/
%PYTHON% -m pip cache purge

echo/
echo ...INSTALLING REQUIREMENTS...
echo/
%PYTHON% -m pip install -r requirments.txt --target %ARTHA_ENV_DIR%Lib\site-packages

echo/
echo DETECTING TORCH VERSION...
set TORCH_VER=2.8.0
for /f "delims=" %%I in ('%PYTHON% -c "from importlib.metadata import version; print(version('torch'))" 2^>nul') do set TORCH_VER=%%I
echo/
echo DETECTED / RECOMMENDED TORCH VERSION: %TORCH_VER%
echo/

set istorch=
set /p istorch=INSTALL TORCH %TORCH_VER% WITH CUDA?[y/n]: 

if /i not "%istorch%"=="y" goto CUDASKIP

:CUDAMENU

echo/
echo SELECT CUDA VERSION:
echo --------------------
echo [1] CUDA 12.6
echo [2] CUDA 12.8 (Recommended)
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
%PYTHON% -m pip install -U torch==%TORCH_VER% torchaudio==%TORCH_VER% "fsspec<=2025.10.0" --index-url https://download.pytorch.org/whl/%CUDA_TAG% --target %ARTHA_ENV_DIR%Lib\site-packages
echo/

:CUDASKIP

echo/
set isffmpeg=
set /p isffmpeg=INSTALL FFMPEG?[y/n]: 

if /i not "%isffmpeg%"=="y" goto MPEGSKIP

echo/
echo SELECT FFMPEG VERSION:
echo [1] Stable (v7.1.1)
echo [2] Latest Release
echo/
    
set /p ffmpeg_choice="Enter choice [1 or 2]: "
	
set "ARTHA_FFMPEG_URL=%ARTHA_FFMPEG_STABLE_URL%"
    
if "%ffmpeg_choice%"=="2" set "ARTHA_FFMPEG_URL=%ARTHA_FFMPEG_LATEST_URL%"

if not exist %ARTHA_FFMPEG_DIR% (
echo/
echo ...CREATING FFMPEG DIRECTORY...
echo/
md %ARTHA_FFMPEG_DIR%
) else (
echo/
echo ...DELETING EXISTING FFMPEG DIRECTORY...
echo/
rmdir /s /q %ARTHA_FFMPEG_DIR%
md %ARTHA_FFMPEG_DIR%
)

echo ...DOWNLOADING FFMPEG...
echo/
curl -L -o %ARTHA_FFMPEG_DIR%ffmpeg.zip %ARTHA_FFMPEG_URL%

echo/
echo ...EXTRACTING FFMPEG...
echo/
	
tar -xf %ARTHA_FFMPEG_DIR%ffmpeg.zip --strip-components=1 -C %ARTHA_FFMPEG_DIR%
	
echo ...DELETING ZIP FILE...
echo/
	
del %ARTHA_FFMPEG_DIR%ffmpeg.zip

:MPEGSKIP

echo/
set isytdip=
set /p isytdip=INSTALL YT-DIP?[y/n]: 

if /i not "%isytdip%"=="y" goto YTDIPSKIP

if not exist %ARTHA_YT_DIP_DIR% (
echo/
echo ...CREATING YT-DIP DIRECTORY...
echo/
md %ARTHA_YT_DIP_DIR%
) else (
echo/
echo ...DELETING EXISTING YT-DIP DIRECTORY...
echo/
rmdir /s /q %ARTHA_YT_DIP_DIR%
md %ARTHA_YT_DIP_DIR%
)

curl -L -o %ARTHA_YT_DIP_DIR%yt-dlp_x86.exe %ARTHA_YT_DIP_URL%

:YTDIPSKIP

echo/
echo ...CREATING NECESSARY FILES...
echo/
xcopy uix %ARTHA_BASE_DIR% /s
xcopy wui %ARTHA_HOME_DIR% /s

echo/
set isitts=
set /p isitts=SPARSE INDEX-TTS REPO?[y/n]: 

if /i not "%isitts%"=="y" goto ITTSSKIP

echo/
echo ...CLONING INDEXTTS FILES...
echo/
set PATH=%PATH%;%ARTHA_GITHUB_DIR%;%ARTHA_GITHUB_DIR%bin;%ARTHA_GITHUB_DIR%cmd

git --version >nul 2>&1
if %errorlevel% neq 0 (

    echo Git not found
	echo IndexTTS clone failed!
	echo/
    goto exit
)

set REPO_URL=https://github.com/index-tts/index-tts.git

cd %ARTHA_HOME_DIR%

git init
git remote add -f origin %REPO_URL%
git config core.sparseCheckout true
echo indextts/ >> .git/info/sparse-checkout
git pull origin main

:ITTSSKIP

echo/
set isrvc=
set /p isrvc=SPARSE APPLIO RVC REPO?[y/n]: 

if /i not "%isrvc%"=="y" goto RVCSKIP

echo/
echo ...CLONING APPLIO RVC FILES...
echo/
set PATH=%PATH%;%ARTHA_GITHUB_DIR%;%ARTHA_GITHUB_DIR%bin;%ARTHA_GITHUB_DIR%cmd

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Git not found
    echo Applio clone failed!
    echo/
    goto exit
)

set APPLIO_URL=https://github.com/IAHispano/Applio.git

if not exist %ARTHA_HOME_DIR%temp_applio md %ARTHA_HOME_DIR%temp_applio
cd %ARTHA_HOME_DIR%temp_applio

git init
git remote add -f origin %APPLIO_URL%
git config core.sparseCheckout true
echo rvc/ >> .git/info/sparse-checkout
git pull origin main

echo/
echo ...MOVING RVC FILES...
echo/
xcopy rvc ..\rvc\ /e /i /y
cd ..
rmdir /s /q temp_applio

:RVCSKIP

echo/
echo ...FIXING DEPENDENCIES...
echo/
set ARTHA_FIX_SPEECHBRAIN=%ARTHA_ENV_DIR%Lib\site-packages\speechbrain\utils\
if exist %ARTHA_FIX_SPEECHBRAIN% (
    copy /y %ARTHA_FIX_DIR%importutils.py %ARTHA_FIX_SPEECHBRAIN%
)
set ARTHA_FIX_INDEXTTS=%ARTHA_HOME_DIR%indextts\gpt\
if exist %ARTHA_FIX_INDEXTTS% (
    copy /y %ARTHA_FIX_DIR%model_v2.py %ARTHA_FIX_INDEXTTS%
)
set ARTHA_FIX_RVC=%ARTHA_HOME_DIR%rvc\
if exist %ARTHA_FIX_RVC% (
	copy /y %ARTHA_FIX_DIR%core.py %ARTHA_FIX_RVC%
    copy /y %ARTHA_FIX_DIR%infer.py %ARTHA_FIX_RVC%infer\
	copy /y %ARTHA_FIX_DIR%train.py %ARTHA_FIX_RVC%train\
)

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