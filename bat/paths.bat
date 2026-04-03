@echo off

::DOWNLOAD LOCATIONS

set ARTHA_PYTHON_URL=https://www.python.org/ftp/python/
set ARTHA_PIP_URL=https://bootstrap.pypa.io/get-pip.py
set ARTHA_MYSQL_URL=https://dev.mysql.com/get/Downloads/MySQL-8.0/mysql-8.0.35-winx64.zip
set ARTHA_GITHUB_URL=https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/PortableGit-2.43.0-64-bit.7z.exe
set ARTHA_FFMPEG_STABLE_URL=https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-7.1.1-full_build.7z
set ARTHA_FFMPEG_LATEST_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z
set ARTHA_YT_DIP_URL=https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_x86.exe
set ARTHA_OLLAMA_URL=https://ollama.com/download/OllamaSetup.exe

::PATHS

set ARTHA_HOME=wui
set ARTHA_PATH=%~dp0
set ARTHA_BASE_DIR=%ARTHA_PATH:~0,-4%
set ARTHA_HOME_DIR=%ARTHA_BASE_DIR%%ARTHA_HOME%\
set ARTHA_BIN_DIR=%ARTHA_BASE_DIR%bin\
set ARTHA_PYTHON_DIR=%ARTHA_BIN_DIR%python\
set ARTHA_PYTHON_SCR=%ARTHA_PYTHON_DIR%Scripts\

set PYTHON=%ARTHA_PYTHON_DIR%python.exe

set ARTHA_EXE_DIR=%ARTHA_BASE_DIR%bat\exe\
set ARTHA_FIX_DIR=%ARTHA_BASE_DIR%bat\fix\
set ARTHA_BCK_DIR=%ARTHA_BASE_DIR%bat\%ARTHA_HOME%\

set ARTHA_ENV_DIR=%ARTHA_BASE_DIR%env\

set ARTHA_GITHUB_DIR=%ARTHA_BIN_DIR%github\
set ARTHA_MYSQLD_DIR=%ARTHA_BIN_DIR%mysql\
set ARTHA_FFMPEG_DIR=%ARTHA_BIN_DIR%ffmpeg\
set ARTHA_YT_DIP_DIR=%ARTHA_BIN_DIR%yt-dip\
set ARTHA_OLLAMA_DIR=%ARTHA_BIN_DIR%ollama\

set ARTHA_GITHUB_BIN=%ARTHA_GITHUB_DIR%bin\
set ARTHA_HUGGIN_BIN=%ARTHA_BIN_DIR%hf\
set ARTHA_MYSQLD_BIN=%ARTHA_MYSQLD_DIR%bin\
set ARTHA_FFMPEG_BIN=%ARTHA_FFMPEG_DIR%bin\
set ARTHA_ESPEAK_BIN=%ARTHA_BIN_DIR%espeak\

::CUDA

set ARTHA_NVCUDA_VER=v12.8
set ARTHA_NVCUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%ARTHA_NVCUDA_VER%\
set ARTHA_NVCUDA_BIN=%ARTHA_NVCUDA_DIR%bin\
set ARTHA_NVCUDA_LIB=%ARTHA_NVCUDA_DIR%lib\
set ARTHA_NVCUDA_NVP=%ARTHA_NVCUDA_DIR%libnvvp\
set ARTHA_NVCUDA_INC=%ARTHA_NVCUDA_DIR%include\

::VC

set ARTHA_VSCODE_CLD=C:\Program Files\Microsoft Visual Studio\2022\Professional\
set ARTHA_VSCODE_CLE=%ARTHA_VSCODE_CLD%VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\

::TRITON

set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
set TORCHINDUCTOR_CACHE_DIR=%ARTHA_BASE_DIR%models\torch\cache
set TORCHDYNAMO_VERBOSE=1

::ESPEAK

set PHONEMIZER_ESPEAK_PATH=%ARTHA_ESPEAK_BIN%
set PHONEMIZER_ESPEAK_LIBRARY=%ARTHA_ESPEAK_BIN%libespeak-ng.dll

::EVIRONMENT VARIABLES

set PATH=C:\Windows;C:\Windows\system32;%ARTHA_PYTHON_DIR%;%ARTHA_PYTHON_SCR%;%ARTHA_HOME_DIR%

if exist %ARTHA_GITHUB_BIN% set PATH=%PATH%;%ARTHA_GITHUB_BIN%
if exist %ARTHA_HUGGIN_BIN% set PATH=%PATH%;%ARTHA_HUGGIN_BIN%
if exist %ARTHA_FFMPEG_BIN% set PATH=%PATH%;%ARTHA_FFMPEG_BIN%
if exist %ARTHA_MYSQLD_BIN% set PATH=%PATH%;%ARTHA_MYSQLD_BIN%
if exist %ARTHA_OLLAMA_DIR% set PATH=%PATH%;%ARTHA_OLLAMA_DIR%

if exist "%ARTHA_NVCUDA_DIR%" set PATH=%PATH%;%ARTHA_NVCUDA_BIN%
if exist "%ARTHA_NVCUDA_DIR%" set PATH=%PATH%;%ARTHA_NVCUDA_LIB%
if exist "%ARTHA_NVCUDA_DIR%" set PATH=%PATH%;%ARTHA_NVCUDA_NVP%
if exist "%ARTHA_NVCUDA_DIR%" set PATH=%PATH%;%ARTHA_NVCUDA_INC%

if exist "%ARTHA_VSCODE_CLE%cl.exe" set PATH=%PATH%;%ARTHA_VSCODE_CLE%

set UV_TOOL_BIN_DIR=%ARTHA_BIN_DIR%hf\
set HF_HOME=%ARTHA_BASE_DIR%models\hf\
set HF_HUB_CACHE=%ARTHA_BASE_DIR%models\hf\hf_cache\
set XDG_CACHE_HOME=%ARTHA_BASE_DIR%models\
set TORCH_HOME=%ARTHA_BASE_DIR%models\torch\
set TORCH_HF_HOME=%ARTHA_BASE_DIR%models\huggingface\
set OLLAMA_MODELS=%ARTHA_BASE_DIR%models\ollama\

::TEMP DIRECTORY

mkdir "%ARTHA_HOME_DIR%\temp" 2>nul
set "TMP=%ARTHA_HOME_DIR%\temp"
set "TEMP=%ARTHA_HOME_DIR%\temp"
set "GRADIO_TEMP_DIR=%ARTHA_HOME_DIR%\temp"

::TOKENS

set HF_TOKEN=

::KEYS

set GEMINI_API_KEY=