@echo off

Title WEBUI PYTHON - %~dp0

call paths.bat

call site.bat

%PYTHON% --version
echo/
echo %PATH%

echo/

echo USE THE SYNTAX BELOW TO INSTALL PACKAGES
echo ----------------------------------------
echo %%PYTHON%% -m pip install numpy --target %%ARTHA_ENV_DIR%%Lib\site-packages
echo/

cmd /k