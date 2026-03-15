@echo off

Title WEBUI PYTHON - %~dp0

call paths.bat

call site.bat

echo STARTING TENSORBOARD
echo --------------------
%PYTHON% -m tensorboard.main --logdir %ARTHA_HOME_DIR%/trains

cmd /k