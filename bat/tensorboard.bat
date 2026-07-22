@echo off

Title WEBUI PYTHON - %~dp0

call paths.bat

call site.bat

echo STARTING TENSORBOARD
echo --------------------
cd /d %ARTHA_HOME_DIR%
%PYTHON% -m tensorboard.main --logdir_spec Trainer:trains,Finetune:finetunes

cmd /k