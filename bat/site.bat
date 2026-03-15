@echo off

echo import site>%ARTHA_PYTHON_DIR%Lib\sitecustomize.py
echo site.addsitedir(r^"%ARTHA_ENV_DIR%\Lib\site-packages^")>>%ARTHA_PYTHON_DIR%\Lib\sitecustomize.py
echo site.addsitedir(r^"%ARTHA_BASE_DIR%%ARTHA_HOME%^")>>%ARTHA_PYTHON_DIR%\Lib\sitecustomize.py