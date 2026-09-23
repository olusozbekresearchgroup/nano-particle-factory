@echo off
setlocal
python -m npf.gui %*
if errorlevel 1 pause
