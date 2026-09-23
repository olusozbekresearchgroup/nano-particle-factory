@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0install_npf.ps1" %*
if errorlevel 1 pause
