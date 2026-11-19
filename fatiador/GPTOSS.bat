@echo off
chcp 65001 >nul
title Fatiador - gpt-oss-120b
echo Teste com o gpt-oss-120b. Continua de onde parou; deixe a janela aberta.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gptoss.ps1" %*
echo.
pause
