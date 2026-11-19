@echo off
chcp 65001 >nul
title Fatiador - gpt-oss v2 (calibracao com conversas)
echo Leva umas 2 horas. Deixe a janela aberta e o app de conversa FECHADO.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gptoss_v2.ps1"
echo.
pause
