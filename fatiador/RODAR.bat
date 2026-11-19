@echo off
chcp 65001 >nul
title Fatiador - experimento de caminhos por assunto
echo Rodando o experimento completo. Pode demorar 1 hora ou mais; deixe a janela aberta.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rodar_tudo.ps1" %*
echo.
pause
