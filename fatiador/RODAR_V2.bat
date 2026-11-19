@echo off
chcp 65001 >nul
title Fatiador v2 - com super-especialistas no nucleo
echo Rodando: fatias de 32 e 64 (agora com os super-especialistas) e depois o ajuste fino.
echo Deixe a janela aberta. Leva umas 2 horas.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rodar_tudo.ps1" -Manter 32
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0rodar_tudo.ps1" -Manter 64
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ajuste.ps1"
echo.
echo Tudo terminado.
pause
