@echo off
chcp 65001 >nul
title Fatiador - conversa (feche esta janela para desligar a IA)
cd /d "%~dp0"
py -m pip install -q numpy gguf
echo Abrindo o app no navegador: http://localhost:8765
echo Para usar fatias menores e mais rapidas: CONVERSAR.bat --manter 32
echo.
py 5_conversar.py %*
pause
