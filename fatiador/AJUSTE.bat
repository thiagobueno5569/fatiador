@echo off
chcp 65001 >nul
title Fatiador - ajuste fino dos roteadores
echo Ajuste fino dos roteadores. Pode demorar ate 1 hora; deixe a janela aberta.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0ajuste.ps1" %*
echo.
pause
