@echo off
chcp 65001 >nul
title Fatiador - conversa com o gpt-oss-120b fatiado (feche para desligar)
cd /d "%~dp0"
echo Abrindo o app no navegador: http://localhost:8765
py 5_conversar.py --original "E:\modelos\gpt-oss-120b-MXFP4.gguf" --dados gptoss --especialistas especialistas_chat.json --fatias fatias_chat32 --manter 32 %*
pause
