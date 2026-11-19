# gpt-oss v2: calibra com CONVERSAS (formato de chat), fatia com 32 e 64 especialistas
# e testa conversando. Registro em gptoss\log_v2.txt. Resultado: gptoss\teste_conversa.md
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
Start-Transcript -Path (Join-Path $PSScriptRoot "gptoss\log_v2.txt") -Append | Out-Null
function Etapa($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }
function Checar($nome) { if ($LASTEXITCODE -ne 0) { throw "A etapa '$nome' falhou (codigo $LASTEXITCODE)." } }
try {
    $py = "py"
    $LL = (Get-ChildItem "llama" -Recurse -Filter "llama-imatrix.exe" | Select-Object -First 1).DirectoryName
    $Gguf = "E:\modelos\gpt-oss-120b-MXFP4.gguf"
    if (-not (Test-Path $Gguf)) { throw "Nao achei $Gguf" }
    Set-Location "gptoss"

    Etapa "1/4 Criando textos de calibracao em formato de conversa"
    & $py ..\6_calibrar_chat.py --gguf $Gguf --textos textos | Out-Host; Checar "calibrar"

    Etapa "2/4 Medindo os caminhos com as conversas (agora lendo do NVMe)"
    if (-not (Test-Path "especialistas_chat.json")) {
        $t0 = Get-Date
        & $py ..\1_medir_especialistas.py --llama $LL --gguf $Gguf --textos textos --arquivo calibracao_chat.txt --parse-special --chunks 16 --saida especialistas_chat.json | Out-Host
        Checar "medir"
        Write-Host ("Medicao levou {0:N0} min (no SSD SATA eram 52 a 74 min POR assunto)" -f ((Get-Date) - $t0).TotalMinutes)
    } else { Write-Host "especialistas_chat.json ja existe, pulando" }

    Etapa "3/4 Fatiando com 32 e com 64 especialistas"
    foreach ($m in @(32, 64)) {
        if (-not (Test-Path "fatias_chat$m\calculo-$($m)esp.gguf")) {
            & $py ..\2_fatiar_gguf.py --gguf $Gguf --especialistas especialistas_chat.json --manter $m --saida "fatias_chat$m" | Out-Host
            Checar "fatiar $m"
        } else { Write-Host "fatias_chat$m ja existe, pulando" }
    }

    Etapa "4/4 Testando conversa (fatia antiga x fatias novas x modelo inteiro)"
    Set-Location ..
    & $py 7_testar_conversa.py --llama llama --saida gptoss\teste_conversa.md `
        "antiga_32_wikipedia=gptoss\fatias\calculo-32esp.gguf" `
        "chat_32_calculo=gptoss\fatias_chat32\calculo-32esp.gguf" `
        "chat_32_eletronica=gptoss\fatias_chat32\eletronica-32esp.gguf" `
        "chat_32_culinaria=gptoss\fatias_chat32\culinaria-32esp.gguf" `
        "chat_64_calculo=gptoss\fatias_chat64\calculo-64esp.gguf" `
        "modelo_inteiro=$Gguf" | Out-Host
    Checar "testar conversa"
    Write-Host ""
    Write-Host "PRONTO! Respostas em gptoss\teste_conversa.md" -ForegroundColor Green
}
catch { Write-Host ""; Write-Host "ERRO: $_" -ForegroundColor Red; Write-Host "Mande o gptoss\log_v2.txt pro Claude." -ForegroundColor Yellow }
finally { Stop-Transcript | Out-Null }
