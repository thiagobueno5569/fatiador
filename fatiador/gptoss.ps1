# Teste com um modelo MAIOR que a RAM: gpt-oss-120b (117 bilhoes de parametros, 63 GB).
# Use o GPTOSS.bat. Registro em gptoss\log.txt
param([int]$Manter = 32, [string]$Assuntos = "eletronica,calculo,culinaria", [int]$Chunks = 16, [switch]$ApagarFatiasQwen)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force "gptoss" | Out-Null
Start-Transcript -Path (Join-Path $PSScriptRoot "gptoss\log.txt") -Append | Out-Null
function Etapa($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }
function Checar($nome) { if ($LASTEXITCODE -ne 0) { throw "A etapa '$nome' falhou (codigo $LASTEXITCODE)." } }
function LivreGB($caminho) { [math]::Round((Get-PSDrive ((Resolve-Path $caminho).Drive.Name)).Free / 1GB, 0) }
try {
    $py = $null
    foreach ($c in @("py", "python")) { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } }
    if (-not $py) { throw "Python nao encontrado." }
    $LL = (Get-ChildItem "llama" -Recurse -Filter "llama-imatrix.exe" | Select-Object -First 1).DirectoryName
    if (-not $LL) { throw "Nao achei o llama.cpp na pasta llama (rode o RODAR.bat antes)." }

    if ($ApagarFatiasQwen -and (Test-Path "fatias")) {
        Etapa "0/4 Apagando as fatias do Qwen3 (pasta fatias) pra liberar espaco"
        $ok = $false
        for ($i = 0; $i -lt 120; $i++) {
            try { Remove-Item "fatias" -Recurse -Force -ErrorAction Stop; $ok = $true; break }
            catch {
                if ($i -eq 0) { Write-Host "As fatias estao em uso: FECHE a janela do app de conversa (CONVERSAR.bat)..." -ForegroundColor Yellow }
                Start-Sleep -Seconds 5
            }
        }
        if (-not $ok) { throw "Nao consegui apagar a pasta fatias (ainda em uso)." }
        Write-Host "Pasta fatias do Qwen3 apagada."
    }

    Etapa "1/4 Baixando o gpt-oss-120b (63 GB; pode levar 1 hora; se cair, rode de novo que continua)"
    $pastaModelo = "E:\modelos"
    New-Item -ItemType Directory -Force $pastaModelo | Out-Null
    $Gguf = Join-Path $pastaModelo "gpt-oss-120b-MXFP4.gguf"
    $tamanhoCerto = 63387346208
    $atual = 0; if (Test-Path $Gguf) { $atual = (Get-Item $Gguf).Length }
    if ($atual -ne $tamanhoCerto -or (Test-Path "$Gguf.progresso.json")) {
        $falta = [math]::Ceiling(($tamanhoCerto - $atual) / 1GB) + 2
        $livre = LivreGB $pastaModelo
        Write-Host "Falta baixar $falta GB; espaco livre: $livre GB"
        if ($livre -lt $falta) { throw "Pouco espaco em disco pra baixar o modelo (precisa de $falta GB)." }
        $url = "https://huggingface.co/ggml-org/gpt-oss-120b-GGUF/resolve/main/gpt-oss-120b-MXFP4.gguf"
        $sha = ""
        try {
            $arvore = Invoke-RestMethod "https://huggingface.co/api/models/ggml-org/gpt-oss-120b-GGUF/tree/main" -Headers @{ "User-Agent" = "fatiador" }
            $sha = ($arvore | Where-Object { $_.path -eq "gpt-oss-120b-MXFP4.gguf" }).lfs.oid
        } catch { Write-Host "Nao consegui o SHA-256 oficial; vou baixar sem conferir." -ForegroundColor Yellow }
        # 8 conexoes em paralelo, continua de onde parou; confere o SHA-256 no final.
        $extra = @(); if ($sha) { $extra = @("--sha256", $sha) }
        & $py baixar_modelo.py $url $Gguf $tamanhoCerto --conexoes 8 @extra | Out-Host
        Checar "baixar modelo"
    }
    Write-Host "Modelo: $Gguf"

    Etapa "2/4 Medindo os caminhos e detectando super-especialistas (o modelo e maior que a RAM: 1 a 2 horas)"
    Set-Location "gptoss"
    New-Item -ItemType Directory -Force "textos" | Out-Null
    foreach ($a in $Assuntos.Split(",")) { Copy-Item -Recurse -Force (Join-Path "..\textos" $a) "textos" }
    if (-not (Test-Path "especialistas.json")) {
        & $py ..\1_medir_especialistas.py --llama $LL --gguf $Gguf --textos textos --chunks $Chunks | Out-Host; Checar "medir"
    } else { Write-Host "especialistas.json ja existe, pulando" }

    Etapa "3/4 Fatiando ($Manter de 128 especialistas)"
    $n = $Assuntos.Split(",").Count
    $porFatia = [math]::Ceiling(63.4 * $Manter / 128 + 3)
    $livre = LivreGB "."
    Write-Host "Cada fatia: uns $porFatia GB; $n fatias; espaco livre: $livre GB"
    if ((-not (Test-Path "fatias")) -and $livre -lt ($porFatia * $n + 2)) { throw "Pouco espaco em disco pras fatias. Libere espaco (ex.: apague a pasta fatias do Qwen) ou use menos assuntos." }
    $prontas = @($Assuntos.Split(",") | Where-Object { Test-Path ("fatias\" + $_ + "-" + $Manter + "esp.gguf") }).Count
    if ($prontas -eq $n) { Write-Host "As $n fatias ja existem, pulando." }
    else {
        Remove-Item "fatias" -Recurse -Force -ErrorAction SilentlyContinue
        & $py ..\2_fatiar_gguf.py --gguf $Gguf --manter $Manter | Out-Host; Checar "fatiar"
    }

    Etapa "4/4 Comparando com o modelo inteiro (o inteiro roda do SSD e e bem lento: 30 a 60 min)"
    & $py ..\3_comparar.py --llama $LL --original $Gguf --textos textos --chunks 8 | Out-Host; Checar "comparar"
    Copy-Item "resultados.md" "resultados-gptoss-$($Manter)esp.md" -Force
    Write-Host ""
    Write-Host "PRONTO! Resultado em gptoss\resultados-gptoss-$($Manter)esp.md" -ForegroundColor Green
    Write-Host "Pra conversar com ele: CONVERSAR_GPTOSS.bat"
}
catch {
    Write-Host ""
    Write-Host "ERRO: $_" -ForegroundColor Red
    Write-Host "Mande o arquivo gptoss\log.txt pro Claude." -ForegroundColor Yellow
}
finally { Stop-Transcript | Out-Null }
