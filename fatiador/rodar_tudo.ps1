# Roda o experimento inteiro: textos -> caminhos por assunto -> fatias -> comparacao.
# Use o RODAR.bat (duplo clique). Tudo fica registrado em log.txt.
param(
    [string]$Gguf = "",          # caminho do modelo; vazio = procura no LM Studio
    [int]$Manter = 32,           # especialistas por camada em cada fatia
    [string]$CpuMoeFatia = "auto" # camadas da fatia com especialistas na RAM (auto = calcula)
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
Start-Transcript -Path (Join-Path $PSScriptRoot "log.txt") -Append | Out-Null

function Etapa($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }
function Checar($nome) { if ($LASTEXITCODE -ne 0) { throw "A etapa '$nome' falhou (codigo $LASTEXITCODE)." } }

try {
    # ---------------------------------------------------------------- Python
    Etapa "1/6 Python e bibliotecas"
    $py = $null
    foreach ($c in @("py", "python")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
    }
    if (-not $py) { throw "Python nao encontrado. Instale em https://www.python.org/downloads/ (marque 'Add to PATH')." }
    & $py -m pip install -q -U numpy gguf; Checar "pip install"
    & $py -c "import numpy, gguf; print('numpy e gguf ok')"; Checar "import"

    # ---------------------------------------------------------------- llama.cpp
    Etapa "2/6 llama.cpp com CUDA"
    $exeImatrix = Get-ChildItem "llama" -Recurse -Filter "llama-imatrix.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exeImatrix) {
        # As versoes com os programas do Windows sao marcadas como "pre-lancamento",
        # entao procura na lista de versoes recentes a primeira que tenha o pacote CUDA.
        $lista = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30" -Headers @{ "User-Agent" = "fatiador" }
        $bin = $null; $rt = $null
        foreach ($rel in $lista) {
            foreach ($cuda in @("12", "13")) {
                $bin = $rel.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cuda-$cuda\.[\d.]+-x64\.zip$" } | Select-Object -First 1
                if ($bin) {
                    $ver = [regex]::Match($bin.name, "cuda-([\d.]+)-x64").Groups[1].Value
                    $rt = $rel.assets | Where-Object { $_.name -eq "cudart-llama-bin-win-cuda-$ver-x64.zip" } | Select-Object -First 1
                    break
                }
            }
            if ($bin) { Write-Host "Versao do llama.cpp: $($rel.tag_name)"; break }
        }
        if (-not $bin) {
            Write-Host "Nao achei versao CUDA; usando a versao so de CPU (mais lenta)." -ForegroundColor Yellow
            foreach ($rel in $lista) {
                $bin = $rel.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cpu-x64\.zip$" } | Select-Object -First 1
                if ($bin) { break }
            }
        }
        if (-not $bin) { throw "Nao achei nenhum pacote do llama.cpp para Windows no GitHub." }
        New-Item -ItemType Directory -Force "llama" | Out-Null
        foreach ($a in $(@($bin, $rt) | Where-Object { $_ })) {
            Write-Host "Baixando $($a.name) ..."
            $zip = Join-Path $env:TEMP $a.name
            Invoke-WebRequest $a.browser_download_url -OutFile $zip
            Expand-Archive $zip -DestinationPath "llama" -Force
        }
        $exeImatrix = Get-ChildItem "llama" -Recurse -Filter "llama-imatrix.exe" | Select-Object -First 1
        if (-not $exeImatrix) { throw "Baixei o llama.cpp mas nao achei o llama-imatrix.exe" }
        # junta as DLLs do CUDA na mesma pasta dos programas
        Get-ChildItem "llama" -Recurse -Filter "cu*.dll" | Where-Object { $_.DirectoryName -ne $exeImatrix.DirectoryName } |
            ForEach-Object { Copy-Item $_.FullName $exeImatrix.DirectoryName -Force }
    }
    $LL = $exeImatrix.DirectoryName
    Write-Host "llama.cpp em $LL"

    # ---------------------------------------------------------------- modelo
    Etapa "3/6 Procurando o Qwen3-30B-A3B"
    if (-not $Gguf) {
        $pastas = @("$env:USERPROFILE\.lmstudio\models", "$env:USERPROFILE\.cache\lm-studio\models")
        $achado = foreach ($p in $pastas) {
            if (Test-Path $p) { Get-ChildItem $p -Recurse -Filter "*.gguf" -ErrorAction SilentlyContinue }
        }
        $modelo = $achado | Where-Object { $_.Name -match "Qwen3-30B-A3B" -and $_.Name -notmatch "mmproj|-0000[2-9]-of-" } |
            Sort-Object Length | Select-Object -First 1
        if (-not $modelo) { throw "Nao achei o Qwen3-30B-A3B no LM Studio. Rode de novo com: RODAR.bat -Gguf ""C:\caminho\modelo.gguf""" }
        $Gguf = $modelo.FullName
    }
    $tamGB = [math]::Round((Get-Item $Gguf).Length / 1GB, 1)
    Write-Host "Modelo: $Gguf ($tamGB GB)"

    $livreGB = [math]::Round((Get-PSDrive ((Get-Location).Drive.Name)).Free / 1GB, 0)
    $precisa = [math]::Ceiling($tamGB * ($Manter / 128 + 0.08) * 6) + 2
    Write-Host "Espaco livre: $livreGB GB (as 6 fatias precisam de uns $precisa GB)"
    if ($livreGB -lt $precisa) { throw "Pouco espaco em disco. Libere espaco ou mova a pasta pra outro disco." }

    # ---------------------------------------------------------------- etapas
    Etapa "4/6 Baixando textos da Wikipedia"
    $prontos = @(Get-ChildItem "textos" -Directory -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "avaliacao.txt") })
    if ($prontos.Count -lt 6) {
        & $py 0_baixar_textos.py | Out-Host; Checar "baixar textos"
        $prontos = @(Get-ChildItem "textos" -Directory | Where-Object { Test-Path (Join-Path $_.FullName "avaliacao.txt") })
        Remove-Item "especialistas.json" -ErrorAction SilentlyContinue
    }
    Write-Host "Assuntos prontos: $($prontos.Name -join ', ')"
    if ($prontos.Count -lt 2) { throw "So consegui textos de $($prontos.Count) assunto(s); a Wikipedia bloqueou ou a internet caiu. Tente de novo." }

    Etapa "5/6 Medindo os caminhos de cada assunto (demora: uns 3 a 6 min por assunto)"
    $medidos = 0
    if (Test-Path "especialistas.json") {
        $medidos = @((Get-Content "especialistas.json" -Raw | ConvertFrom-Json).assuntos.PSObject.Properties).Count
    }
    if ($medidos -lt $prontos.Count) {
        & $py 1_medir_especialistas.py --llama $LL --gguf $Gguf | Out-Host; Checar "medir especialistas"
    } else { Write-Host "especialistas.json ja tem os $medidos assuntos, pulando" }

    Etapa "6/6 Fatiando e comparando (demora: uns 20 a 40 min)"
    Remove-Item "fatias" -Recurse -Force -ErrorAction SilentlyContinue
    & $py 2_fatiar_gguf.py --gguf $Gguf --manter $Manter | Out-Host; Checar "fatiar"
    & $py 3_comparar.py --llama $LL --original $Gguf --cpu-moe-fatia $CpuMoeFatia | Out-Host; Checar "comparar"
    Copy-Item "resultados.md" "resultados-$($Manter)esp-v2.md" -Force

    Write-Host ""
    Write-Host "PRONTO! O resultado esta em resultados.md (e o registro completo em log.txt)." -ForegroundColor Green
    Write-Host "As fatias ocupam espaco: pode apagar a pasta 'fatias' quando terminar."
}
catch {
    Write-Host ""
    Write-Host "ERRO: $_" -ForegroundColor Red
    Write-Host "Mande o arquivo log.txt pro Claude que ele resolve." -ForegroundColor Yellow
}
finally {
    Stop-Transcript | Out-Null
}
