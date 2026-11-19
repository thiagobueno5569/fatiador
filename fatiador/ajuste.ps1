# Ajuste fino dos roteadores (passo 4). Use o AJUSTE.bat. Registro em log_ajuste.txt.
param([int]$Passos = 400, [double]$Lambda = 0.5)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
Start-Transcript -Path (Join-Path $PSScriptRoot "log_ajuste.txt") -Append | Out-Null
function Etapa($t) { Write-Host ""; Write-Host "==== $t ====" -ForegroundColor Cyan }
function Checar($nome) { if ($LASTEXITCODE -ne 0) { throw "A etapa '$nome' falhou (codigo $LASTEXITCODE)." } }
try {
    # O PyTorch e o transformers ainda nao sao estaveis no Python 3.15: prefere 3.12 / 3.11.
    $py = $null; $pyArgs = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $ErrorActionPreference = "Continue"
        foreach ($v in @("3.12", "3.11", "3.13")) {
            try { $ok = (& py "-$v" -c "print('ok')" 2>&1 | Out-String) } catch { $ok = "" }
            if ($ok -match "^ok") { $py = "py"; $pyArgs = @("-$v"); break }
        }
        $ErrorActionPreference = "Stop"
    }
    if (-not $py) { foreach ($c in @("python", "py")) { if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break } } }
    if (-not $py) { throw "Python nao encontrado." }
    Write-Host "Usando Python: $py $pyArgs"

    Etapa "1/3 PyTorch com CUDA e transformers (download grande na primeira vez, ~3 GB)"
    $temCuda = & $py @pyArgs -c "exec('try:\n import torch; print(torch.cuda.is_available())\nexcept Exception:\n print(False)')"
    if ($temCuda -ne "True") {
        & $py @pyArgs -m pip install -U torch --index-url https://download.pytorch.org/whl/cu126 | Out-Host; Checar "instalar torch"
    }
    & $py @pyArgs -m pip install -q "transformers==5.23.0" accelerate | Out-Host; Checar "instalar transformers"
    & $py @pyArgs -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NENHUMA (vai ser muito lento)')" | Out-Host

    Etapa "2/3 Textos da Wikipedia"
    & $py @pyArgs 0_baixar_textos.py | Out-Host; Checar "baixar textos"

    Etapa "3/3 Ajuste fino dos roteadores (uns 20 a 60 min)"
    & $py @pyArgs 4_ajuste_fino.py --passos $Passos --lambda $Lambda | Out-Host; Checar "ajuste fino"

    Write-Host ""
    Write-Host "PRONTO! Resultado em resultados_ajuste.md" -ForegroundColor Green
}
catch {
    Write-Host ""
    Write-Host "ERRO: $_" -ForegroundColor Red
    Write-Host "Mande o arquivo log_ajuste.txt pro Claude." -ForegroundColor Yellow
}
finally { Stop-Transcript | Out-Null }
