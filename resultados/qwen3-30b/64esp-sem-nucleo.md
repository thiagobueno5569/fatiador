# Resultado: modelo normal × fatias por assunto

## Qualidade (perplexidade — menor é melhor)

| Assunto | Normal | Fatia certa | Diferença | Fatia errada (controle) |
|---|---|---|---|---|
| calculo | 3.53 | 3.90 | +10.4% | 49.99 (culinaria) |
| culinaria | 6.77 | 64.00 | +844.8% | 38.96 (eletronica) |
| eletronica | 4.73 | 41.60 | +780.2% | 77.97 (futebol) |
| futebol | 5.23 | 43.78 | +737.6% | 40.32 (historia) |
| historia | 4.31 | 37.84 | +777.3% | 42.23 (programacao) |
| programacao | 4.55 | 31.26 | +586.4% | 7.34 (calculo) |

## Velocidade e tamanho

| Modelo | Tamanho | Ler prompt (tok/s) | Gerar resposta (tok/s) |
|---|---|---|---|
| Normal | 18.6 GB | 213 | 16.0 |
| Fatia (calculo) | 9.8 GB | 240 | 13.9 |

Ganho na geração: **0.9x**