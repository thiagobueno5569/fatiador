# Resultado: modelo normal × fatias por assunto

## Qualidade (perplexidade — menor é melhor)

| Assunto | Normal | Fatia certa | Diferença | Fatia errada (controle) |
|---|---|---|---|---|
| calculo | 3.53 | 3.72 | +5.5% | 9.44 (culinaria) |
| culinaria | 6.77 | 6.85 | +1.1% | 7.56 (eletronica) |
| eletronica | 4.73 | 5.28 | +11.8% | 6.92 (futebol) |
| futebol | 5.23 | 5.51 | +5.3% | 5.57 (historia) |
| historia | 4.31 | 4.44 | +3.0% | 5.46 (programacao) |
| programacao | 4.55 | 5.39 | +18.3% | 6.54 (calculo) |

## Velocidade e tamanho

| Modelo | Tamanho | Ler prompt (tok/s) | Gerar resposta (tok/s) |
|---|---|---|---|
| Normal | 18.6 GB | 222 | 16.1 |
| Fatia (calculo) | 9.8 GB | 528 | 24.3 |

Ganho na geração: **1.5x**