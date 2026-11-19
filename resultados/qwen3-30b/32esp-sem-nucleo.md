# Resultado: modelo normal × fatias por assunto

## Qualidade (perplexidade — menor é melhor)

| Assunto | Normal | Fatia certa | Diferença | Fatia errada (controle) |
|---|---|---|---|---|
| calculo | 3.53 | 59.63 | +1590.4% | 104.58 (culinaria) |
| culinaria | 6.77 | 82.30 | +1115.0% | 84.10 (eletronica) |
| eletronica | 4.73 | 61.19 | +1194.7% | 89.93 (futebol) |
| futebol | 5.23 | 47.56 | +809.7% | 53.55 (historia) |
| historia | 4.31 | 45.93 | +965.0% | 62.85 (programacao) |
| programacao | 4.55 | 56.32 | +1136.5% | 101.38 (calculo) |

## Velocidade e tamanho

| Modelo | Tamanho | Ler prompt (tok/s) | Gerar resposta (tok/s) |
|---|---|---|---|
| Normal | 18.6 GB | 220 | 15.7 |
| Fatia (calculo) | 5.4 GB | 1505 | 73.4 |

Ganho na geração: **4.7x**