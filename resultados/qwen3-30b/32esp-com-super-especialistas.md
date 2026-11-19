# Resultado: modelo normal × fatias por assunto

## Qualidade (perplexidade — menor é melhor)

| Assunto | Normal | Fatia certa | Diferença | Fatia errada (controle) |
|---|---|---|---|---|
| calculo | 3.53 | 4.68 | +32.7% | 21.18 (culinaria) |
| culinaria | 6.77 | 9.10 | +34.4% | 16.45 (eletronica) |
| eletronica | 4.73 | 7.60 | +60.7% | 12.09 (futebol) |
| futebol | 5.23 | 8.54 | +63.4% | 9.56 (historia) |
| historia | 4.31 | 5.87 | +36.1% | 13.50 (programacao) |
| programacao | 4.55 | 8.91 | +95.6% | 10.04 (calculo) |

## Velocidade e tamanho

| Modelo | Tamanho | Ler prompt (tok/s) | Gerar resposta (tok/s) |
|---|---|---|---|
| Normal | 18.6 GB | 213 | 15.9 |
| Fatia (calculo) | 5.4 GB | 1233 | 55.9 |

Ganho na geração: **3.5x**