# 🔪 Fatiador — caminhos por assunto em modelos Mixture-of-Experts

> **Rodar modelos de IA gigantes numa placa de vídeo comum carregando só os especialistas do assunto da conversa.**

Uma ideia de pesquisa testada de ponta a ponta num PC doméstico (RTX 2060 com 6 GB, 32 GB de RAM): os modelos *Mixture-of-Experts* (MoE) são **cortados em fatias por assunto**. Cada fatia guarda um **núcleo comum** e só os especialistas que aquele assunto realmente usa. Tudo foi medido em modelos reais (Qwen3-30B-A3B e gpt-oss-120b), com grupo de controle.

<p align="center">
  <img src="docs/img/qualidade_fatias.png" width="900" alt="Qualidade das fatias por assunto">
</p>

## Resultados em uma tabela

| Modelo | Fatia | Tamanho | Qualidade (vs. modelo inteiro) | Velocidade |
|---|---|---|---|---|
| Qwen3-30B-A3B (inteiro) | — | 18,6 GB | referência | 15,7 tokens/s |
| Qwen3-30B-A3B | **64 de 128 especialistas + núcleo** | 9,8 GB | **só 1% a 18% pior** | 24,3 tokens/s (**1,5×**) |
| Qwen3-30B-A3B | **32 de 128 especialistas + núcleo** | **5,4 GB** (cabe na placa) | 33% a 96% pior | **55,9 tokens/s (3,5×)** |
| gpt-oss-120b (inteiro) | — | 63,4 GB (**o dobro da RAM**) | referência | não roda direito no PC |
| gpt-oss-120b | 32 de 128 especialistas | 17,5 GB | 2,1× pior em texto (cálculo); conversa ainda em ajuste ⚠️ | 10 a 24 tokens/s |

*Qualidade medida por perplexidade em artigos da Wikipédia **diferentes** dos usados para calibrar. Os 6 assuntos são cálculo, culinária, eletrônica, futebol, história e programação.*

---

## 💡 A ideia

Um modelo MoE tem, em cada camada, dezenas de "especialistas" (pequenas redes), e um **roteador** escolhe alguns deles para cada palavra. O Qwen3-30B tem 128 especialistas por camada e usa 8 por vez. Em vez de manter todos na memória, a ideia é:

1. **Medir** quais especialistas cada assunto realmente usa;
2. **Cortar** o modelo em fatias: **núcleo comum** (atenção, embeddings, normalizações, *super-especialistas*) + **os especialistas favoritos do assunto**;
3. **Carregar só a fatia** da conversa atual e **trocar de fatia** quando o assunto muda, como um cérebro que ativa só as regiões que importam.

```mermaid
flowchart LR
    P[Pergunta do usuário] --> R{Roteador de assunto<br/>TF-IDF · cosseno}
    R -->|eletrônica| F1[Fatia eletrônica<br/>núcleo + 32 esp./camada]
    R -->|cálculo| F2[Fatia cálculo]
    R -->|culinária| F3[Fatia culinária]
    F1 --> G[(Placa de vídeo<br/>6 GB)]
    F2 -.->|troca em segundos| G
    F3 -.-> G
    M[(Modelo inteiro<br/>no SSD)] -.->|fatiado uma vez| F1 & F2 & F3
```

## 📐 A matemática

**Roteador como geometria.** Numa camada com especialistas $e_1,\dots,e_N$, o roteador tem um vetor $w_i$ por especialista. Para uma palavra com estado oculto $h$, o roteador calcula

$$s_i = w_i \cdot h, \qquad \text{usados} = \mathrm{top}_k(s_1,\dots,s_N)$$

Ou seja, cada especialista "manda" num **cone** do espaço, e a palavra usa os $k$ especialistas que apontam mais perto dela. Um **assunto** é uma *nuvem* de pontos nesse espaço.

**Fatia.** Para o assunto $t$, contamos em cada camada $\ell$ quantas vezes cada especialista foi escolhido, $c^{(t)}_{\ell,i}$, e mantemos

$$S^{(t)}_\ell = \mathcal{N}_\ell \;\cup\; \mathrm{top}_{M - |\mathcal{N}_\ell|}\big(c^{(t)}_{\ell,1},\dots,c^{(t)}_{\ell,N}\big)$$

em que $\mathcal{N}_\ell$ é o **núcleo** (super-especialistas) e $M$ é o tamanho da fatia (32 ou 64). O roteador da fatia só enxerga $S^{(t)}_\ell$: se a palavra "queria" um especialista cortado, ela usa o próximo mais próximo dentro da fatia.

**Cobertura** (quanto a fatia cobre do que o assunto usa):

$$\text{cobertura}^{(t)} = \frac{1}{L}\sum_\ell \frac{\sum_{i\in S^{(t)}_\ell} c^{(t)}_{\ell,i}}{\sum_i c^{(t)}_{\ell,i}}$$

**Detecção automática de super-especialistas.** O `llama-imatrix` guarda, por especialista, a soma dos quadrados das ativações que entram na projeção `down`. Dividindo pelo número de vezes que ele foi usado, temos a ativação de pico típica $a_{\ell,i}$:

$$\rho_{\ell,i} = \frac{a_{\ell,i}}{\mathrm{mediana}_j\, a_{\ell,j}} > 5000 \;\Rightarrow\; \text{super-especialista} \Rightarrow \text{entra no núcleo}$$

**Velocidade com cache** (proposta da próxima etapa). Se uma fração $m$ das escolhas cair fora da placa e for atendida pela RAM, o tempo por palavra fica entre os dois extremos medidos:

$$\frac{1}{v(m)} = \frac{m}{v_{\text{RAM}}} + \frac{1-m}{v_{\text{GPU}}}, \qquad v_{\text{RAM}} = 15{,}8,\; v_{\text{GPU}} = 72{,}4 \text{ tokens/s}$$

Com $m = 20\%$, a estimativa é de ~42 tokens/s, com **a mesma qualidade do modelo inteiro**.

---

## 🔬 Descobertas

### 1. Os caminhos por assunto existem (com grupo de controle)
Nos 12 testes do Qwen3, a fatia do **assunto certo** sempre ganhou da fatia de **outro assunto**. No gpt-oss-120b o efeito é ainda mais forte: em cálculo, a fatia certa ficou em **17,5** de perplexidade e a errada em **149**.

### 2. Super-especialistas: 3 especialistas em 6.144 decidem tudo
As primeiras fatias **desmoronaram** (perplexidade 10 a 17 vezes pior). A causa: o Qwen3-30B tem **3 "super-especialistas"** (camada 1/especialista 68, camada 2/92 e camada 3/82) que quase nunca são escolhidos, mas sem os quais o modelo quebra. Eles ficavam entre a 62ª e a 123ª posição no ranking de uso e eram sempre cortados. Colocando os três no **núcleo**, a fatia de 64 especialistas passou de **6 a 9 vezes pior** pra **só 1% a 18% pior**.

<p align="center"><img src="docs/img/super_especialistas.png" width="900" alt="Detecção de super-especialistas"></p>

### 3. Dá pra achar os super-especialistas automaticamente
A ativação de pico deles é de **15 mil a 400 mil vezes** maior que a dos outros especialistas da camada, e nenhum outro passa de mil. O detector encontrou **exatamente** os 3 relatados na literatura (Su et al., 2025), sem nenhuma lista prévia. No gpt-oss-120b ele **não encontrou nenhum**: nem todo MoE tem super-especialistas.

### 4. Assuntos técnicos têm caminhos mais próprios
<p align="center"><img src="docs/img/sobreposicao.png" width="520" alt="Sobreposição entre assuntos"></p>

Cálculo compartilha só **40 a 61%** dos especialistas favoritos com os outros assuntos. Culinária, futebol e história compartilham **75 a 78%** entre si: é a "linguagem do dia a dia". A ideia funciona melhor justamente onde especialização importa.

### 5. O formato de conversa também faz parte do núcleo
Calibrando só com texto de enciclopédia, a fatia do gpt-oss entendia o assunto (perplexidade razoável), mas **se perdia ao conversar**: gerava "… … …" e nunca chegava à resposta final. Os especialistas do **formato de chat** (marcadores `<|channel|>`, fase de raciocínio) tinham sido cortados. Daí veio o `6_calibrar_chat.py`, que calibra com conversas no formato do próprio modelo (em teste).

### 6. Velocidade

<p align="center"><img src="docs/img/velocidade.png" width="800" alt="Velocidade"></p>

<p align="center"><img src="docs/img/mapa_calculo.png" width="900" alt="Mapa de uso dos especialistas em cálculo"></p>

---

## 🧪 Simulações (antes dos modelos reais)

A ideia começou numa simulação do zero em NumPy (`simulacoes/`):

| Experimento | Resultado |
|---|---|
| `experimento_blocos.py`: matriz em blocos por assunto vs. rede densa (mesmo nº de parâmetros) | mesma acurácia (91,4% vs. 90,5%) com **6,5× menos multiplicações** |
| Mesmo experimento, aprendendo um assunto de cada vez | a rede densa **esqueceu** os primeiros assuntos (média de 63%); a de blocos manteve **92% em todos** (sem *esquecimento catastrófico*) |
| `experimento_cerebro.py`: núcleo + blocos com memória em 3 níveis (VRAM → RAM → SSD), com perda de bits quando "esquecidos" | modelo de 14 GB ocupando só **5 GB na placa**; **relembrar** um assunto esquecido foi **1,2 a 4,5× mais rápido** que aprender do zero (6 sementes) |

---

## 🚀 Como rodar (Windows)

Pré-requisitos: Python 3.11+, uma placa NVIDIA e um modelo MoE em GGUF (testado com Qwen3-30B-A3B Q4_K_M e gpt-oss-120b MXFP4).

```text
fatiador\RODAR.bat -Manter 64     # baixa textos, mede caminhos, fatia e compara (llama.cpp CUDA é baixado sozinho)
fatiador\CONVERSAR.bat            # app de conversa: escolhe a fatia pelo assunto e mostra tudo graficamente
fatiador\GPTOSS_V2.bat            # experimento com o gpt-oss-120b (63 GB), calibrado com conversas
```

| Script | O que faz |
|---|---|
| `0_baixar_textos.py` | Baixa artigos da Wikipédia por assunto (calibração e avaliação **separadas**) |
| `1_medir_especialistas.py` | Roda o `llama-imatrix`, conta o uso de cada especialista por camada e **detecta super-especialistas** |
| `2_fatiar_gguf.py` | Gera um GGUF por assunto: núcleo + top-M especialistas (lê o modelo direto do disco, sem precisar de GPU) |
| `3_comparar.py` | Perplexidade (inteiro × fatia certa × fatia errada), velocidade e tamanho |
| `4_ajuste_fino.py` | Ajuste fino só dos roteadores com uma **força de agrupamento por assunto** (IBM Granite 3.1 1B-A400M) |
| `5_conversar.py` + `conversar.html` | App web: roteador de assunto, troca de fatia, mapa 48×128 dos especialistas ligados, velocidade ao vivo |
| `6_calibrar_chat.py` | Calibração no formato de conversa do modelo (harmony / chatml) |
| `7_testar_conversa.py` | Faz perguntas reais a cada fatia e detecta quando ela "desmonta" |
| `baixar_modelo.py` | Download paralelo (8 conexões) com retomada e conferência de SHA-256 |

**Hardware dos testes:** Xeon E5-2690 v4 (14 núcleos), 32 GB de DDR4 (2 de 4 canais), RTX 2060 6 GB, SSD SATA → NVMe WD SN750 SE.

## 📁 Estrutura

```
fatiador/      código do pipeline e do app
simulacoes/    os experimentos iniciais em NumPy
resultados/    tabelas e contagens de especialistas medidas (Qwen3-30B e gpt-oss-120b)
docs/img/      gráficos gerados a partir dos dados reais
```

## ⚖️ Limitações (com honestidade)

- **Perplexidade não é tudo:** é uma medida de texto corrido. A descoberta 5 mostra que uma fatia pode ter perplexidade razoável e mesmo assim conversar mal. Os testes de conversa (`7_testar_conversa.py`) entram justamente pra isso.
- **6 assuntos, ~30 artigos cada:** é pouco pra conclusões gerais. A escolha da fatia por TF-IDF é simples de propósito.
- **Modelos prontos não foram treinados para isso:** os especialistas não se organizam por assunto humano. A proposta forte é **treinar** (ou ajustar) com roteamento por assunto (`4_ajuste_fino.py`).
- O cache "especialistas na placa + o resto na RAM" ainda é uma **estimativa**: o llama.cpp não divide uma camada de especialistas entre GPU e CPU.

## 🗺️ Próximos passos

- [ ] Resultado do teste de conversa do gpt-oss calibrado com chat (32 × 64 especialistas)
- [ ] Ajuste fino dos roteadores com força de agrupamento (Granite), medindo antes e depois
- [ ] Cache por conversa: fatia na placa, especialistas ausentes atendidos pela RAM (qualidade igual à do modelo inteiro)
- [ ] Mais assuntos, e um roteador de assunto por embeddings
- [ ] Curva completa qualidade × tamanho (16, 24, 32, 48, 64, 96)

## 📚 Trabalhos relacionados

- Su et al. (2025), *Unveiling Super Experts in Mixture-of-Experts LLMs*, [arXiv:2507.23279](https://arxiv.org/abs/2507.23279), ICLR 2026
- Dong et al. (2025), *Domain-Specific Pruning of Large MoE Models with Few-shot Demonstrations* (EASY-EP), [arXiv:2504.06792](https://arxiv.org/abs/2504.06792)
- Eliseev & Mazur (2023), *Fast Inference of Mixture-of-Experts Language Models with Offloading*
- Aljundi et al. (2017), *Expert Gate: Lifelong Learning with a Network of Experts*
- Martinetz & Schulten (1991), *Neural Gas*; Kohonen (1982), *Self-Organizing Maps*
- [llama.cpp](https://github.com/ggml-org/llama.cpp): `llama-imatrix`, `llama-perplexity`, `llama-server`

## 👤 Autor

**Thiago Bueno da Silva**, estudante de Engenharia Elétrica na UFSJ. Ideia original: matrizes divididas em blocos/vetores por assunto, com um núcleo comum, especialistas que se fortalecem com o uso e uma hierarquia de memória (VRAM → RAM → SSD), pra que um modelo enorme rode numa placa comum.

Desenvolvido com a ajuda do Claude (Anthropic) na implementação e nas medições.

Licença: [MIT](LICENSE).
