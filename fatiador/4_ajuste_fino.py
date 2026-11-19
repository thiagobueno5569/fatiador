"""
Passo 4 — AJUSTE FINO: move as setas do roteador pra agrupar os especialistas por assunto.

Ideia do Thiago: os especialistas "se movem" pra melhor posição entre os assuntos.
Aqui, as setas do roteador (uma por especialista, em cada camada) são treinadas com
duas forças ao mesmo tempo:
  1. Prever bem a próxima palavra (a perda normal de um modelo de linguagem), pra não
     estragar o modelo.
  2. Força de agrupamento: uma palavra de eletrônica deve escolher especialistas do
     grupo de eletrônica. É a "atração" da simulação da esfera.
Os especialistas em si ficam congelados (economiza memória: cabe numa RTX 2060).

Mede ANTES e DEPOIS do ajuste, pra cada assunto, a perplexidade (menor = melhor):
  * modelo inteiro;
  * fatia certa: só os especialistas do grupo do assunto podem ser usados;
  * fatia errada: grupo de outro assunto (controle).
Se a ideia funcionar, a "fatia certa" fica bem mais perto do "modelo inteiro" depois do ajuste.

Uso:  python 4_ajuste_fino.py              (modelo padrão: IBM Granite 3.1 1B-A400M)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------- roteadores
class Roteadores:
    """Substitui o forward de cada roteador por um que (a) pode aplicar uma máscara de
    fatia e (b) guarda os logits pra calcular a força de agrupamento."""

    def __init__(self, modelo):
        self.lista = [m for m in modelo.modules()
                      if type(m).__name__.endswith("TopKRouter") and hasattr(m, "weight")]
        if not self.lista:
            raise SystemExit("Não achei roteadores do tipo TopKRouter (use transformers 5.x).")
        self.mascaras = None          # lista de tensores bool [n_esp] por camada, ou None
        self.logits = []              # preenchido durante o treino
        self.guardar = False
        for i, r in enumerate(self.lista):
            r.forward = self._novo_forward(r, i)

    def _novo_forward(self, r, camada):
        softmax_primeiro = hasattr(r, "norm_topk_prob")   # Qwen3 / OLMoE; Granite: topk primeiro

        def forward(h):
            h = h.reshape(-1, r.weight.shape[1])
            logits = F.linear(h.to(r.weight.dtype), r.weight).float()
            if self.guardar:
                self.logits.append((camada, logits))
            if self.mascaras is not None:
                logits = logits.masked_fill(~self.mascaras[camada].to(logits.device), float("-inf"))
            if softmax_primeiro:
                prob = torch.softmax(logits, dim=-1)
                val, idx = torch.topk(prob, r.top_k, dim=-1)
                if r.norm_topk_prob:
                    val = val / val.sum(-1, keepdim=True)
            else:
                topv, idx = torch.topk(logits, r.top_k, dim=-1)
                val = torch.softmax(topv, dim=-1)
            return logits, val.to(h.dtype), idx

        return forward


# ---------------------------------------------------------------- dados
def blocos(tok, texto, tam, maximo):
    ids = tok(texto, return_tensors="pt").input_ids[0]
    out = [ids[i:i + tam] for i in range(0, len(ids) - tam + 1, tam)]
    return out[:maximo]


def carregar_textos(tok, pasta, tam, max_treino, max_aval):
    dados = {}
    for p in sorted(Path(pasta).iterdir()):
        if (p / "calibracao.txt").exists() and (p / "avaliacao.txt").exists():
            tr = blocos(tok, (p / "calibracao.txt").read_text(encoding="utf-8"), tam, max_treino)
            av = blocos(tok, (p / "avaliacao.txt").read_text(encoding="utf-8"), tam, max_aval)
            if tr and av:
                dados[p.name] = (tr, av)
    if len(dados) < 2:
        raise SystemExit("Preciso de pelo menos 2 assuntos em textos/ (rode o 0_baixar_textos.py).")
    return dados


# ---------------------------------------------------------------- medidas
@torch.no_grad()
def contar_uso(modelo, rot, blocos_, disp):
    n = rot.lista[0].weight.shape[0]
    cont = [torch.zeros(n) for _ in rot.lista]
    rot.guardar = True
    for b in blocos_:
        rot.logits.clear()
        modelo(b.unsqueeze(0).to(disp))
        for camada, lg in rot.logits:
            idx = torch.topk(lg, rot.lista[camada].top_k, dim=-1).indices.cpu()
            cont[camada] += torch.bincount(idx.reshape(-1), minlength=n).float()
    rot.guardar = False
    rot.logits.clear()
    return cont


@torch.no_grad()
def perplexidade(modelo, rot, blocos_, disp, mascaras=None):
    rot.mascaras = mascaras
    total, n = 0.0, 0
    for b in blocos_:
        x = b.unsqueeze(0).to(disp)
        perda = modelo(x, labels=x).loss
        total += perda.item() * (len(b) - 1)
        n += len(b) - 1
    rot.mascaras = None
    return math.exp(total / n)


def avaliar(modelo, rot, dados, grupos, disp):
    nomes = list(dados)
    linhas = {}
    for i, a in enumerate(nomes):
        outro = nomes[(i + 1) % len(nomes)]
        av = dados[a][1]
        linhas[a] = {
            "inteiro": perplexidade(modelo, rot, av, disp),
            "fatia_certa": perplexidade(modelo, rot, av, disp, grupos[a]),
            "fatia_errada": perplexidade(modelo, rot, av, disp, grupos[outro]),
            "outro": outro,
        }
        l = linhas[a]
        print(f"  {a:12s} inteiro {l['inteiro']:7.2f} | fatia certa {l['fatia_certa']:7.2f} | "
              f"fatia de {outro} {l['fatia_errada']:7.2f}", flush=True)
    return linhas


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--textos", default="textos")
    ap.add_argument("--manter", type=int, default=0, help="especialistas no grupo de cada assunto (0 = 1/4)")
    ap.add_argument("--passos", type=int, default=400)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.5, help="força de agrupamento")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max-treino", type=int, default=200, help="blocos de treino por assunto")
    ap.add_argument("--max-aval", type=int, default=12, help="blocos de avaliação por assunto")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    random.seed(0); torch.manual_seed(0)
    disp = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if disp == "cpu" else torch.bfloat16
    print(f"Carregando {a.modelo} em {disp}...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, dtype=dtype).to(disp)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    rot = Roteadores(modelo)
    n_esp, k = rot.lista[0].weight.shape[0], rot.lista[0].top_k
    manter = a.manter or max(k, n_esp // 4)
    print(f"{len(rot.lista)} camadas, {n_esp} especialistas, {k} ativos; grupo de cada assunto = {manter}")

    dados = carregar_textos(tok, a.textos, a.seq, a.max_treino, a.max_aval)
    print("Assuntos:", ", ".join(f"{n} ({len(t)} blocos)" for n, (t, _) in dados.items()))

    # grupos de especialistas: os 'manter' mais usados por cada assunto, em cada camada
    grupos = {}
    for nome, (tr, _) in dados.items():
        cont = contar_uso(modelo, rot, tr[:40], disp)
        grupos[nome] = []
        for c in cont:
            m = torch.zeros(n_esp, dtype=torch.bool)
            m[torch.topk(c, manter).indices] = True
            grupos[nome].append(m)

    print("\nANTES do ajuste (perplexidade, menor = melhor):", flush=True)
    t0 = time.time()
    antes = avaliar(modelo, rot, dados, grupos, disp)

    # ---- treino: só as setas do roteador se movem
    pesos = []
    for r in rot.lista:
        r.weight.data = r.weight.data.float()
        r.weight.requires_grad_(True)
        pesos.append(r.weight)
    otim = torch.optim.AdamW(pesos, lr=a.lr, weight_decay=0.0)
    if disp == "cuda":
        # use_reentrant=False: senão a força de agrupamento não recebe gradiente
        modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        modelo.enable_input_require_grads()
    modelo.train()
    nomes = list(dados)
    print(f"\nAjustando os roteadores por {a.passos} passos...", flush=True)
    for passo in range(1, a.passos + 1):
        assunto = nomes[passo % len(nomes)]
        b = random.choice(dados[assunto][0]).unsqueeze(0).to(disp)
        rot.guardar = True
        rot.logits.clear()
        perda_lm = modelo(b, labels=b).loss
        agrup = 0.0
        for camada, lg in rot.logits:
            logp = torch.log_softmax(lg, dim=-1)
            dentro = logp.masked_fill(~grupos[assunto][camada].to(lg.device), float("-inf"))
            agrup = agrup - torch.logsumexp(dentro, dim=-1).mean()
        agrup = agrup / len(rot.logits)
        rot.guardar = False
        rot.logits.clear()
        if not agrup.requires_grad:
            raise SystemExit("A força de agrupamento ficou sem gradiente (versão do PyTorch?).")
        perda = perda_lm + a.lam * agrup
        otim.zero_grad(set_to_none=True)
        perda.backward()
        torch.nn.utils.clip_grad_norm_(pesos, 1.0)
        otim.step()
        if passo % 25 == 0 or passo == 1:
            print(f"  passo {passo:4d}  perda da linguagem {perda_lm.item():.3f}  "
                  f"perda de agrupamento {agrup.item():.3f}", flush=True)
    modelo.eval()
    if disp == "cuda":
        modelo.gradient_checkpointing_disable()

    print("\nDEPOIS do ajuste:", flush=True)
    depois = avaliar(modelo, rot, dados, grupos, disp)
    torch.save({i: r.weight.detach().cpu() for i, r in enumerate(rot.lista)}, "roteadores_ajustados.pt")

    # ---- relatório
    out = [f"# Ajuste fino dos roteadores — {a.modelo}\n",
           f"Grupo de cada assunto: {manter} de {n_esp} especialistas por camada. "
           f"{a.passos} passos, força de agrupamento λ = {a.lam}.\n",
           "Perplexidade (menor = melhor). A coluna **distância** é quanto a fatia certa fica "
           "pior que o modelo inteiro.\n",
           "| Assunto | Inteiro antes | Fatia antes | Distância antes | Inteiro depois | Fatia depois | Distância depois | Fatia errada depois |",
           "|---|---|---|---|---|---|---|---|"]
    for n in nomes:
        x, y = antes[n], depois[n]
        out.append(f"| {n} | {x['inteiro']:.2f} | {x['fatia_certa']:.2f} | "
                   f"{x['fatia_certa'] / x['inteiro'] - 1:+.0%} | {y['inteiro']:.2f} | "
                   f"{y['fatia_certa']:.2f} | {y['fatia_certa'] / y['inteiro'] - 1:+.0%} | "
                   f"{y['fatia_errada']:.2f} ({y['outro']}) |")
    out.append(f"\nTempo total: {(time.time() - t0) / 60:.0f} min")
    Path("resultados_ajuste.md").write_text("\n".join(out), encoding="utf-8")
    Path("resultados_ajuste.json").write_text(json.dumps({"antes": antes, "depois": depois}), encoding="utf-8")
    print("\n" + "\n".join(out) + "\n\nSalvo em resultados_ajuste.md")


if __name__ == "__main__":
    main()
