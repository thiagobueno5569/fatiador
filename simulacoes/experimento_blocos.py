"""
Experimento: matriz dividida em blocos por assunto (ideia do Thiago)
vs. rede densa que lê a matriz inteira sempre.

- Cada "assunto" vive numa região própria do espaço de entrada (um subespaço deslocado).
- Modelo DENSO: uma MLP grande, usa todos os pesos pra toda entrada.
- Modelo em BLOCOS: K blocos menores (mesmo total de parâmetros).
  Um roteador compara a entrada com "vetores de assunto" (centróides) e lê SÓ o bloco
  mais próximo. Os vetores de assunto são criados sozinhos, sem rótulo: se uma entrada
  está longe de todos os centróides, nasce um vetor novo + um bloco novo.

Dois testes:
  1) Treino misturado (todos os assuntos juntos): qualidade e custo por inferência.
  2) Treino em sequência (um assunto de cada vez): quem esquece o que aprendeu antes?
"""
import time
import numpy as np

rng = np.random.default_rng(0)

D, K, C = 64, 8, 4          # dimensão da entrada, nº de assuntos, nº de classes
H_TOTAL = 256               # neurônios ocultos totais (denso) = K blocos de 32
N_POR_ASSUNTO = 3000


# ---------------------------------------------------------------- dados
def gerar_assuntos():
    assuntos = []
    for _ in range(K):
        centro = rng.normal(size=D)
        centro *= 6 / np.linalg.norm(centro)
        base = np.linalg.qr(rng.normal(size=(D, 8)))[0]       # subespaço de dim 8
        regra = rng.normal(size=(C, D))                        # regra própria do assunto
        assuntos.append((centro, base, regra))
    return assuntos


def amostrar(assunto, n):
    centro, base, regra = assunto
    x = centro + rng.normal(size=(n, 8)) @ base.T + 0.05 * rng.normal(size=(n, D))
    y = np.argmax((x - centro) @ regra.T, axis=1)
    return x, y


ASSUNTOS = gerar_assuntos()
treino = [amostrar(a, N_POR_ASSUNTO) for a in ASSUNTOS]
teste = [amostrar(a, 1000) for a in ASSUNTOS]


# ---------------------------------------------------------------- MLP simples com Adam
class MLP:
    def __init__(self, h):
        self.p = {
            "W1": rng.normal(size=(D, h)) * np.sqrt(2 / D), "b1": np.zeros(h),
            "W2": rng.normal(size=(h, C)) * np.sqrt(2 / h), "b2": np.zeros(C),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        self.x = x
        self.a = np.maximum(0, x @ self.p["W1"] + self.p["b1"])
        return self.a @ self.p["W2"] + self.p["b2"]

    def passo(self, x, y, lr=2e-3):
        logits = self.forward(x)
        prob = np.exp(logits - logits.max(1, keepdims=True))
        prob /= prob.sum(1, keepdims=True)
        prob[np.arange(len(y)), y] -= 1
        g_out = prob / len(y)
        g = {"W2": self.a.T @ g_out, "b2": g_out.sum(0)}
        g_a = (g_out @ self.p["W2"].T) * (self.a > 0)
        g["W1"], g["b1"] = self.x.T @ g_a, g_a.sum(0)
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def prever(self, x):
        return self.forward(x).argmax(1)


def treinar(modelo, x, y, epocas=15, lote=64):
    for _ in range(epocas):
        idx = rng.permutation(len(x))
        for i in range(0, len(x), lote):
            j = idx[i:i + lote]
            modelo.passo(x[j], y[j])


# ---------------------------------------------------------------- modelo em blocos
class ModeloBlocos:
    """Vetores de assunto (centróides) apontam para blocos da matriz."""

    def __init__(self, h_bloco, limiar):
        self.h, self.limiar = h_bloco, limiar
        self.centros, self.contagem, self.blocos = [], [], []

    def rotear(self, x, criar=False):
        rotas = np.empty(len(x), dtype=int)
        for i, xi in enumerate(x):
            if self.centros:
                dist = np.linalg.norm(np.array(self.centros) - xi, axis=1)
                k = int(dist.argmin())
            if not self.centros or (criar and dist[k] > self.limiar):
                # memória nova -> nasce um vetor de assunto e um bloco novo
                self.centros.append(xi.copy()); self.contagem.append(1)
                self.blocos.append(MLP(self.h)); k = len(self.centros) - 1
            elif criar:
                # consolida: o vetor de assunto anda em direção à entrada
                self.contagem[k] += 1
                self.centros[k] += (xi - self.centros[k]) / self.contagem[k]
            rotas[i] = k
        return rotas

    def treinar(self, x, y, epocas=15):
        rotas = self.rotear(x, criar=True)
        for k in np.unique(rotas):
            treinar(self.blocos[k], x[rotas == k], y[rotas == k], epocas)

    def prever(self, x):
        rotas = self.rotear(x)
        saida = np.empty(len(x), dtype=int)
        for k in np.unique(rotas):
            saida[rotas == k] = self.blocos[k].prever(x[rotas == k])
        return saida


def acuracia(modelo, conjuntos):
    return [float((modelo.prever(x) == y).mean()) for x, y in conjuntos]


def custo_por_amostra(modelo):
    """Multiplicações por inferência."""
    if isinstance(modelo, MLP):
        h = modelo.p["W1"].shape[1]
        return D * h + h * C
    n = len(modelo.centros)
    return n * D + D * modelo.h + modelo.h * C      # roteador + 1 bloco


def tempo_inferencia(modelo, x, rep=20):
    modelo.prever(x)
    t0 = time.perf_counter()
    for _ in range(rep):
        modelo.prever(x)
    return (time.perf_counter() - t0) / rep / len(x) * 1e6   # µs por amostra


# ================================================================ TESTE 1: tudo misturado
x_all = np.concatenate([t[0] for t in treino]); y_all = np.concatenate([t[1] for t in treino])
x_te = np.concatenate([t[0] for t in teste])

denso = MLP(H_TOTAL)
treinar(denso, x_all, y_all)
blocos = ModeloBlocos(H_TOTAL // K, limiar=6.5)
blocos.treinar(x_all, y_all)

print("=" * 64)
print("TESTE 1 — todos os assuntos treinados juntos")
print("=" * 64)
for nome, m in [("Denso (matriz inteira)", denso), ("Blocos por assunto", blocos)]:
    acc = np.mean(acuracia(m, teste))
    print(f"{nome:24s} acurácia {acc:6.1%}   multiplicações/inferência {custo_por_amostra(m):6d}")
print(f"Vetores de assunto criados sozinhos: {len(blocos.centros)} (assuntos reais: {K})")
print(f"Economia teórica de contas: {custo_por_amostra(denso) / custo_por_amostra(blocos):.1f}x")

# ================================================================ TESTE 2: em sequência
denso_seq = MLP(H_TOTAL)
blocos_seq = ModeloBlocos(H_TOTAL // K, limiar=6.5)
for x, y in treino:                  # aprende um assunto de cada vez
    treinar(denso_seq, x, y)
    blocos_seq.treinar(x, y)

print()
print("=" * 64)
print("TESTE 2 — assuntos aprendidos um de cada vez (memória nova por cima)")
print("=" * 64)
ad, ab = acuracia(denso_seq, teste), acuracia(blocos_seq, teste)
print("assunto:  " + "  ".join(f"{i + 1:>4d}" for i in range(K)))
print("denso:    " + "  ".join(f"{a:4.0%}" for a in ad) + f"   média {np.mean(ad):.0%}")
print("blocos:   " + "  ".join(f"{a:4.0%}" for a in ab) + f"   média {np.mean(ab):.0%}")
