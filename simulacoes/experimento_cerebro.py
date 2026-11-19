"""
Experimento 2 — "cérebro": núcleo fixo + blocos de assunto que nascem, se fortalecem
com o uso e vão esfriando quando são esquecidos.

Ideia do Thiago:
  * Núcleo comum (linguagem/raciocínio) fica sempre na placa e nunca é podado.
  * Cada assunto tem um bloco próprio. Usar o bloco = sinapse forte; não usar = ela
    enfraquece com o tempo (força decai todo dia).
  * A força decide onde o bloco mora, como a cache de um processador:
        VRAM (quente, 16 bits)  ->  RAM (morna, 8 bits)  ->  SSD (fria, 4 e depois 2 bits)
    Quanto mais tempo frio, mais bits ele perde = o "esquecimento". Nada é apagado.
  * Quando o assunto volta, o bloco sobe pra placa e reaprende usando as conversas.

Mede: memória usada na placa, tempo de carregar blocos, acurácia ao longo do tempo
e quanto mais rápido é "relembrar" do que aprender do zero.
"""
import numpy as np

import sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
D, C, K = 64, 4, 8
NOMES = ["eletrônica", "cálculo", "programação", "física",
         "culinária", "futebol", "música", "história"]

# ---- escala "real" pra contar memória (ilustrativa) ----------------------------
GB_NUCLEO, GB_BLOCO_16BIT = 2.0, 1.5        # tamanho em 16 bits
VAGAS_VRAM, VAGAS_RAM = 2, 3                # quantos blocos cabem em cada andar
BANDA = {"RAM": 12.0, "SSD": 3.5}           # GB/s até a placa (PCIe / NVMe)
DECAI = 0.97                                # força multiplica por isso todo dia


# ---------------------------------------------------------------- dados
def gerar_assunto():
    centro = rng.normal(size=D); centro *= 6 / np.linalg.norm(centro)
    base = np.linalg.qr(rng.normal(size=(D, 8)))[0]
    regra = rng.normal(size=(C, D))
    return centro, base, regra


def amostrar(a, n):
    centro, base, regra = a
    x = centro + rng.normal(size=(n, 8)) @ base.T + 0.05 * rng.normal(size=(n, D))
    return x, np.argmax((x - centro) @ regra.T, axis=1)


ASSUNTOS = [gerar_assunto() for _ in range(K)]
TESTE = [amostrar(a, 600) for a in ASSUNTOS]

# ---------------------------------------------------------------- núcleo fixo
# Camada de "características gerais", igual pra todos os assuntos, congelada.
W_NUCLEO = rng.normal(size=(D, 256)) / np.sqrt(D)


def nucleo(x):
    return np.maximum(0, x @ W_NUCLEO)


# ---------------------------------------------------------------- bloco de assunto
class Bloco:
    def __init__(self, entrada=256, h=48):
        self.p = {"W1": rng.normal(size=(entrada, h)) * np.sqrt(2 / entrada), "b1": np.zeros(h),
                  "W2": rng.normal(size=(h, C)) * np.sqrt(2 / h), "b2": np.zeros(C)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t, self.bits = 0, 16

    def forward(self, f):
        self.f = f
        self.a = np.maximum(0, f @ self.p["W1"] + self.p["b1"])
        return self.a @ self.p["W2"] + self.p["b2"]

    def passo(self, f, y, lr=2e-3):
        z = self.forward(f)
        pr = np.exp(z - z.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        pr[np.arange(len(y)), y] -= 1; g_o = pr / len(y)
        g = {"W2": self.a.T @ g_o, "b2": g_o.sum(0)}
        g_a = (g_o @ self.p["W2"].T) * (self.a > 0)
        g["W1"], g["b1"] = self.f.T @ g_a, g_a.sum(0)
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            self.p[k] -= lr * (self.m[k] / (1 - 0.9 ** self.t)) / (
                np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        self.bits = 16                       # treinou -> volta a ter precisão total

    def prever(self, f):
        return self.forward(f).argmax(1)

    def perder_bits(self, bits):
        """Esquecimento: guarda os pesos com menos bits. Perda permanente."""
        if bits >= self.bits:
            return
        for k in ("W1", "W2"):
            w = self.p[k]
            niveis = 2 ** (bits - 1) - 1
            escala = np.abs(w).max() / niveis
            self.p[k] = np.round(w / escala) * escala
        self.bits = bits

    def gb(self):
        return GB_BLOCO_16BIT * self.bits / 16


def acc(bloco, k):
    x, y = TESTE[k]
    return float((bloco.prever(nucleo(x)) == y).mean())


def treinar(bloco, x, y, epocas):
    f = nucleo(x)
    for _ in range(epocas):
        idx = rng.permutation(len(x))
        for i in range(0, len(x), 64):
            j = idx[i:i + 64]; bloco.passo(f[j], y[j])


# ---------------------------------------------------------------- o "cérebro"
class Cerebro:
    def __init__(self, limiar=6.5):
        self.limiar = limiar
        self.centros, self.n, self.blocos = [], [], []
        self.forca, self.ultimo_uso, self.andar = [], [], []

    def rotear(self, x):
        """Vetor de assunto da conversa = média das mensagens. Acha o mais próximo."""
        v = x.mean(0)
        if self.centros:
            d = np.linalg.norm(np.array(self.centros) - v, axis=1)
            k = int(d.argmin())
            if d[k] <= self.limiar:
                return k, False
        # assunto novo: nasce um vetor e um bloco
        self.centros.append(v.copy()); self.n.append(0); self.blocos.append(Bloco())
        self.forca.append(0.0); self.ultimo_uso.append(0); self.andar.append("VRAM")
        return len(self.blocos) - 1, True

    def reorganizar(self, dia):
        """Ranking por força decide o andar; frio há muito tempo perde mais bits."""
        ordem = np.argsort(self.forca)[::-1]
        for pos, k in enumerate(ordem):
            if pos < VAGAS_VRAM:
                self.andar[k] = "VRAM"
            elif pos < VAGAS_VRAM + VAGAS_RAM:
                self.andar[k] = "RAM"; self.blocos[k].perder_bits(8)
            else:
                self.andar[k] = "SSD"
                frio = dia - self.ultimo_uso[k]
                self.blocos[k].perder_bits(2 if frio > 90 else 4)

    def conversa(self, dia, x, y):
        k, novo = self.rotear(x)
        latencia = 0.0
        if not novo and self.andar[k] != "VRAM":
            latencia = self.blocos[k].gb() / BANDA[self.andar[k]]
        andar_antes = "novo" if novo else self.andar[k]
        # responde primeiro (é isso que o usuário vê), depois aprende com a conversa
        acerto = float((self.blocos[k].prever(nucleo(x)) == y).mean())
        treinar(self.blocos[k], x, y, epocas=40 if novo else 3)
        self.n[k] += 1
        self.centros[k] += (x.mean(0) - self.centros[k]) / (self.n[k] + 1)
        # sinapses: todo mundo enfraquece, quem foi usado fortalece
        self.forca = [f * DECAI for f in self.forca]
        self.forca[k] += 1.0; self.ultimo_uso[k] = dia
        self.andar[k] = "VRAM"
        self.reorganizar(dia)
        return k, andar_antes, latencia, acerto


# ---------------------------------------------------------------- a "vida" do usuário
def agenda(dia):
    """Interesses mudam com o tempo, como numa pessoa real."""
    if dia < 120:                       # semestre de Elétrica
        return rng.choice([0, 1, 2])
    if dia < 240:                       # férias: física, culinária, futebol
        return 0 if rng.random() < 0.05 else rng.choice([3, 4, 5])
    if dia < 320:                       # música e história
        return rng.choice([6, 7])
    return rng.choice([1, 6])           # cálculo volta depois de 200 dias esquecido


cerebro = Cerebro()
log = []
for dia in range(400):
    t = agenda(dia)
    x, y = amostrar(ASSUNTOS[t], 400 if dia < 1 else 120)
    k, andar, lat, acerto = cerebro.conversa(dia, x, y)
    log.append((dia, t, k, andar, lat, acerto))

# ---------------------------------------------------------------- relatório
print("=" * 70)
print("O CÉREBRO DEPOIS DE 400 DIAS")
print("=" * 70)
print(f"Blocos criados sozinhos: {len(cerebro.blocos)} (assuntos reais: {K})\n")
print(f"{'bloco':<14}{'força':>7}{'andar':>7}{'bits':>6}{'tamanho':>9}{'acurácia':>10}")
mapa = {}
for (_, t, k, *_r) in log:
    mapa.setdefault(k, t)
for k in range(len(cerebro.blocos)):
    b = cerebro.blocos[k]
    print(f"{NOMES[mapa[k]]:<14}{cerebro.forca[k]:7.2f}{cerebro.andar[k]:>7}"
          f"{b.bits:6d}{b.gb():8.2f}G{acc(b, mapa[k]):10.0%}")

vram = GB_NUCLEO + sum(b.gb() for b, a in zip(cerebro.blocos, cerebro.andar) if a == "VRAM")
tudo = GB_NUCLEO + GB_BLOCO_16BIT * len(cerebro.blocos)
guardado = GB_NUCLEO + sum(b.gb() for b in cerebro.blocos)
print(f"\nModelo inteiro em 16 bits:     {tudo:5.1f} GB")
print(f"Ocupando a placa (VRAM):       {vram:5.1f} GB  (cabe numa RTX 2060 de 6 GB)")
print(f"Total guardado em todo lugar:  {guardado:5.1f} GB  (esquecimento comprime)")

dias = [l for l in log if l[0] > 0]
for andar in ["VRAM", "RAM", "SSD"]:
    sel = [l for l in dias if l[3] == andar]
    if sel:
        print(f"Conversas com bloco já em {andar:4s}: {len(sel):3d}  "
              f"tempo médio pra carregar {np.mean([l[4] for l in sel]):.2f} s")

print("\n" + "=" * 70)
print("O RETORNO DO CÁLCULO (esquecido por ~200 dias, guardado em 2 bits)")
print("=" * 70)
volta = [l for l in log if l[1] == 1 and l[0] >= 320][:6]
for dia, _, _, andar, lat, acerto in volta:
    print(f"dia {dia}: veio de {andar:4s} (carregou em {lat:.2f}s)  acurácia na conversa {acerto:.0%}")

# ---- relembrar x aprender do zero ------------------------------------------------
print("\n" + "=" * 70)
print("RELEMBRAR É MAIS RÁPIDO QUE APRENDER DO ZERO?")
print("=" * 70)
k_calc = mapa_inv = {v: k for k, v in mapa.items()}[1]
x_r, y_r = amostrar(ASSUNTOS[1], 2000)
alvo = 0.90


def passos_ate(bloco, alvo):
    f = nucleo(x_r)
    for passo in range(1, 3001):
        j = rng.integers(0, len(x_r), 64); bloco.passo(f[j], y_r[j])
        if passo % 10 == 0 and acc(bloco, 1) >= alvo:
            return passo
    return None


velho = Bloco(); treinar(velho, *amostrar(ASSUNTOS[1], 3000), epocas=15)
ref = acc(velho, 1)
velho.perder_bits(2)
esquecido = acc(velho, 1)
p_relembrar = passos_ate(velho, alvo)
p_zero = passos_ate(Bloco(), alvo)
print(f"Bloco treinado: {ref:.0%}  ->  depois de esquecer (2 bits): {esquecido:.0%}")
print(f"Passos de treino até {alvo:.0%}:  relembrando {p_relembrar}   do zero {p_zero}")
