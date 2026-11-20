"""
Motor do "modo cérebro": roda um modelo Mixture-of-Experts (arquitetura qwen3moe: Qwen3-30B,
Qwen3-235B...) direto do arquivo GGUF, especialista por especialista, com memória em 3 níveis.

Cada especialista é um "neurônio" com uma FORÇA DE CONEXÃO:
  * cada vez que o roteador escolhe o especialista, a força dele sobe (reforço, como Hebb);
  * a cada palavra gerada, todas as forças decaem um pouco (esquecimento);
  * os mais fortes moram na placa de vídeo (VRAM, já descomprimidos: memória de trabalho);
  * os médios moram na RAM (comprimidos, prontos pra subir: memória recente);
  * os fracos ficam só no SSD (no próprio arquivo GGUF: memória de longo prazo).
De tempos em tempos o motor reorganiza: conexões são CRIADAS (especialista sobe de nível)
e CORTADAS (especialista desce), sem lista de assuntos e sem texto de calibração.

Modo "fiel": qualquer especialista que o roteador pedir é atendido (de onde estiver), então a
resposta é a mesma do modelo inteiro; só a velocidade depende da memória.
Modo "plástico": o roteador só enxerga as conexões vivas; um especialista desconectado só é
recrutado se o estímulo for forte (roteador muito confiante nele). Mais parecido com um
cérebro (e mais rápido), mas pode mudar a resposta.

Neurônios vitais (super-especialistas): o motor mede o pico de ativação de cada especialista
enquanto roda; quem tiver pico milhares de vezes acima da mediana da camada é marcado como
vital e nunca é cortado.
"""
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gguf import GGUFReader, GGMLQuantizationType as T

from quant import desquantizar

NIVEL_SSD, NIVEL_RAM, NIVEL_VRAM = 0, 1, 2


# ----------------------------------------------------------------------------- arquivo GGUF
class ArquivoGGUF:
    """Abre um GGUF (inclusive dividido em partes -0000N-of-0000M) sem carregar nada na RAM."""

    def __init__(self, caminho):
        caminho = Path(caminho)
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", caminho.name)
        partes = [caminho]
        if m:
            partes = [caminho.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf")
                      for i in range(1, int(m.group(3)) + 1)]
        self.leitores = [GGUFReader(str(p)) for p in partes]
        self.campos = self.leitores[0].fields
        self.tensores = {t.name: t for r in self.leitores for t in r.tensors}
        self.arquivo_de = {t.name: str(p) for p, r in zip(partes, self.leitores) for t in r.tensors}
        self.arq = self.valor("general.architecture")
        self.tamanho = sum(p.stat().st_size for p in partes)

    def valor(self, chave, padrao=None):
        f = self.campos.get(chave)
        if f is None:
            return padrao
        v = f.contents()
        return v

    def hp(self, nome, padrao=None):
        return self.valor(f"{self.arq}.{nome}", padrao)

    def bruto(self, nome):
        t = self.tensores[nome]
        return t.data, T(t.tensor_type)


# ----------------------------------------------------------------------------- pesos densos
class Linear:
    """Uma matriz densa (atenção, saída...). Fica na GPU em fp16 ou comprimida (descomprime a cada uso)."""

    def __init__(self, dados, qtipo, forma, dev, fp16):
        self.forma, self.qtipo = forma, qtipo
        if fp16 or qtipo in (T.F32, T.F16, T.BF16):
            self.w = desquantizar(dados, qtipo, forma, dev, torch.float16 if dev.type == "cuda" else torch.float32)
            self.q = None
        else:
            self.w = None
            self.q = torch.from_numpy(np.ascontiguousarray(dados).view(np.uint8)).to(dev)
        self.dev = dev

    def peso(self):
        if self.w is not None:
            return self.w
        return desquantizar(self.q, self.qtipo, self.forma, self.dev,
                            torch.float16 if self.dev.type == "cuda" else torch.float32)

    def __call__(self, x):
        w = self.peso()
        return (x.to(w.dtype) @ w.T).float()

    def bytes(self):
        return (self.w if self.w is not None else self.q).numel() * (self.w.element_size() if self.w is not None else 1)


def lin_escalado(x, w):
    """Multiplicação em fp16 sem estourar: os super-especialistas geram valores enormes."""
    s = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    return ((x / s).to(w.dtype) @ w.T).float() * s


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ----------------------------------------------------------------------------- a memória (o "cérebro")
class Memoria:
    """
    Três níveis:
      VRAM  - "memória de trabalho": especialistas guardados COMPRIMIDOS num bloco fixo da placa
              (descomprimir na GPU custa quase nada; comprimido cabe ~3x mais que em fp16);
      RAM   - "memória recente": o que foi lido do disco há pouco fica no cache do sistema
              (contabilizado aqui por uma fila LRU, sem cópia extra);
      SSD   - "longo prazo": o próprio arquivo GGUF.
    """

    def __init__(self, mot, cap_vram, ram_gb, tau_curto=256.0, tau_longo=20000.0,
                 arquivo=None, log=print):
        self.m = mot
        L, E = mot.L, mot.E
        self.forca = np.zeros((L, E), np.float32)        # memória de curto prazo (a conversa)
        self.longa = np.zeros((L, E), np.float32)        # memória consolidada (entre conversas)
        self.usos = np.zeros((L, E), np.int64)
        self.vital = np.zeros((L, E), bool)
        self.pico = torch.zeros((L, E), device=mot.dev)  # pico de ativação medido (na GPU, sem sincronizar)
        self.tau_curto, self.tau_longo = tau_curto, tau_longo
        self.arquivo = Path(arquivo) if arquivo else None
        self.log = log
        self.stats = {"vram": 0, "ram": 0, "ssd": 0, "criadas": 0, "cortadas": 0, "recrutados": 0}
        self.eventos = []
        # bloco fixo na VRAM: uma linha de bytes por especialista, para gate/up/down
        self.nb = [max(mot.exp_brutos[l][k][0][0].nbytes for l in range(L)) for k in range(3)]
        self.cap_vram = max(0, int(cap_vram))
        self.pool = [torch.empty((self.cap_vram, nb), dtype=torch.uint8, device=mot.dev) for nb in self.nb]
        self.slot = {}                                     # (l, e) -> linha do bloco
        self.slot_gpu = torch.full((L, E), -1, dtype=torch.long, device=mot.dev)
        self.livres = list(range(self.cap_vram))[::-1]
        self.lru = OrderedDict()                           # o que está no cache de RAM do sistema
        self.cap_ram = max(0, int(ram_gb * 1e9 // max(1, mot.bytes_especialista_q)))
        # área de passagem (memória fixada) pra copiar do disco/RAM pra placa de uma vez
        self.max_lote = 8
        self._arqs = {}
        self._fila = ThreadPoolExecutor(6)
        self._local = threading.local()
        self._anel, self._anel_i = None, 0
        self.passagem = None
        if mot.dev.type == "cuda":
            try:
                self.passagem = [torch.empty(self.max_lote * nb, dtype=torch.uint8).pin_memory() for nb in self.nb]
            except RuntimeError:
                self.passagem = None
            try:
                self._anel = torch.empty((64, 256), dtype=torch.long).pin_memory()
            except RuntimeError:
                self._anel = None
        if self.arquivo and self.arquivo.exists():
            self.lembrar()

    # --- persistência ("sono": consolidação)
    def consolidar(self):
        self.longa = 0.7 * self.longa + 0.3 * (self.forca / max(1e-6, float(self.forca.max())))
        if self.arquivo:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
            self.arquivo.write_text(json.dumps({
                "modelo": self.m.nome, "longa": np.round(self.longa, 4).tolist(),
                "vital": np.argwhere(self.vital).tolist(), "usos": self.usos.tolist()}), encoding="utf-8")

    def lembrar(self):
        try:
            d = json.loads(self.arquivo.read_text(encoding="utf-8"))
            if d.get("modelo") != self.m.nome:
                return
            self.longa = np.array(d["longa"], np.float32)
            self.usos = np.array(d.get("usos", self.usos), np.int64)
            for l, e in d.get("vital", []):
                self.vital[l, e] = True
            self.forca = self.longa.copy() * 4      # começa a conversa com o que já sabia
            self.log(f"Memória consolidada carregada: {int((self.longa > 0.05).sum())} conexões fortes, "
                     f"{int(self.vital.sum())} neurônios vitais")
            self.reorganizar()
        except Exception as e:
            self.log(f"(memória antiga ignorada: {e})")

    def _gpu_idx(self, lista):
        """Lista de inteiros -> tensor na GPU sem travar (memória fixada + cópia assíncrona)."""
        if self._anel is None or len(lista) > self._anel.shape[1]:
            return torch.tensor(lista, dtype=torch.long, device=self.m.dev)
        b = self._anel[self._anel_i]
        self._anel_i = (self._anel_i + 1) % self._anel.shape[0]
        b[:len(lista)] = torch.as_tensor(lista, dtype=torch.long)
        return b[:len(lista)].to(self.m.dev, non_blocking=True)

    @property
    def nivel(self):
        n = np.zeros((self.m.L, self.m.E), np.int8)
        for (l, e) in self.lru:
            n[l, e] = NIVEL_RAM
        for (l, e) in self.slot:
            n[l, e] = NIVEL_VRAM
        return n

    def _do_disco(self, l, lista):
        """Bytes comprimidos de vários especialistas (vindos do cache de RAM ou do SSD) -> GPU."""
        m = len(lista)
        saida = []
        if self.passagem is not None:
            torch.cuda.current_stream().synchronize()          # a passagem pode estar em uso
        for k in range(3):
            dados = self.m.exp_brutos[l][k][0]
            nb = dados[0].nbytes
            caminho, base = self.m.exp_arq[l][k]
            if self.passagem is not None and m <= self.max_lote:
                buf = self.passagem[k][:m * nb].numpy()
            else:
                buf = np.empty(m * nb, np.uint8)
            mv = memoryview(buf)

            def ler(j, e, caminho=caminho, mv=mv, nb=nb, base=base):
                arqs = getattr(self._local, "arqs", None)    # cada thread com seus próprios arquivos abertos
                if arqs is None:
                    arqs = self._local.arqs = {}
                g = arqs.get(caminho)
                if g is None:
                    g = arqs[caminho] = open(caminho, "rb", buffering=0)
                g.seek(base + e * nb)
                g.readinto(mv[j * nb:(j + 1) * nb])
            list(self._fila.map(lambda je: ler(*je), list(enumerate(lista))))
            t = torch.from_numpy(buf).view(m, nb) if self.passagem is None or m > self.max_lote \
                else self.passagem[k][:m * nb].view(m, nb)
            saida.append(t.to(self.m.dev, non_blocking=self.passagem is not None))
        for e in lista:
            k = (l, e)
            if k in self.lru:
                self.stats["ram"] += 1
                self.lru.move_to_end(k)
            else:
                self.stats["ssd"] += 1
                self.lru[k] = True
                if len(self.lru) > self.cap_ram:
                    self.lru.popitem(last=False)
        return saida

    def carregar(self, l, lista, lista_gpu=None):
        """Pesos (gate, up, down) descomprimidos de vários especialistas da camada l, na ordem pedida."""
        m = len(lista)
        pos_disco = [j for j, e in enumerate(lista) if (l, e) not in self.slot]
        self.stats["vram"] += m - len(pos_disco)
        de_disco = self._do_disco(l, [lista[j] for j in pos_disco]) if pos_disco else None
        if lista_gpu is None:
            lista_gpu = self._gpu_idx(lista)
        sl = self.slot_gpu[l][lista_gpu].clamp(min=0)
        pos_gpu = self._gpu_idx(pos_disco) if pos_disco and len(pos_disco) < m else None
        res = []
        for k in range(3):
            dados, qt = self.m.exp_brutos[l][k]
            nb = dados[0].nbytes
            if not pos_disco:
                linhas = self.pool[k][sl, :nb]
            elif len(pos_disco) == m:
                linhas = de_disco[k]
            else:
                linhas = self.pool[k][sl, :nb]
                linhas.index_copy_(0, pos_gpu, de_disco[k])
            forma = (self.m.ff, self.m.d) if k < 2 else (self.m.d, self.m.ff)
            res.append(desquantizar(linhas, qt, (m * forma[0], forma[1]), self.m.dev, self.m.cdt)
                       .view(m, *forma))
        return res

    # --- aprendizado (reforço e esquecimento)
    def reforcar(self, l, idx, pesos):
        np.add.at(self.forca[l], idx, pesos)
        np.add.at(self.usos[l], idx, 1)

    def esquecer(self, n_tokens):
        self.forca *= math.exp(-n_tokens / self.tau_curto)
        self.longa *= math.exp(-n_tokens / self.tau_longo)

    def reorganizar(self):
        """Cria e corta conexões: os mais fortes sobem pra VRAM, os fracos descem."""
        L, E = self.m.L, self.m.E
        picos = self.pico.cpu().numpy()
        for l in range(L):
            p = picos[l]
            usados = p > 0
            if usados.sum() >= 4:
                ref = np.median(p[usados])
                if ref > 0:
                    novos = (p / ref > self.m.limiar_vital) & ~self.vital[l]
                    for e in np.where(novos)[0]:
                        self.vital[l, e] = True
                        self.log(f"Neurônio vital detectado: camada {l}, especialista {e} "
                                 f"(pico {p[e] / ref:,.0f}x a mediana)")
        self.eventos = []
        if not self.cap_vram:
            return
        pont = self.forca + 0.25 * self.longa + self.vital * 1e6
        ordem = np.argsort(pont, axis=None)[::-1][:self.cap_vram]
        alvo = {(int(i // E), int(i % E)) for i in ordem if pont.flat[i] > 0}
        # histerese: só troca quem está claramente mais forte que o mais fraco da VRAM
        for k in list(self.slot):
            if k not in alvo and not self.vital[k]:
                self.livres.append(self.slot.pop(k))
                self.slot_gpu[k[0], k[1]] = -1
                self.stats["cortadas"] += 1
                self.eventos.append((k[0], k[1], -1))
        novos = sorted(alvo - set(self.slot))
        por_camada = {}
        for l, e in novos:
            por_camada.setdefault(l, []).append(e)
        for l, es in por_camada.items():
            for i in range(0, len(es), self.max_lote):
                lote = es[i:i + self.max_lote]
                bytes_ = self._do_disco(l, lote)
                for j, e in enumerate(lote):
                    if not self.livres:
                        break
                    s = self.livres.pop()
                    for k in range(3):
                        self.pool[k][s, :bytes_[k].shape[1]] = bytes_[k][j]
                    self.slot[(l, e)] = s
                    self.slot_gpu[l, e] = s
                    self.stats["criadas"] += 1
                    self.eventos.append((l, e, +1))

    def conectado(self, l):
        """Máscara (na GPU) dos especialistas com conexão viva na camada l, pro modo plástico."""
        viva = self.vital[l] | (self.forca[l] > 0.05)
        for (ll, e) in self.slot:
            if ll == l:
                viva[e] = True
        return torch.from_numpy(viva).to(self.m.dev)

    def taxa(self):
        t = max(1, self.stats["vram"] + self.stats["ram"] + self.stats["ssd"])
        return {k: self.stats[k] / t for k in ("vram", "ram", "ssd")}


# ----------------------------------------------------------------------------- o modelo
class Motor:
    def __init__(self, gguf_caminho, device=None, vram_gb=None, ram_gb=12.0, ctx=4096,
                 memoria=None, modo="fiel", log=print):
        self.log = log
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cdt = torch.float16 if self.dev.type == "cuda" else torch.float32
        g = self.g = ArquivoGGUF(gguf_caminho)
        if g.arq != "qwen3moe":
            raise SystemExit(f"Arquitetura {g.arq} ainda não suportada pelo modo cérebro (use Qwen3 MoE).")
        self.nome = Path(gguf_caminho).name
        self.L = int(g.hp("block_count"))
        self.d = int(g.hp("embedding_length"))
        self.nh = int(g.hp("attention.head_count"))
        self.nkv = int(g.hp("attention.head_count_kv"))
        self.hd = int(g.hp("attention.key_length", self.d // self.nh))
        self.E = int(g.hp("expert_count"))
        self.k = int(g.hp("expert_used_count"))
        self.ff = int(g.hp("expert_feed_forward_length"))
        self.eps = float(g.hp("attention.layer_norm_rms_epsilon", 1e-6))
        self.theta = float(g.hp("rope.freq_base", 1e6))
        self.ctx = ctx
        self.modo = modo
        self.limiar_vital = 1000.0
        self.recrutar = 0.3          # modo plástico: probabilidade mínima pra criar conexão nova
        log(f"Modelo {self.nome}: {self.L} camadas, {self.E} especialistas/camada ({self.k} ativos), "
            f"dim {self.d}, {g.tamanho / 1e9:.1f} GB em disco, dispositivo {self.dev}")

        # orçamento da placa
        if vram_gb is None:
            vram_gb = (torch.cuda.mem_get_info()[0] / 1e9 - 0.6) if self.dev.type == "cuda" else 4.0
        t0 = time.time()
        attn, saida = self._tamanho_denso_fp16()
        denso_fp16 = attn < 0.45 * vram_gb * 1e9
        saida_fp16 = denso_fp16 and attn + saida < 0.55 * vram_gb * 1e9
        self._carregar_denso(denso_fp16, saida_fp16)
        usado = sum(x.bytes() for x in self._lineares()) + self.kv_bytes()
        self.kv_alocar()
        reserva = 1.2e9 if self.dev.type == "cuda" else 0.05e9     # contas temporárias
        sobra = vram_gb * 1e9 - usado - reserva
        vram_esp = max(0, int(sobra // self.bytes_especialista_q))
        log(f"Pesos densos {'em fp16' if denso_fp16 else 'comprimidos'} na {self.dev}: {usado / 1e9:.2f} GB; "
            f"cabem {vram_esp} especialistas (comprimidos) na VRAM ({vram_esp / (self.L * self.E):.1%}). "
            f"Carregou em {time.time() - t0:.0f} s")
        self.mem = Memoria(self, vram_esp, ram_gb, arquivo=memoria, log=log)
        log(f"Memória recente (cache de RAM): até {self.mem.cap_ram} especialistas "
            f"({min(1, self.mem.cap_ram / (self.L * self.E)):.0%} do total)")

    # --- carregamento
    def _tamanho_denso_fp16(self):
        attn = saida = 0
        for n, t in self.g.tensores.items():
            b = int(np.prod([int(x) for x in t.shape])) * 2
            if n == "output.weight":
                saida = b
            elif "_exps" not in n and n != "token_embd.weight":
                attn += b
        return attn, saida

    def _lin(self, nome, fp16):
        dados, qt = self.g.bruto(nome)
        ne = [int(x) for x in self.g.tensores[nome].shape]
        return Linear(dados, qt, (ne[1], ne[0]), self.dev, fp16)

    def _vetor(self, nome):
        dados, qt = self.g.bruto(nome)
        n = int(np.prod([int(x) for x in self.g.tensores[nome].shape]))
        return desquantizar(dados, qt, (n,), self.dev, torch.float32)

    def _carregar_denso(self, fp16, saida_fp16):
        self.emb_dados, self.emb_qt = self.g.bruto("token_embd.weight")
        self.vocab = int(self.g.tensores["token_embd.weight"].shape[1])
        self.norm_saida = self._vetor("output_norm.weight")
        nome_saida = "output.weight" if "output.weight" in self.g.tensores else "token_embd.weight"
        self.saida = self._lin(nome_saida, saida_fp16)
        self.cam = []
        self.exp_brutos = []
        self.exp_arq = []
        for l in range(self.L):
            p = f"blk.{l}."
            c = {"attn_norm": self._vetor(p + "attn_norm.weight"),
                 "ffn_norm": self._vetor(p + "ffn_norm.weight"),
                 "q": self._lin(p + "attn_q.weight", fp16), "k": self._lin(p + "attn_k.weight", fp16),
                 "v": self._lin(p + "attn_v.weight", fp16), "o": self._lin(p + "attn_output.weight", fp16),
                 "roteador": self._vetor(p + "ffn_gate_inp.weight").reshape(self.E, self.d)}
            if p + "attn_q_norm.weight" in self.g.tensores:
                c["q_norm"] = self._vetor(p + "attn_q_norm.weight")
                c["k_norm"] = self._vetor(p + "attn_k_norm.weight")
            self.cam.append(c)
            self.exp_brutos.append([self.g.bruto(p + f"ffn_{k}_exps.weight") for k in ("gate", "up", "down")])
            self.exp_arq.append([(self.g.arquivo_de[n], int(self.g.tensores[n].data_offset))
                                 for n in (p + f"ffn_{k}_exps.weight" for k in ("gate", "up", "down"))])
        self.bytes_especialista_q = sum(b[0][0].nbytes for b in self.exp_brutos[0])

    def _lineares(self):
        yield self.saida
        for c in self.cam:
            for k in ("q", "k", "v", "o", "qkv"):
                if k in c:
                    yield c[k]

    def kv_bytes(self):
        return 2 * self.L * self.ctx * self.nkv * self.hd * (2 if self.dev.type == "cuda" else 4)

    def kv_alocar(self):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=self.dev, dtype=torch.float32) / self.hd))
        ang = torch.outer(torch.arange(self.ctx, device=self.dev, dtype=torch.float32), inv)
        self.cos, self.sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        try:
            F.scaled_dot_product_attention(torch.zeros(4, 1, 8, device=self.dev, dtype=self.cdt),
                                           torch.zeros(2, 1, 8, device=self.dev, dtype=self.cdt),
                                           torch.zeros(2, 1, 8, device=self.dev, dtype=self.cdt), enable_gqa=True)
            self.gqa = True
        except Exception:
            self.gqa = False
        for c in self.cam:                                  # q, k, v numa matriz só (1 conta em vez de 3)
            if all(c[x].w is not None for x in "qkv"):
                fus = Linear.__new__(Linear)
                fus.w, fus.q, fus.dev = torch.cat([c["q"].w, c["k"].w, c["v"].w]), None, self.dev
                c["qkv"] = fus
                del c["q"], c["k"], c["v"]
        self.kc = torch.zeros(self.L, self.ctx, self.nkv, self.hd, device=self.dev, dtype=self.cdt)
        self.vc = torch.zeros_like(self.kc)
        self.pos = 0

    # --- partes do transformer
    def embed(self, ids):
        linhas = self.emb_dados[np.asarray(ids)]
        return desquantizar(linhas, self.emb_qt, (len(ids), self.d), self.dev, torch.float32)

    def rope(self, x, pos0):
        n = x.shape[0]
        cos, sin = self.cos[pos0:pos0 + n], self.sin[pos0:pos0 + n]
        a, b = x[..., : self.hd // 2], x[..., self.hd // 2:]
        return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)

    def atencao(self, l, h):
        c = self.cam[l]
        n = h.shape[0]
        if "qkv" in c:
            qkv = c["qkv"](h)
            nq, nk = self.nh * self.hd, self.nkv * self.hd
            q, k, v = qkv[:, :nq], qkv[:, nq:nq + nk], qkv[:, nq + nk:]
        else:
            q, k, v = c["q"](h), c["k"](h), c["v"](h)
        q, k, v = q.view(n, self.nh, self.hd), k.view(n, self.nkv, self.hd), v.view(n, self.nkv, self.hd)
        if "q_norm" in c:
            q, k = rms(q, c["q_norm"], self.eps), rms(k, c["k_norm"], self.eps)
        q, k = self.rope(q, self.pos), self.rope(k, self.pos)
        p0, p1 = self.pos, self.pos + n
        self.kc[l, p0:p1], self.vc[l, p0:p1] = k.to(self.cdt), v.to(self.cdt)
        K, V = self.kc[l, :p1].transpose(0, 1), self.vc[l, :p1].transpose(0, 1)     # (nkv, p1, hd)
        Q = q.to(self.cdt).transpose(0, 1)                                       # (nh, n, hd)
        mask = None if n == 1 else torch.ones(n, p1, dtype=torch.bool, device=self.dev).tril(diagonal=p0)
        if self.gqa:
            o = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, enable_gqa=True)
        else:
            rep = self.nh // self.nkv
            o = F.scaled_dot_product_attention(Q, K.repeat_interleave(rep, 0), V.repeat_interleave(rep, 0),
                                               attn_mask=mask)
        return c["o"](o.transpose(0, 1).reshape(n, self.nh * self.hd).float())

    def moe(self, l, h, ativos):
        c = self.cam[l]
        logits = h @ c["roteador"].T                           # (n, E)
        if self.modo == "plastico":
            vivo = self.mem.conectado(l)
            prob_tudo = logits.softmax(-1)
            forte = prob_tudo > self.recrutar
            novos = (forte & ~vivo).any(0)
            if bool(novos.any()):                              # estímulo forte: conexão nova
                self.mem.stats["recrutados"] += int(novos.sum())
                vivo = vivo | novos
            logits = logits.masked_fill(~vivo, float("-inf"))
        prob = logits.softmax(-1)
        w, idx = prob.topk(self.k, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        par = torch.cat([idx.float(), w.float()], dim=-1).cpu().numpy()   # 1 sincronização por camada
        idx_c, w_c = par[:, :self.k].astype(np.int64), par[:, self.k:]
        if h.shape[0] == 1:                                   # gerando: os 8 especialistas de uma vez
            G, U, D = self.mem.carregar(l, [int(e) for e in idx_c[0]], idx[0])
            y = self._lote(h, G, U, D)                        # (k, d)
            self.mem.pico[l, idx[0]] = torch.maximum(self.mem.pico[l, idx[0]], y.abs().amax(-1))
            saida = (y * w[0][:, None]).sum(0, keepdim=True)
        else:                                                 # lendo a pergunta: agrupa por especialista
            saida = torch.zeros_like(h)
            unicos = [int(e) for e in np.unique(idx_c)]
            for i in range(0, len(unicos), self.mem.max_lote):
                lote = unicos[i:i + self.mem.max_lote]
                G, U, D = self.mem.carregar(l, lote)
                for j, e in enumerate(lote):
                    linhas, col = np.nonzero(idx_c == e)
                    x = h[linhas]
                    a = F.silu((x.to(G.dtype) @ G[j].T).float()) * (x.to(U.dtype) @ U[j].T).float()
                    y = lin_escalado(a, D[j])
                    self.mem.pico[l, e] = torch.maximum(self.mem.pico[l, e], y.abs().amax())
                    saida.index_add_(0, torch.from_numpy(linhas).to(self.dev),
                                     y * torch.from_numpy(w_c[linhas, col]).to(self.dev)[:, None])
        self.mem.reforcar(l, idx_c.ravel(), w_c.ravel())
        ativos.append(idx_c[-1].tolist())
        return saida

    def _lote(self, x, G, U, D):
        """1 palavra x vários especialistas em lote. x: (1, d); G,U: (k, ff, d); D: (k, d, ff)."""
        xc = x.to(G.dtype).T                                  # (d, 1)
        a = F.silu(torch.matmul(G, xc).float()) * torch.matmul(U, xc).float()   # (k, ff, 1)
        s = a.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        return (torch.matmul(D, (a / s).to(D.dtype)).float() * s).squeeze(-1)    # (k, d)

    @torch.no_grad()
    def passo(self, ids):
        """Processa uma lista de tokens (a pergunta inteira ou 1 token) e devolve os logits do último."""
        if self.pos + len(ids) > self.ctx:
            raise RuntimeError("acabou o contexto (aumente --ctx ou comece outra conversa)")
        h = self.embed(ids)
        ativos = []
        for l in range(self.L):
            c = self.cam[l]
            h = h + self.atencao(l, rms(h, c["attn_norm"], self.eps))
            h = h + self.moe(l, rms(h, c["ffn_norm"], self.eps), ativos)
        self.pos += len(ids)
        self.mem.esquecer(len(ids))
        x = rms(h[-1:], self.norm_saida, self.eps)
        return self.saida(x)[0], ativos

    def reiniciar(self):
        self.pos = 0
