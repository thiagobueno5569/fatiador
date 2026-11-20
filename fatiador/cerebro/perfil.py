"""Mede onde o tempo vai (atenção, roteador, busca de especialistas, contas) no modelo de verdade."""
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motor as M  # noqa: E402
from conversa import Cerebro  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


def cronometrar(obj, nome, rotulo):
    orig = getattr(obj, nome)

    def f(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        T[rotulo] += time.perf_counter() - t
        N[rotulo] += 1
        return r
    setattr(obj, nome, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--tokens", type=int, default=40)
    a = ap.parse_args()
    m = M.Motor(a.modelo, ram_gb=10)
    c = Cerebro(m)
    linhas = []
    for rodada in range(2):
        T.clear(); N.clear()
        cronometrar(m, "atencao", "atenção (densa)")
        cronometrar(m, "_lote", "contas dos especialistas")
        cronometrar(m.mem, "carregar", "carregar especialistas (total)")
        cronometrar(m.mem, "_do_disco", "  ...dos quais: ler da RAM/SSD e enviar")
        cronometrar(m.mem, "reorganizar", "reorganizar conexões")
        t0 = time.perf_counter()
        txt, info = c.responder("Explique em poucas palavras o que é uma derivada.", max_tokens=a.tokens, temperatura=0)
        tot = time.perf_counter() - t0
        linhas.append(f"\nRodada {rodada + 1}: {info['tokens']} tokens, {info['tok_s']:.2f} tok/s, leitura da pergunta "
                      f"{info['leitura_s']:.1f} s, total {tot:.1f} s | acertos {m.mem.taxa()}")
        for k in sorted(T, key=T.get, reverse=True):
            linhas.append(f"  {k:42s} {T[k]:7.2f} s  ({N[k]} chamadas, {1000 * T[k] / max(1, N[k]):.2f} ms cada)")
        # desfaz os cronômetros
        for obj, nome in [(m, "atencao"), (m, "_lote"), (m.mem, "carregar"), (m.mem, "_do_disco"),
                          (m.mem, "reorganizar")]:
            if nome in obj.__dict__:
                del obj.__dict__[nome]
    out = "\n".join(linhas)
    print(out)
    (Path(__file__).resolve().parent / "perfil.txt").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
