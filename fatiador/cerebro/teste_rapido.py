"""
Teste do modo cérebro no modelo de verdade: faz perguntas de assuntos diferentes, no modo fiel
e no plástico, e grava respostas, velocidade e o movimento das conexões em teste_cerebro.md.
"""
import argparse
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from motor import Motor        # noqa: E402
from conversa import Cerebro   # noqa: E402

PERGUNTAS = ["Qual a derivada de e^(2x - 4y) em relação a x? Mostre a conta.",
             "Quanto vale a integral de x² de 0 a 3?",
             "Para que serve um capacitor num circuito?",
             "Como se faz pão de queijo mineiro? Seja breve.",
             "E a derivada de x² · sen(x)?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--ram", type=float, default=10.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--saida", default=str(AQUI / "teste_cerebro.md"))
    ap.add_argument("--perguntas", type=int, default=len(PERGUNTAS))
    ap.add_argument("--modos", default="fiel,plastico")
    ap.add_argument("--ctx", type=int, default=4096)
    a = ap.parse_args()
    linhas = [f"# Teste do modo cérebro ({time.strftime('%d/%m %H:%M')})\n"]

    def log(*x):
        s = " ".join(map(str, x))
        print(s, flush=True)
        linhas.append(f"    {s}")

    t0 = time.time()
    m = Motor(a.modelo, ram_gb=a.ram, ctx=a.ctx, memoria=str(AQUI / "memoria" / (Path(a.modelo).stem + ".json")), log=log)
    linhas.append(f"\nCarregou em {time.time() - t0:.0f} s\n")
    c = Cerebro(m, sistema="Você é um assistente útil. Responda em português do Brasil. "
                           "Escreva fórmulas em LaTeX entre $...$ ou $$...$$.")
    for modo in a.modos.split(","):
        m.modo = modo
        c.nova_conversa()
        linhas.append(f"\n## Modo {modo}\n")
        for q in PERGUNTAS[:a.perguntas]:
            antes = dict(m.mem.stats)
            texto, info = c.responder(q, max_tokens=a.max_tokens, temperatura=0)
            s = m.mem.stats
            usos = sum(s[k] - antes[k] for k in ("vram", "ram", "ssd"))
            taxa = {k: (s[k] - antes[k]) / max(1, usos) for k in ("vram", "ram", "ssd")}
            resumo = (f"{info['tokens']} tokens · **{info['tok_s']:.1f} tok/s** · pergunta lida em "
                      f"{info['leitura_s']:.1f} s · especialistas vindos da VRAM {taxa['vram']:.0%}, "
                      f"RAM {taxa['ram']:.0%}, SSD {taxa['ssd']:.0%} · conexões criadas "
                      f"{s['criadas'] - antes['criadas']}, cortadas {s['cortadas'] - antes['cortadas']}"
                      + (f", recrutadas {s['recrutados'] - antes['recrutados']}" if modo == "plastico" else ""))
            print(f"[{modo}] {q}\n  {resumo}", flush=True)
            linhas += [f"**P:** {q}  \n*{resumo}*\n", texto.strip() + "\n"]
            Path(a.saida).write_text("\n".join(linhas), encoding="utf-8")
    vit = [(int(l), int(e)) for l, e in zip(*m.mem.vital.nonzero())]
    linhas.append(f"\nNeurônios vitais detectados sozinhos: {vit}\n")
    m.mem.consolidar()
    Path(a.saida).write_text("\n".join(linhas), encoding="utf-8")
    print("Salvo em", a.saida)


if __name__ == "__main__":
    main()
