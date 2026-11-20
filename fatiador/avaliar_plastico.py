"""
Avalia o modo plástico (llama.cpp do fork, branch cerebro) contra o modelo completo:
  A) qualidade: perplexidade num texto misto (6 assuntos em sequência, a fatia viva tem que acompanhar)
  B) velocidade e respostas de verdade numa conversa de 5 perguntas
Grava tudo em D:\\IA\\resultado_plastico.md (vai sendo atualizado a cada medição).

Uso: python avaliar_plastico.py [etapas]   ex.: A30 B30 C235 D235  (padrão: todas)
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

IA = Path(r"D:\IA")
import os
BIN = Path(os.environ.get("LLAMA_BIN", str(IA / "llama-cerebro")))
TEXTO = IA / "avaliacao_mista.txt"
SAIDA = IA / os.environ.get("SAIDA_MD", "resultado_plastico.md")
LOG = open(IA / "avaliar_plastico_log.txt", "a", encoding="utf-8", errors="replace")
MEM = IA / "memoria"
MEM.mkdir(exist_ok=True)
PORTA = 8094

M30 = next(iter(sorted((IA / "modelos").rglob("Qwen3-30B-A3B*.gguf"))), None)
M235 = next(iter(sorted((IA / "modelos").rglob("*235B*00001-of-*.gguf"))), None)
VITAIS = {"30b": "1:68,2:92,3:82", "235b": "2:39,3:120"}   # achados pelo detector do modo cérebro
CAMADAS = {"30b": 48, "235b": 94}

PERGUNTAS = ["Qual a derivada de e^(2x - 4y) em relação a x? Mostre a conta.",
             "Quanto vale a integral de x² de 0 a 3?",
             "Para que serve um capacitor num circuito?",
             "Como se faz pão de queijo mineiro? Seja breve.",
             "Explique a lei de Ohm com um exemplo numérico."]

linhas = [f"# Modo plástico no llama.cpp: resultados ({time.strftime('%d/%m %H:%M')})\n"]


def salvar():
    SAIDA.write_text("\n".join(linhas), encoding="utf-8")


def plastico(qual, n, vitais=True, skip=4, memoria=None):
    a = ["--moe-plastic", str(n), "--moe-plastic-skip", str(skip)]
    if vitais:
        a += ["--moe-plastic-vital", VITAIS[qual]]
    if memoria:
        a += ["--moe-plastic-memory", str(MEM / memoria)]
    return a


def rodar(cmd, timeout):
    LOG.write("\n\n===== " + " ".join(map(str, cmd)) + "\n")
    LOG.flush()
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, creationflags=subprocess.CREATE_NO_WINDOW)
    out = p.stdout + p.stderr
    LOG.write(out[-20000:])
    LOG.flush()
    return out, time.time() - t0


def stats_plastico(out):
    m = re.search(r"plastic MoE: (\d+) tokens, ([\d.]+)% of the wanted router weight on live experts, (\d+) recruited, "
                  r"(\d+) connections created, (\d+) cut", out)
    if not m:
        return ""
    return (f"cobertura {m.group(2)}% do que o roteador queria · {m.group(3)} recrutados · "
            f"{m.group(4)} conexões criadas · {m.group(5)} cortadas")


def perplexidade(qual, modelo, configs, chunks):
    L = CAMADAS[qual]
    linhas.append(f"\n## {qual}: perplexidade no texto misto ({chunks} blocos de 512 tokens; menor = melhor)\n")
    linhas.append("| Configuração | PPL | vs completo | Tempo | Fatia viva |\n|---|---|---|---|---|")
    base = None
    for rotulo, extra in configs:
        print(f"[{qual}] PPL: {rotulo}", flush=True)
        cmd = [BIN / "llama-perplexity.exe", "-m", modelo, "-f", TEXTO, "-c", "512", "--chunks", str(chunks),
               "-ngl", "99", "--n-cpu-moe", str(L), "-fa", "on", "-lv", "4"] + extra
        try:
            out, seg = rodar(cmd, 4 * 3600)
        except subprocess.TimeoutExpired:
            linhas.append(f"| {rotulo} | tempo esgotado | | | |")
            salvar()
            continue
        m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", out)
        if not m:
            linhas.append(f"| {rotulo} | falhou (veja avaliar_plastico_log.txt) | | {seg / 60:.1f} min | |")
            salvar()
            continue
        ppl = float(m.group(1))
        base = base or ppl
        linhas.append(f"| {rotulo} | {ppl:.3f} ± {float(m.group(2)):.3f} | {100 * (ppl / base - 1):+.1f}% | "
                      f"{seg / 60:.1f} min | {stats_plastico(out)} |")
        salvar()


def subir(modelo, args):
    cmd = [BIN / "llama-server.exe", "-m", modelo, "-ngl", "99", "--port", str(PORTA), "--host", "127.0.0.1",
           "--jinja", "-fa", "on", "--no-warmup", "-np", "1", "-lv", "4"] + args
    LOG.write("\n\n===== " + " ".join(map(str, cmd)) + "\n")
    LOG.flush()
    p = subprocess.Popen([str(c) for c in cmd], stdout=LOG, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            raise RuntimeError("o servidor fechou")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORTA}/health", timeout=2) as r:
                if r.status == 200:
                    return p, time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    p.kill()
    raise RuntimeError("demorou demais pra carregar")


def conversa(qual, modelo, configs, max_tokens, n_perguntas):
    L = CAMADAS[qual]
    linhas.append(f"\n## {qual}: conversa de {n_perguntas} perguntas ({max_tokens} tokens cada)\n")
    respostas = {}
    for rotulo, extra in configs:
        print(f"[{qual}] conversa: {rotulo}", flush=True)
        try:
            p, seg = subir(modelo, ["--n-cpu-moe", str(L), "-c", "8192"] + extra)
        except Exception as e:
            linhas.append(f"**{rotulo}**: não carregou ({e})\n")
            salvar()
            continue
        msgs, vel, txts = [], [], []
        try:
            for q in PERGUNTAS[:n_perguntas]:
                msgs.append({"role": "user", "content": q})
                corpo = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True}
                req = urllib.request.Request(f"http://127.0.0.1:{PORTA}/v1/chat/completions", data=json.dumps(corpo).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=4 * 3600) as r:
                    d = json.load(r)
                txt = d["choices"][0]["message"].get("content") or ""
                msgs.append({"role": "assistant", "content": txt})
                vel.append(d.get("timings", {}).get("predicted_per_second", 0))
                txts.append(txt)
                print(f"    {vel[-1]:6.2f} tok/s | {q[:50]}", flush=True)
        except Exception as e:
            linhas.append(f"**{rotulo}**: erro na conversa ({e})\n")
        finally:
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
        if vel:
            linhas.append(f"### {rotulo}\nCarregou em {seg:.0f} s · **média {sum(vel) / len(vel):.2f} tok/s** · por pergunta: "
                          + ", ".join(f"{v:.2f}" for v in vel) + "\n")
            respostas[rotulo] = txts
        salvar()
        time.sleep(5)
    linhas.append(f"\n<details><summary>Respostas completas ({qual})</summary>\n")
    for i, q in enumerate(PERGUNTAS[:n_perguntas]):
        linhas.append(f"\n**P{i + 1}: {q}**\n")
        for rotulo, txts in respostas.items():
            if i < len(txts):
                linhas.append(f"- _{rotulo}_:\n\n```\n{txts[i].strip()[:1500]}\n```\n")
    linhas.append("</details>\n")
    salvar()


def main():
    etapas = set(sys.argv[1:]) or {"A30", "B30", "C235", "D235"}
    cache = ["--moe-cache-mib", "2500", "--moe-cache-cpu-share", "1.0"]
    if "A30" in etapas and M30:
        perplexidade("30b", M30, [
            ("completo (128 especialistas)", []),
            ("plástico 64 + vitais + 4 camadas inteiras", plastico("30b", 64)),
            ("plástico 48 + vitais + 4 camadas inteiras", plastico("30b", 48)),
            ("plástico 32 + vitais + 4 camadas inteiras", plastico("30b", 32)),
            ("plástico 32 sem vitais, sem camadas inteiras", plastico("30b", 32, vitais=False, skip=0)),
            ("plástico 16 + vitais + 4 camadas inteiras", plastico("30b", 16)),
        ], chunks=40)
    if "G30" in etapas and M30:
        perplexidade("30b", M30, [
            ("completo (128 especialistas)", []),
            ("plástico 48, mesmo número em cada camada", plastico("30b", 48) + ["--moe-plastic-uniform"]),
            ("plástico 48, orçamento dividido entre as camadas", plastico("30b", 48)),
            ("plástico 32, mesmo número em cada camada", plastico("30b", 32) + ["--moe-plastic-uniform"]),
            ("plástico 32, orçamento dividido entre as camadas", plastico("30b", 32)),
            ("plástico 24, orçamento dividido entre as camadas", plastico("30b", 24)),
        ], chunks=40)
    if "B30" in etapas and M30:
        conversa("30b", M30, [
            ("completo + cache na placa (faltas no processador)", cache),
            ("plástico 32 + vitais + cache na placa", plastico("30b", 32) + cache),
        ], max_tokens=200, n_perguntas=5)
    if "C235" in etapas and M235:
        conversa("235b", M235, [
            ("completo", []),
            ("plástico 32 + vitais (1ª sessão, aprendendo do zero)", plastico("235b", 32, memoria="qwen3-235b.txt")),
            ("plástico 32 + vitais (2ª sessão, com a memória salva)", plastico("235b", 32, memoria="qwen3-235b.txt")),
        ], max_tokens=150, n_perguntas=5)
    if "D235" in etapas and M235:
        perplexidade("235b", M235, [
            ("completo", []),
            ("plástico 32 + vitais + 4 camadas inteiras", plastico("235b", 32)),
        ], chunks=8)
    print("\nPronto:", SAIDA, flush=True)


if __name__ == "__main__":
    main()
