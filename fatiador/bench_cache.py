"""
Compara o llama.cpp normal com o cache de especialistas na placa (--moe-cache-mib), numa conversa
de verdade com várias perguntas (o cache aprende ao longo da conversa). Grava D:\\IA\\bench_cache.md.

Uso: python bench_cache.py [30b|235b|tudo]
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

IA = Path(r"D:\IA")
SRV = next((IA / "llama").rglob("llama-server.exe"))
PORTA = 8093
PERGUNTAS = ["Qual a derivada de e^(2x - 4y) em relação a x? Mostre a conta.",
             "Quanto vale a integral de x² de 0 a 3?",
             "E a derivada de x² · sen(x)?",
             "Para que serve um capacitor num circuito?",
             "Explique a lei de Ohm com um exemplo numérico."]


def achar(padrao, pasta):
    return next(iter(sorted(Path(pasta).rglob(padrao))), None)


MODELOS = {
    "30b": (achar("Qwen3-30B-A3B*.gguf", IA / "modelos"), 48, 200, 8192),
    "235b": (achar("*235B*00001-of-*.gguf", IA / "modelos"), 94, 100, 4096),
}


def subir(modelo, args, log):
    cmd = [str(SRV), "-m", str(modelo), "-ngl", "99", "--port", str(PORTA), "--host", "127.0.0.1",
           "--jinja", "-fa", "on", "--no-warmup"] + args
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    t0 = time.time()
    while time.time() - t0 < 1800:
        if p.poll() is not None:
            raise RuntimeError("o servidor fechou (veja bench_cache_log.txt)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORTA}/health", timeout=2) as r:
                if r.status == 200:
                    return p, time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    p.kill()
    raise RuntimeError("demorou demais pra carregar")


def conversa(max_tokens):
    msgs, res = [], []
    for q in PERGUNTAS:
        msgs.append({"role": "user", "content": q})
        corpo = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": True}
        req = urllib.request.Request(f"http://127.0.0.1:{PORTA}/v1/chat/completions",
                                     data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            d = json.load(r)
        txt = d["choices"][0]["message"].get("content") or ""
        msgs.append({"role": "assistant", "content": txt})
        t = d.get("timings", {})
        res.append((q, t.get("predicted_per_second", 0), t.get("prompt_per_second", 0), t.get("predicted_n", 0), txt))
        print(f"    {t.get('predicted_per_second', 0):6.2f} tok/s | {q[:50]}", flush=True)
    return res


def main():
    qual = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    saida = IA / ("bench_cache_extra.md" if "--so-cpu-share" in sys.argv else "bench_cache.md")
    linhas = [f"# llama.cpp normal x cache de especialistas na placa ({time.strftime('%d/%m %H:%M')})\n",
              f"Servidor: `{SRV}`\n"]
    log = open(IA / "bench_cache_log.txt", "a", encoding="utf-8", errors="replace")
    for nome, (modelo, L, max_tok, ctx) in MODELOS.items():
        if qual not in (nome, "tudo") or modelo is None:
            continue
        cache = "2500" if nome == "30b" else "1000"
        base = ["--n-cpu-moe", str(L), "-c", str(ctx)]
        configs = [("especialistas todos na RAM (normal)", base),
                   (f"+ cache de {cache} MiB na placa, faltas enviadas pra placa (padrão)",
                    base + ["--moe-cache-mib", cache]),
                   (f"+ cache de {cache} MiB na placa, faltas calculadas no processador (como o modo cérebro)",
                    base + ["--moe-cache-mib", cache, "--moe-cache-cpu-share", "1.0"])]
        if "--so-cpu-share" in sys.argv:
            configs = configs[2:]
        linhas.append(f"\n## {modelo.name}\n")
        for rotulo, args in configs:
            print(f"\n=== {nome}: {rotulo} ===", flush=True)
            log.write(f"\n\n===== {nome}: {rotulo} {' '.join(args)}\n")
            log.flush()
            try:
                p, seg = subir(modelo, args, log)
            except Exception as e:
                linhas.append(f"**{rotulo}**: não carregou ({e})\n")
                continue
            try:
                res = conversa(max_tok)
            except Exception as e:
                linhas.append(f"**{rotulo}**: erro na conversa ({e})\n")
                res = []
            finally:
                p.terminate()
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
            if res:
                media = sum(r[1] for r in res) / len(res)
                linhas.append(f"### {rotulo}\nCarregou em {seg:.0f} s · **média {media:.2f} tok/s** · "
                              f"por pergunta: " + ", ".join(f"{r[1]:.2f}" for r in res) + "\n")
                linhas.append(f"> Primeira resposta: {res[0][4][:300].replace(chr(10), ' ')}\n")
            saida.write_text("\n".join(linhas), encoding="utf-8")
            time.sleep(5)
    saida.write_text("\n".join(linhas), encoding="utf-8")
    print("\nSalvo em", saida)


if __name__ == "__main__":
    main()
