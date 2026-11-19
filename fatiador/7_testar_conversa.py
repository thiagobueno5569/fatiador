"""
Passo 7 — TESTE DE CONVERSA: faz perguntas de verdade a cada fatia e salva as respostas.

A perplexidade mede texto corrido; aqui a gente vê se a fatia consegue CONVERSAR
(seguir o formato de chat, raciocinar e responder com coerência).

Uso:
  python 7_testar_conversa.py --llama llama --saida teste.md \
      fatia_chat=gptoss/fatias_chat32/calculo-32esp.gguf inteiro=E:/modelos/gpt-oss.gguf
Cada modelo é "nome=caminho". O assunto das perguntas sai do nome do arquivo
(calculo-32esp.gguf -> calculo); pro modelo inteiro, use --assuntos.
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

AQUI = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("comparador", AQUI / "3_comparar.py")
comparador = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparador)

PERGUNTAS = {
    "calculo": ["O que é a derivada de uma função? Dê um exemplo simples.",
                "Quanto vale a integral de x² de 0 a 3? Mostre a conta."],
    "eletronica": ["Para que serve um capacitor num circuito?",
                   "Explique a lei de Ohm com um exemplo numérico."],
    "culinaria": ["Como se faz pão de queijo mineiro?",
                  "Qual a diferença entre feijoada e tutu de feijão?"],
    "futebol": ["O que é impedimento no futebol?", "Quem foi Garrincha?"],
    "historia": ["O que foi a Inconfidência Mineira?", "Por que a Era Vargas foi importante?"],
    "programacao": ["O que é recursividade? Mostre um exemplo em Python.",
                    "Qual a diferença entre lista e tupla em Python?"],
}
PORTA = 8091


def subir(a, modelo):
    srv = next((AQUI / a.llama).rglob("llama-server.exe"), None) or next((AQUI / a.llama).rglob("llama-server"))
    tam = Path(modelo).stat().st_size / 1e9
    n_cpu = comparador.cpu_moe_auto(modelo, a.vram)
    cmd = [str(srv), "-m", str(modelo), "-ngl", "99", "-c", "4096", "--port", str(PORTA),
           "--host", "127.0.0.1", "--jinja", "--n-cpu-moe", n_cpu] + a.extra
    if tam < 20:
        cmd += ["--load-mode", "none"]
    log = open(AQUI / "log_teste_servidor.txt", "a", encoding="utf-8", errors="replace")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError("llama-server fechou (veja log_teste_servidor.txt)")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORTA}/health", timeout=2) as r:
                if r.status == 200:
                    return p, time.time() - t0
        except Exception:
            pass
        time.sleep(1)
    p.kill()
    raise RuntimeError("demorou demais pra carregar")


def perguntar(q, max_tokens):
    corpo = {"messages": [{"role": "user", "content": q}], "max_tokens": max_tokens,
             "temperature": 0.3, "chat_template_kwargs": {"reasoning_effort": "low"}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORTA}/v1/chat/completions",
                                 data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    tim = d.get("timings", {})
    return {"resposta": (msg.get("content") or "").strip(),
            "raciocinio": (msg.get("reasoning_content") or "").strip(),
            "tok_s": round(tim.get("predicted_per_second", 0), 1),
            "tokens": tim.get("predicted_n"), "segundos": round(time.time() - t0, 1)}


def lixo(txt):
    """Fração de pontinhos/reticências: sinal de que o modelo 'desmontou'."""
    if not txt:
        return 1.0
    return sum(c in ".…" for c in txt) / len(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default="llama")
    ap.add_argument("--saida", default="teste_conversa.md")
    ap.add_argument("--assuntos", default="calculo", help="pro modelo inteiro: assuntos separados por vírgula")
    ap.add_argument("--vram", type=float, default=4.6)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("modelos", nargs="+", help='nome=caminho')
    ap.add_argument("--extra", action="append", default=[], help="argumento extra pro llama-server (repita)")
    a = ap.parse_args()

    out = [f"# Teste de conversa ({time.strftime('%d/%m %H:%M')})\n"]
    for item in a.modelos:
        nome, caminho = item.split("=", 1)
        assunto = Path(caminho).name.split("-")[0]
        assuntos = [assunto] if assunto in PERGUNTAS else a.assuntos.split(",")
        print(f"\n=== {nome}: {Path(caminho).name} ===", flush=True)
        try:
            p, seg = subir(a, caminho)
        except Exception as e:
            out.append(f"## {nome}\nNão carregou: {e}\n"); print("  erro:", e); continue
        out.append(f"## {nome} — `{Path(caminho).name}` ({Path(caminho).stat().st_size / 1e9:.1f} GB, carregou em {seg:.0f} s)\n")
        try:
            for ass in assuntos:
                for q in PERGUNTAS.get(ass, []):
                    print(f"  pergunta: {q}", flush=True)
                    r = perguntar(q, a.max_tokens)
                    estado = "⚠️ DESMONTOU" if lixo(r["resposta"] or r["raciocinio"]) > 0.3 or not r["resposta"] else "ok"
                    print(f"    {estado} | {r['tok_s']} tok/s | {len(r['resposta'])} caracteres", flush=True)
                    out += [f"**P:** {q}  \n*{estado} · {r['tok_s']} tok/s · {r['tokens']} tokens · {r['segundos']} s*\n",
                            f"> raciocínio: {r['raciocinio'][:400].replace(chr(10), ' ')}\n",
                            (r["resposta"][:1500] or "_(sem resposta final)_") + "\n"]
        finally:
            p.terminate()
            try:
                p.wait(15)
            except subprocess.TimeoutExpired:
                p.kill()
        Path(a.saida).write_text("\n".join(out), encoding="utf-8")
    Path(a.saida).write_text("\n".join(out), encoding="utf-8")
    print(f"\nSalvo em {a.saida}")


if __name__ == "__main__":
    main()
