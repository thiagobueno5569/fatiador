"""
Passo 3 — compara o modelo NORMAL com as FATIAS por assunto, usando o llama.cpp.

Para cada assunto mede:
  * Perplexidade (PPL) no texto de avaliação: quanto MENOR, melhor o modelo entende o assunto.
      - normal:          modelo inteiro (especialistas na RAM, núcleo na placa)
      - fatia certa:     fatia do próprio assunto, inteira na placa
      - fatia errada:    fatia de OUTRO assunto (controle: prova que o assunto importa)
  * Velocidade (tokens/s) e tamanho de cada um.

Uso (Windows, com o llama.cpp baixado em C:\\llama):
  python 3_comparar.py --llama C:\\llama --original Qwen3-30B-A3B-Q4_K_M.gguf
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def exe(pasta, nome):
    for cand in (Path(pasta) / f"{nome}.exe", Path(pasta) / nome):
        if cand.exists():
            return str(cand)
    raise SystemExit(f"Não achei {nome} em {pasta}")


def rodar(cmd, tentativas=3):
    for _ in range(tentativas):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0:
            return r
    print((r.stdout + r.stderr)[-1500:])
    raise SystemExit(f"{Path(cmd[0]).name} falhou {tentativas} vezes (veja a mensagem acima)")


def perplexidade(a, modelo, texto, extra, ctx=512):
    cmd = [exe(a.llama, "llama-perplexity"), "-m", str(modelo), "-f", str(texto),
           "-c", str(ctx), "--chunks", str(a.chunks), "-ngl", "99", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if "you need at least" in (r.stdout + r.stderr) and ctx > 128:
        # texto de avaliação curto demais pra esse contexto: usa um contexto menor
        return perplexidade(a, modelo, texto, extra, ctx // 2)
    if r.returncode != 0:
        r = rodar(cmd)
    m = re.search(r"Final estimate: PPL = ([\d.]+)", r.stdout + r.stderr)
    if not m:
        print((r.stdout + r.stderr)[-1500:])
        raise SystemExit("llama-perplexity não deu resultado (texto de avaliação curto demais?)")
    return float(m.group(1))


def velocidade(a, modelo, extra):
    cmd = [exe(a.llama, "llama-bench"), "-m", str(modelo), "-p", "512", "-n", "128", "-r", "2",
           "-ngl", "99", "-o", "json", *extra]
    r = rodar(cmd)
    try:
        testes = json.loads(r.stdout[r.stdout.index("["):])
    except ValueError:
        print((r.stdout + r.stderr)[-1500:])
        raise SystemExit("llama-bench falhou (veja a mensagem acima)")
    ler = {t["n_prompt"] > 0: t["avg_ts"] for t in testes}
    return ler[True], ler[False]          # (ler o prompt, gerar resposta)


def cpu_moe_auto(modelo, vram_gb):
    """Quantas camadas de especialistas mandar pra RAM pra fatia caber na placa."""
    import gguf
    leitor = gguf.GGUFReader(str(modelo))
    por_camada, nucleo = {}, 0
    for t in leitor.tensors:
        m = re.match(r"blk\.(\d+)\..*_exps", t.name)
        if m:
            por_camada[int(m.group(1))] = por_camada.get(int(m.group(1)), 0) + t.n_bytes
        else:
            nucleo += t.n_bytes
    total = nucleo + sum(por_camada.values())
    sobra = total - vram_gb * 1e9
    if sobra <= 0 or not por_camada:
        return "0"
    media = sum(por_camada.values()) / len(por_camada)
    return str(min(len(por_camada), int(sobra // media) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="pasta com llama-perplexity e llama-bench")
    ap.add_argument("--original", required=True)
    ap.add_argument("--fatias", default="fatias")
    ap.add_argument("--textos", default="textos")
    ap.add_argument("--cpu-moe-original", default="99",
                    help="camadas com especialistas na CPU no modelo normal (99 = todas)")
    ap.add_argument("--cpu-moe-fatia", default="auto",
                    help="camadas da fatia com especialistas na RAM; auto = calcula pra caber na placa")
    ap.add_argument("--vram", type=float, default=4.6, help="GB da placa disponíveis pros pesos")
    ap.add_argument("--chunks", type=int, default=16)
    a = ap.parse_args()

    ext_orig = ["--n-cpu-moe", a.cpu_moe_original]
    fatias = {p.name.split("-")[0]: p for p in sorted(Path(a.fatias).glob("*.gguf"))}
    if not fatias:
        raise SystemExit("Nenhuma fatia encontrada. Rode o 2_fatiar_gguf.py antes.")
    if a.cpu_moe_fatia == "auto":
        a.cpu_moe_fatia = cpu_moe_auto(next(iter(fatias.values())), a.vram)
        print(f"Fatia: {a.cpu_moe_fatia} camada(s) de especialistas na RAM pra caber na placa")
    ext_fat = ["--n-cpu-moe", a.cpu_moe_fatia] if a.cpu_moe_fatia != "0" else []
    nomes = list(fatias)

    linhas = []
    for i, assunto in enumerate(nomes):
        texto = Path(a.textos) / assunto / "avaliacao.txt"
        outro = nomes[(i + 1) % len(nomes)]
        print(f"[{assunto}] medindo perplexidade...", flush=True)
        p_norm = perplexidade(a, a.original, texto, ext_orig)
        p_cert = perplexidade(a, fatias[assunto], texto, ext_fat)
        p_err = perplexidade(a, fatias[outro], texto, ext_fat) if len(nomes) > 1 else float("nan")
        linhas.append((assunto, p_norm, p_cert, outro, p_err))
        print(f"   normal {p_norm:.2f} | fatia certa {p_cert:.2f} | fatia de {outro} {p_err:.2f}")

    print("\nmedindo velocidade...", flush=True)
    v_norm = velocidade(a, a.original, ext_orig)
    v_fat = velocidade(a, fatias[nomes[0]], ext_fat)
    g = lambda p: os.path.getsize(p) / 1e9

    out = ["# Resultado: modelo normal × fatias por assunto\n",
           "## Qualidade (perplexidade — menor é melhor)\n",
           "| Assunto | Normal | Fatia certa | Diferença | Fatia errada (controle) |",
           "|---|---|---|---|---|"]
    for assunto, pn, pc, outro, pe in linhas:
        out.append(f"| {assunto} | {pn:.2f} | {pc:.2f} | {(pc / pn - 1):+.1%} | {pe:.2f} ({outro}) |")
    out += ["\n## Velocidade e tamanho\n",
            "| Modelo | Tamanho | Ler prompt (tok/s) | Gerar resposta (tok/s) |",
            "|---|---|---|---|",
            f"| Normal | {g(a.original):.1f} GB | {v_norm[0]:.0f} | {v_norm[1]:.1f} |",
            f"| Fatia ({nomes[0]}) | {g(fatias[nomes[0]]):.1f} GB | {v_fat[0]:.0f} | {v_fat[1]:.1f} |",
            f"\nGanho na geração: **{v_fat[1] / v_norm[1]:.1f}x**"]
    Path("resultados.md").write_text("\n".join(out), encoding="utf-8")
    print("\n" + "\n".join(out) + "\n\nSalvo em resultados.md")


if __name__ == "__main__":
    main()
