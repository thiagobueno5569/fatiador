"""
Passo 1 — descobre quais especialistas (experts) cada assunto usa, camada por camada.

Roda o próprio GGUF no llama.cpp (llama-imatrix) com os textos de calibração de cada assunto.
O llama-imatrix anota quantas vezes cada especialista de cada camada foi escolhido pelo
roteador: são os "caminhos" de cada assunto. Roda no seu PC, não precisa de Kaggle.

Uso:
  python 1_medir_especialistas.py --llama C:\\llama --gguf "C:\\...\\Qwen3-30B-A3B-Q4_K_M.gguf"
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import gguf


# Super-especialista: ativação de pico milhares de vezes maior que a dos outros da camada.
# No Qwen3-30B os 3 conhecidos ficam entre 15 mil e 400 mil vezes; os demais abaixo de mil.
LIMIAR_SUPER = 5000


def exe(pasta, nome):
    for cand in (Path(pasta) / f"{nome}.exe", Path(pasta) / nome):
        if cand.exists():
            return str(cand)
    raise SystemExit(f"Não achei {nome} em {pasta}")


def contar(a, texto, saida):
    cmd = [exe(a.llama, "llama-imatrix"), "-m", a.gguf, "-f", str(texto), "-o", str(saida),
           "-c", "512", "--chunks", str(a.chunks), "-ngl", "99", "--n-cpu-moe", a.cpu_moe,
           "--no-ppl"]
    if a.parse_special:
        cmd.append("--parse-special")    # marcadores de conversa viram tokens especiais de verdade
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode == 0 and Path(saida).exists():
            break
    else:
        print((r.stdout + r.stderr)[-2000:])
        raise SystemExit("llama-imatrix falhou (veja a mensagem acima)")

    leitor = gguf.GGUFReader(str(saida))
    camadas, picos = {}, {}
    for t in leitor.tensors:
        m = re.match(r"blk\.(\d+)\.ffn_down_exps\.weight\.(counts|in_sum2)$", t.name)
        if m:
            arr = np.asarray(t.data, dtype=np.float64)
            if m.group(2) == "counts":
                camadas[int(m.group(1))] = arr.ravel()
            else:
                picos[int(m.group(1))] = arr
    if not camadas:
        raise SystemExit("O arquivo do imatrix não tem contagem por especialista (modelo não é MoE?)")
    supers = []
    for l, c in camadas.items():
        if l in picos and len(c) > 1:
            pico = picos[l].reshape(len(c), -1).max(1) / np.maximum(c, 1)
            ref = np.median(pico[c > 0]) if (c > 0).any() else 0
            if ref > 0:
                for e in np.where(pico / ref > LIMIAR_SUPER)[0]:
                    supers.append((l, int(e)))
    return [camadas[i].tolist() for i in sorted(camadas)], supers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", required=True, help="pasta com o llama-imatrix")
    ap.add_argument("--gguf", required=True, help="modelo original em GGUF")
    ap.add_argument("--textos", default="textos")
    ap.add_argument("--chunks", type=int, default=32, help="blocos de 512 tokens por assunto")
    ap.add_argument("--cpu-moe", default="99", help="camadas com especialistas na RAM")
    ap.add_argument("--saida", default="especialistas.json")
    ap.add_argument("--arquivo", default="calibracao.txt", help="ex.: calibracao_chat.txt")
    ap.add_argument("--parse-special", action="store_true")
    a = ap.parse_args()

    leitor = gguf.GGUFReader(a.gguf)
    arq = leitor.fields["general.architecture"].contents()
    n_exp = int(leitor.fields[f"{arq}.expert_count"].contents())
    top_k = int(leitor.fields[f"{arq}.expert_used_count"].contents())
    print(f"Modelo: {Path(a.gguf).name} | {n_exp} especialistas por camada, {top_k} ativos por token")
    del leitor

    Path("imatrix").mkdir(exist_ok=True)
    res = {"modelo": Path(a.gguf).name, "num_especialistas": n_exp, "top_k": top_k, "assuntos": {}}
    votos = {}
    for pasta in sorted(Path(a.textos).iterdir()):
        texto = pasta / a.arquivo
        if not texto.exists():
            continue
        t0 = time.time()
        camadas, supers = contar(a, texto, Path("imatrix") / f"{pasta.name}-{Path(a.arquivo).stem}.gguf")
        res["assuntos"][pasta.name] = camadas
        for x in supers:
            votos[x] = votos.get(x, 0) + 1
        usados = np.mean([(np.array(c) > 0).mean() for c in camadas])
        print(f"{pasta.name:12s} {time.time() - t0:5.0f}s | usou {usados:.0%} dos especialistas "
              f"pelo menos uma vez", flush=True)

    n_ass = len(res["assuntos"])
    achados = sorted(x for x, v in votos.items() if v * 2 > n_ass)    # aparece na maioria dos assuntos
    res["super"] = ",".join(f"{l}:{e}" for l, e in achados)
    print(f"Super-especialistas detectados (núcleo): {res['super'] or 'nenhum'}")
    Path(a.saida).write_text(json.dumps(res), encoding="utf-8")
    print(f"\nSalvo em {a.saida}.")
    resumo(res)


def resumo(r):
    """Quanto os assuntos se sobrepõem: se cada um usa caminhos próprios, a ideia funciona."""
    nomes = list(r["assuntos"])
    q = max(r["top_k"], r["num_especialistas"] // 4)
    print(f"\nSobreposição dos {q} especialistas favoritos por camada (100% = mesmos caminhos):")
    print(" " * 13 + "".join(f"{n[:6]:>8s}" for n in nomes))
    tops = {n: [set(np.argsort(c)[::-1][:q]) for c in r["assuntos"][n]] for n in nomes}
    for x in nomes:
        linha = [np.mean([len(p & s) / q for p, s in zip(tops[x], tops[y])]) for y in nomes]
        print(f"{x[:12]:13s}" + "".join(f"{v:8.0%}" for v in linha))


if __name__ == "__main__":
    main()
