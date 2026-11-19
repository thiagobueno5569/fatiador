"""
Baixa um arquivo grande com várias conexões em paralelo, continuando de onde parou.
Cada pedaço é pedido separadamente (HTTP Range); se uma conexão travar, só aquele
pedaço é pedido de novo. No final confere o SHA-256 com o que o Hugging Face informa.

Uso: python baixar_modelo.py URL DESTINO TAMANHO [--sha256 HASH] [--conexoes 8]
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

PEDACO = int(os.environ.get("PEDACO_MB","128")) * 1024 * 1024
UA = {"User-Agent": "fatiador-baixador/1.0"}


def baixar_pedaco(url, dest, ini, fim, progresso, trava):
    for tentativa in range(30):
        feito = 0
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={ini}-{fim}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status != 206:
                    raise RuntimeError(f"servidor não aceitou pedaço (HTTP {r.status})")
                with open(dest, "r+b") as f:
                    f.seek(ini)
                    while True:
                        bloco = r.read(1024 * 1024)
                        if not bloco:
                            break
                        f.write(bloco)
                        feito += len(bloco)
                        with trava:
                            progresso[0] += len(bloco)
            if feito == fim - ini + 1:
                return ini
            raise RuntimeError("pedaço incompleto")
        except Exception as e:
            with trava:
                progresso[0] -= feito
            time.sleep(min(30, 2 + tentativa * 2))
    raise RuntimeError(f"desisti do pedaço {ini}-{fim}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("dest"); ap.add_argument("tamanho", type=int)
    ap.add_argument("--sha256", default=None)
    ap.add_argument("--conexoes", type=int, default=8)
    a = ap.parse_args()

    estado_arq = a.dest + ".progresso.json"
    if os.path.exists(estado_arq):
        estado = json.load(open(estado_arq))
    else:
        # o que já existe (de um download sequencial anterior) é considerado pronto
        base = os.path.getsize(a.dest) if os.path.exists(a.dest) else 0
        base = (base // PEDACO) * PEDACO       # descarta o pedaço final, que pode estar pela metade
        estado = {"base": base, "feitos": []}
    if not os.path.exists(a.dest):
        open(a.dest, "wb").close()
    with open(a.dest, "r+b") as f:             # reserva o tamanho final
        f.truncate(a.tamanho)

    feitos = set(estado["feitos"])
    pedacos = [(i, min(i + PEDACO, a.tamanho) - 1) for i in range(estado["base"], a.tamanho, PEDACO)
               if i not in feitos]
    total_falta = sum(f - i + 1 for i, f in pedacos)
    print(f"Já tenho {(a.tamanho - total_falta) / 1e9:.1f} GB; faltam {total_falta / 1e9:.1f} GB "
          f"em {len(pedacos)} pedaços, {a.conexoes} conexões ao mesmo tempo.", flush=True)

    progresso, trava = [0], threading.Lock()
    t0 = time.time()

    def mostrar():
        while not parar.is_set():
            dt = max(1e-6, time.time() - t0)
            v = progresso[0] / dt
            resto = (total_falta - progresso[0]) / v if v > 0 else 0
            sys.stdout.write(f"\r  {progresso[0] / 1e9:6.1f} / {total_falta / 1e9:.1f} GB  "
                             f"{v / 1e6:5.1f} MB/s  falta ~{resto / 60:4.0f} min   ")
            sys.stdout.flush()
            parar.wait(5)

    parar = threading.Event()
    threading.Thread(target=mostrar, daemon=True).start()
    with ThreadPoolExecutor(a.conexoes) as ex:
        futs = [ex.submit(baixar_pedaco, a.url, a.dest, i, f, progresso, trava) for i, f in pedacos]
        for fu in as_completed(futs):
            feitos.add(fu.result())
            estado["feitos"] = sorted(feitos)
            json.dump(estado, open(estado_arq, "w"))
    parar.set()
    print(f"\nDownload completo em {(time.time() - t0) / 60:.0f} min.", flush=True)

    if a.sha256:
        print("Conferindo o arquivo (SHA-256)...", flush=True)
        h = hashlib.sha256()
        with open(a.dest, "rb") as f:
            for bloco in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(bloco)
        if h.hexdigest() != a.sha256.lower():
            os.remove(estado_arq)
            raise SystemExit("O arquivo veio corrompido (SHA-256 diferente). Apague-o e rode de novo.")
        print("Arquivo conferido: íntegro.")
    if os.path.exists(estado_arq):
        os.remove(estado_arq)


if __name__ == "__main__":
    main()
