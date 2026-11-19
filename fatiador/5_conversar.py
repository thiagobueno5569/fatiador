"""
Passo 5 — CONVERSAR com a IA fatiada e ver o que ela está fazendo.

Abre um app no navegador (http://localhost:8765):
  * Você conversa normalmente.
  * Um roteador de assunto compara a sua mensagem com os textos de cada assunto
    e escolhe a fatia certa (eletrônica, cálculo, ...). Se o assunto mudar, troca a fatia.
  * O painel mostra a pontuação de cada assunto, o mapa dos especialistas ligados
    (48 camadas x 128 especialistas, com os super-especialistas do núcleo),
    a velocidade da resposta e o histórico da conversa.
  * Dá pra forçar "modelo inteiro" pra comparar.

Uso:  python 5_conversar.py          (ou dois cliques no CONVERSAR.bat)
"""
import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent


def importar(nome_arquivo):
    spec = importlib.util.spec_from_file_location(nome_arquivo.replace(".py", ""), AQUI / nome_arquivo)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fatiador = importar("2_fatiar_gguf.py")
comparador = importar("3_comparar.py")

PARADAS = set("""a o e é de da do das dos em no na nos nas um uma uns umas para por com sem que se
como mais mas ou ao aos à às pelo pela pelos pelas seu sua seus suas ele ela eles elas isso esse
essa este esta isto também já não sim são foi ser ter tem há entre sobre até quando onde muito
qual quais você vocês eu me meu minha nós lhe the and of to in is""".split())


# ---------------------------------------------------------------- roteador de assunto
class Roteador:
    """TF-IDF simples: cada assunto vira um vetor de palavras; a mensagem também.
    A pontuação é o cosseno entre eles (o mesmo 'quão perto apontam' da esfera)."""

    def __init__(self, pasta_textos):
        self.assuntos, docs = [], []
        for p in sorted(Path(pasta_textos).iterdir()):
            arq = p / "calibracao.txt"
            if arq.exists():
                self.assuntos.append(p.name)
                docs.append(Counter(self.palavras(arq.read_text(encoding="utf-8"))))
        n = len(docs)
        df = Counter(w for d in docs for w in d)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self.vet = [self.normalizar({w: (1 + math.log(c)) * self.idf[w] for w, c in d.items()}) for d in docs]

    @staticmethod
    def palavras(t):
        return [w for w in re.findall(r"[a-zà-ú0-9]+", t.lower()) if len(w) > 2 and w not in PARADAS]

    @staticmethod
    def normalizar(v):
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {k: x / n for k, x in v.items()}

    def pontuar(self, textos):
        c = Counter()
        for peso, t in zip((1.0, 0.5, 0.25), reversed(textos[-3:])):   # mensagem atual pesa mais
            for w in self.palavras(t):
                if w in self.idf:
                    c[w] += peso
        q = self.normalizar({w: (1 + math.log(1 + x)) * self.idf[w] for w, x in c.items()})
        return {a: round(sum(q[w] * v.get(w, 0.0) for w in q), 4) for a, v in zip(self.assuntos, self.vet)}


# ---------------------------------------------------------------- gerente das fatias
class Gerente:
    def __init__(self, a):
        self.a = a
        self.proc = None
        self.carregado = None          # "eletronica" | "inteiro"
        self.trava = threading.Lock()
        self.info = json.loads((a.dados / a.especialistas).read_text(encoding="utf-8"))
        if a.super == "auto":
            a.super = self.info.get("super", "1:68,2:92,3:82" if self.info.get("num_especialistas") == 128
                                    and len(next(iter(self.info["assuntos"].values()))) == 48 else "")
        self.fixos = fatiador.ler_sempre(a.super)

    def caminho(self, alvo):
        if alvo == "inteiro":
            return Path(self.a.original)
        return self.a.dados / self.a.fatias / f"{alvo}-{self.a.manter}esp.gguf"

    def garantir_fatia(self, alvo, avisar):
        p = self.caminho(alvo)
        if alvo != "inteiro" and not p.exists():
            avisar(f"Criando a fatia de {alvo} ({self.a.manter} especialistas), leva uns 3 minutos...")
            p.parent.mkdir(exist_ok=True)
            esc = fatiador.escolher(self.info["assuntos"][alvo], self.a.manter, self.fixos)
            fatiador.fatiar(self.a.original, esc, str(p))
        return p

    def carregar(self, alvo, avisar):
        """Garante que o llama-server está rodando com a fatia 'alvo'. Devolve (trocou, segundos)."""
        if self.carregado == alvo and self.proc and self.proc.poll() is None:
            return False, 0.0
        t0 = time.time()
        p = self.garantir_fatia(alvo, avisar)
        self.parar()
        n_cpu = "99" if alvo == "inteiro" else comparador.cpu_moe_auto(p, self.a.vram)
        avisar(f"Carregando {'o modelo inteiro' if alvo == 'inteiro' else 'a fatia de ' + alvo} na placa...")
        cmd = [self.a.servidor, "-m", str(p), "-ngl", "99", "-c", str(self.a.contexto),
               "--host", "127.0.0.1", "--port", str(self.a.porta_llama), "--jinja"]
        if n_cpu != "0":
            cmd += ["--n-cpu-moe", n_cpu]
        if alvo != "inteiro" and p.stat().st_size < 20e9:
            # a fatia cabe na RAM: carrega tudo de uma vez, senão a 1ª resposta fica lendo do SSD
            cmd += ["--load-mode", "none"]
        cmd += self.a.extra
        self.log = open(AQUI / "log_servidor.txt", "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, creationflags=flags)
        url = f"http://127.0.0.1:{self.a.porta_llama}/health"
        while time.time() - t0 < 600:
            if self.proc.poll() is not None:
                raise RuntimeError("O llama-server fechou ao carregar (veja log_servidor.txt).")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("O llama-server demorou demais pra carregar.")
        self.carregado, self.n_cpu = alvo, n_cpu
        return True, time.time() - t0

    def parar(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def mapa(self, alvo):
        """Matriz camadas x especialistas: uso relativo e quem está ligado na fatia."""
        n_esp = self.info["num_especialistas"]
        if alvo == "inteiro":
            cont = [[sum(c) for c in zip(*[self.info["assuntos"][x][l] for x in self.info["assuntos"]])]
                    for l in range(len(next(iter(self.info["assuntos"].values()))))]
            ligados = [list(range(n_esp))] * len(cont)
        else:
            cont = self.info["assuntos"][alvo]
            ligados = [list(map(int, x)) for x in fatiador.escolher(cont, self.a.manter, self.fixos)]
        uso, cobertura = [], []
        for l, c in enumerate(cont):
            m, tot = max(c) or 1, sum(c) or 1
            uso.append([round(x / m, 3) for x in c])
            cobertura.append(sum(c[i] for i in ligados[l]) / tot)
        return {"alvo": alvo, "n_esp": n_esp, "camadas": len(cont), "uso": uso, "ligados": ligados,
                "super": [[l, e] for l, s in self.fixos.items() for e in s],
                "cobertura": round(sum(cobertura) / len(cobertura), 4)}


# ---------------------------------------------------------------- servidor web
def criar_handler(a, rot, ger):
    estado = {"atual": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                b = (AQUI / "conversar.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path.startswith("/api/info"):
                self._json({"assuntos": [x for x in rot.assuntos if x in ger.info["assuntos"]], "manter": a.manter,
                            "modelo": Path(a.original).name,
                            "tam_original": round(Path(a.original).stat().st_size / 1e9, 1),
                            "carregado": ger.carregado})
            elif self.path.startswith("/api/mapa"):
                alvo = self.path.split("alvo=")[-1] if "alvo=" in self.path else (ger.carregado or rot.assuntos[0])
                self._json(ger.mapa(alvo))
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path != "/api/chat":
                return self.send_error(404)
            corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs, modo = corpo["mensagens"], corpo.get("modo", "auto")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def evento(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            try:
                with ger.trava:
                    textos = [m["content"] for m in msgs if m["role"] == "user"]
                    pont = {k: v for k, v in rot.pontuar(textos).items() if k in ger.info["assuntos"]}
                    melhor = max(pont, key=pont.get)
                    atual = estado["atual"]
                    if modo == "inteiro":
                        alvo = "inteiro"
                    elif modo != "auto":
                        alvo = modo
                    elif atual and atual in pont and pont[melhor] < pont[atual] * 1.15 + 0.01:
                        alvo = atual          # histerese: só troca se o novo assunto ganhar com folga
                    else:
                        alvo = melhor
                    if alvo != "inteiro":
                        estado["atual"] = alvo
                    evento({"tipo": "rota", "pontuacao": pont, "alvo": alvo, "melhor": melhor})
                    trocou, seg = ger.carregar(alvo, lambda t: evento({"tipo": "aviso", "texto": t}))
                    tam = Path(ger.caminho(alvo)).stat().st_size / 1e9
                    evento({"tipo": "carregado", "alvo": alvo, "trocou": trocou, "segundos": round(seg, 1),
                            "tamanho_gb": round(tam, 1), "camadas_ram": int(ger.n_cpu)})
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{a.porta_llama}/v1/chat/completions",
                        data=json.dumps({"messages": msgs, "stream": True, "temperature": 0.7,
                                         "max_tokens": a.max_tokens,
                                         "chat_template_kwargs": {"reasoning_effort": "low"}}).encode(),
                        headers={"Content-Type": "application/json"})
                    t0, n, timings = time.time(), 0, None
                    t_primeiro = None
                    with urllib.request.urlopen(req, timeout=600) as r:
                        for linha in r:
                            linha = linha.decode("utf-8", "replace").strip()
                            if not linha.startswith("data:") or linha == "data: [DONE]":
                                continue
                            d = json.loads(linha[5:])
                            timings = d.get("timings", timings)
                            for ch in d.get("choices", []):
                                pens = (ch.get("delta") or {}).get("reasoning_content")
                                if pens:
                                    t_primeiro = t_primeiro or time.time()
                                    evento({"tipo": "pensando", "texto": pens})
                                txt = (ch.get("delta") or {}).get("content")
                                if txt:
                                    n += 1
                                    t_primeiro = t_primeiro or time.time()
                                    evento({"tipo": "texto", "texto": txt})
                    dur = time.time() - (t_primeiro or t0)
                    fim = {"tipo": "fim", "tokens": n,
                           "tok_s": round(n / dur, 1) if dur > 0 and n > 1 else None,
                           "espera_s": round((t_primeiro or time.time()) - t0, 2)}
                    if timings:
                        fim["tok_s"] = round(timings.get("predicted_per_second", fim["tok_s"] or 0), 1)
                        fim["prompt_tok_s"] = round(timings.get("prompt_per_second", 0), 1)
                        fim["tokens"] = timings.get("predicted_n", n)
                    evento(fim)
            except Exception as e:
                try:
                    evento({"tipo": "erro", "texto": str(e)})
                except Exception:
                    pass

    return H


def achar_original():
    from os.path import expanduser
    for raiz in (Path(expanduser("~")) / ".lmstudio" / "models", Path(expanduser("~")) / ".cache" / "lm-studio" / "models"):
        if raiz.exists():
            achados = [p for p in raiz.rglob("*.gguf") if "Qwen3-30B-A3B" in p.name and "mmproj" not in p.name]
            if achados:
                return str(min(achados, key=lambda p: p.stat().st_size))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", default=None, help="GGUF original (padrão: procura no LM Studio)")
    ap.add_argument("--servidor", default=None, help="caminho do llama-server(.exe)")
    ap.add_argument("--manter", type=int, default=64)
    ap.add_argument("--super", default="auto")
    ap.add_argument("--dados", default=None, help="pasta com especialistas.json e fatias (padrão: esta)")
    ap.add_argument("--fatias", default="fatias", help="subpasta das fatias dentro de --dados")
    ap.add_argument("--especialistas", default="especialistas.json")
    ap.add_argument("--vram", type=float, default=4.6)
    ap.add_argument("--contexto", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--porta", type=int, default=8765)
    ap.add_argument("--porta-llama", type=int, default=8081)
    ap.add_argument("--nao-abrir", action="store_true")
    ap.add_argument("extra", nargs="*", help="argumentos extras pro llama-server (depois de --)")
    a = ap.parse_args()

    a.original = a.original or achar_original()
    if not a.original:
        raise SystemExit("Não achei o Qwen3-30B-A3B. Use --original caminho\\do\\modelo.gguf")
    if not a.servidor:
        cands = list((AQUI / "llama").rglob("llama-server.exe")) + list((AQUI / "llama").rglob("llama-server"))
        if not cands:
            raise SystemExit("Não achei o llama-server na pasta llama (rode o RODAR.bat antes).")
        a.servidor = str(cands[0])

    a.dados = Path(a.dados).resolve() if a.dados else AQUI
    rot = Roteador(AQUI / "textos")
    ger = Gerente(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), criar_handler(a, rot, ger))
    url = f"http://localhost:{a.porta}"
    print(f"Modelo: {a.original}\nFatias de {a.manter} especialistas | assuntos: {', '.join(rot.assuntos)}")
    print(f"\nAbra {url} no navegador. Para sair, feche esta janela.")
    if not a.nao_abrir:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    finally:
        ger.parar()


if __name__ == "__main__":
    main()
