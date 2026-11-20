"""
Modo cérebro — app de conversa com o mapa das conexões ao vivo.

Uso:
  python servidor.py --modelo D:\\IA\\modelos\\...\\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
Abre http://127.0.0.1:8090 no navegador.
"""
import argparse
import base64
import json
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from motor import Motor            # noqa: E402
from conversa import Cerebro       # noqa: E402

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)
    del LOG[:-200]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def estado(c, ativos=None):
    mem = c.m.mem
    f = mem.forca
    fn = np.clip(np.log1p(f) / np.log1p(max(1e-6, float(f.max()))) * 255, 0, 255) if f.max() > 0 else f * 0
    return {"nivel": b64(mem.nivel), "forca": b64(fn), "vital": np.argwhere(mem.vital).tolist(),
            "ativos": ativos or [], "eventos": mem.eventos[-400:], "stats": mem.stats,
            "taxa": mem.taxa(), "vram": len(mem.vram), "ram": len(mem.ram),
            "cap_vram": mem.cap_vram, "cap_ram": mem.cap_ram, "modo": c.m.modo, "pos": c.m.pos, "ctx": c.m.ctx}


class App(BaseHTTPRequestHandler):
    cerebro = None
    trava = threading.Lock()

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _corpo(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        c = self.cerebro
        if self.path in ("/", "/index.html"):
            b = (AQUI / "cerebro.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/info":
            self._json({"modelo": c.m.nome, "L": c.m.L, "E": c.m.E, "k": c.m.k, "log": LOG[-30:],
                        "estado": estado(c)})
        else:
            self.send_error(404)

    def do_POST(self):
        c = self.cerebro
        if self.path == "/modo":
            c.m.modo = self._corpo().get("modo", "fiel")
            return self._json({"modo": c.m.modo})
        if self.path == "/dormir":
            with self.trava:
                c.m.mem.consolidar()
            log("Memória consolidada e salva (sono).")
            return self._json({"ok": True})
        if self.path == "/nova":
            with self.trava:
                c.nova_conversa()
            return self._json({"ok": True})
        if self.path != "/chat":
            return self.send_error(404)
        msg = self._corpo().get("msg", "")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def ev(tipo, dado):
            try:
                self.wfile.write(f"event: {tipo}\ndata: {json.dumps(dado)}\n\n".encode())
                self.wfile.flush()
            except OSError:
                pass

        ultimo = [0.0]

        def cada(pedaco, info):
            ev("texto", {"t": pedaco, "tok_s": round(info["tok_s"], 2), "n": info["n"]})
            if time.time() - ultimo[0] > 0.4:
                ultimo[0] = time.time()
                ev("mapa", estado(c, info["ativos"]))

        with self.trava:
            try:
                ev("aviso", {"t": "lendo a pergunta..."})
                texto, info = c.responder(msg, max_tokens=self.server.max_tokens, a_cada_token=cada)
                ev("mapa", estado(c))
                ev("fim", info)
            except Exception as e:
                traceback.print_exc()
                ev("erro", {"t": str(e)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, help="arquivo GGUF (Qwen3 MoE); para modelo dividido, a parte 00001")
    ap.add_argument("--vram", type=float, default=None, help="GB da placa pro cérebro (padrão: o que estiver livre)")
    ap.add_argument("--ram", type=float, default=10.0, help="GB de RAM pra memória recente")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--memoria", default=None, help="arquivo da memória consolidada")
    ap.add_argument("--modo", default="fiel", choices=["fiel", "plastico"])
    ap.add_argument("--porta", type=int, default=8090)
    ap.add_argument("--sem-navegador", action="store_true")
    a = ap.parse_args()

    memoria = a.memoria or str(AQUI / "memoria" / (Path(a.modelo).stem + ".json"))
    m = Motor(a.modelo, vram_gb=a.vram, ram_gb=a.ram, ctx=a.ctx, memoria=memoria, modo=a.modo, log=log)
    App.cerebro = Cerebro(m, sistema="Você é um assistente útil. Responda em português do Brasil. "
                                     "Escreva fórmulas em LaTeX entre $...$ ou $$...$$.")
    srv = ThreadingHTTPServer(("127.0.0.1", a.porta), App)
    srv.max_tokens = a.max_tokens
    url = f"http://127.0.0.1:{a.porta}"
    log(f"Pronto! Abra {url}")
    if not a.sem_navegador:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        m.mem.consolidar()
        log("Memória consolidada salva. Até mais!")


if __name__ == "__main__":
    main()
