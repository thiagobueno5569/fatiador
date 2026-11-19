"""
Passo 0 — baixa textos da Wikipédia em português, separados por assunto.

Para cada assunto:
  textos/<assunto>/calibracao.txt  -> usado pra descobrir quais especialistas o assunto usa
  textos/<assunto>/avaliacao.txt   -> artigos DIFERENTES, usados só pra medir a qualidade

Uso:  python 0_baixar_textos.py
Pode editar a lista ASSUNTOS à vontade (títulos que não existirem são pulados).
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Os 2 últimos títulos de cada assunto vão pra avaliação; o resto pra calibração.
ASSUNTOS = {
    "eletronica": ["Eletrônica", "Resistor", "Capacitor", "Indutor", "Diodo", "Transistor",
                   "Amplificador operacional", "Circuito RLC", "Lei de Ohm", "Leis de Kirchhoff",
                   "Transistor de efeito de campo", "Retificador"],
    "calculo": ["Cálculo", "Derivada", "Integral", "Limite de uma função", "Regra da cadeia",
                "Série de Taylor", "Equação diferencial", "Integração por partes",
                "Teorema fundamental do cálculo", "Transformada de Laplace",
                "Teorema do valor médio", "Série de Fourier"],
    "programacao": ["Linguagem de programação", "Python", "Algoritmo", "Estrutura de dados",
                    "Programação orientada a objetos", "Recursividade (ciência da computação)",
                    "Compilador", "Java (linguagem de programação)", "Algoritmo de ordenação",
                    "Git", "C (linguagem de programação)", "Árvore binária"],
    "culinaria": ["Culinária do Brasil", "Feijoada", "Pão de queijo", "Culinária de Minas Gerais",
                  "Churrasco", "Moqueca", "Brigadeiro", "Arroz", "Feijão", "Acarajé",
                  "Coxinha", "Tutu de feijão"],
    "futebol": ["Futebol", "Copa do Mundo FIFA", "Pelé", "Seleção Brasileira de Futebol",
                "Campeonato Brasileiro de Futebol - Série A", "Clube Atlético Mineiro",
                "Cruzeiro Esporte Clube", "Impedimento (futebol)", "Copa Libertadores da América",
                "Garrincha", "Maracanã", "Pênalti"],
    "historia": ["História do Brasil", "Independência do Brasil", "Inconfidência Mineira",
                 "Tiradentes", "Brasil Colônia", "Império do Brasil", "Proclamação da República do Brasil",
                 "Era Vargas", "Ciclo do ouro", "Abolição da escravatura no Brasil",
                 "Revolução de 1930", "Ditadura militar no Brasil"],
}

API = "https://pt.wikipedia.org/w/api.php"
CABECALHO = {"User-Agent": "FatiadorExperimento/1.1 (projeto de estudo universitario; python-urllib)"}
CACHE = Path("textos") / "_cache"


def baixar(titulo):
    """Baixa um artigo, esperando e tentando de novo se a Wikipédia pedir calma (erro 429)."""
    arq = CACHE / (re.sub(r'[\\/:*?"<>|]', "_", titulo) + ".txt")
    if arq.exists():
        return arq.read_text(encoding="utf-8")
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                "redirects": 1, "format": "json", "titles": titulo,
                                "maxlag": 5})
    espera = 5
    for tentativa in range(7):
        try:
            req = urllib.request.Request(f"{API}?{q}", headers=CABECALHO)
            with urllib.request.urlopen(req, timeout=30) as r:
                dados = json.load(r)
            if "error" in dados:            # maxlag: servidor ocupado
                raise urllib.error.HTTPError(API, 429, "maxlag", None, None)
            pagina = next(iter(dados["query"]["pages"].values()))
            texto = pagina.get("extract", "")
            CACHE.mkdir(parents=True, exist_ok=True)
            arq.write_text(texto, encoding="utf-8")
            time.sleep(1.5)                 # vai devagar pra não ser bloqueado
            return texto
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or tentativa == 6:
                raise
            pedido = e.headers.get("Retry-After") if e.headers else None
            t = int(pedido) if pedido and pedido.isdigit() else espera
            print(f"  ... Wikipédia pediu calma, esperando {t}s")
            time.sleep(t)
            espera = min(espera * 2, 120)


def limpar(texto):
    linhas = [l.strip() for l in texto.splitlines()]
    corte = {"== Ver também ==", "== Referências ==", "== Ligações externas ==", "== Bibliografia =="}
    saida = []
    for l in linhas:
        if l in corte:
            break
        if l and not l.startswith("=="):
            saida.append(l)
    return "\n".join(saida)


def main():
    raiz = Path("textos")
    for assunto, titulos in ASSUNTOS.items():
        pasta = raiz / assunto
        if (pasta / "calibracao.txt").exists() and (pasta / "avaliacao.txt").exists():
            print(f"{assunto:12s} já baixado, pulando")
            continue
        pasta.mkdir(parents=True, exist_ok=True)
        textos = []
        for t in titulos:
            try:
                txt = limpar(baixar(t))
            except Exception as e:
                print(f"  ! erro em '{t}': {e}")
                txt = ""
            if len(txt) < 500:
                print(f"  - pulando '{t}' (não achei ou é muito curto)")
            else:
                textos.append(txt)
        if len(textos) < 4:
            print(f"!! {assunto}: poucos artigos ({len(textos)}), adicione títulos na lista")
            continue
        (pasta / "calibracao.txt").write_text("\n\n".join(textos[:-2]), encoding="utf-8")
        (pasta / "avaliacao.txt").write_text("\n\n".join(textos[-2:]), encoding="utf-8")
        tam = sum(len(t) for t in textos)
        print(f"{assunto:12s} {len(textos):2d} artigos, {tam / 1000:.0f} mil caracteres")


if __name__ == "__main__":
    main()
