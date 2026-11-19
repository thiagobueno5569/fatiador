"""
Passo 6 — textos de calibração no FORMATO DE CONVERSA do modelo.

Por quê: medindo só com texto de enciclopédia, os especialistas que cuidam do formato de
chat (marcadores <|start|>, <|channel|>, fase de raciocínio "analysis" -> resposta "final")
quase não aparecem e acabam cortados. A fatia entende o assunto, mas se perde na conversa.

Aqui cada parágrafo da Wikipédia vira uma conversa: pergunta do usuário -> raciocínio curto
-> resposta. O resultado vai pra textos/<assunto>/calibracao_chat.txt (os textos originais
ficam intactos) e depois é medido com o llama-imatrix usando --parse-special.

Uso: python 6_calibrar_chat.py --gguf MODELO.gguf --textos gptoss/textos
"""
import argparse
import random
import re
from pathlib import Path

import gguf

PERGUNTAS = [
    "Explique de forma clara: {t}",
    "O que você sabe sobre isto? {t}",
    "Me ajuda a entender esse assunto: {t}",
    "Pode resumir e explicar o seguinte tema: {t}",
    "Estou estudando isso e não entendi direito: {t}. Pode explicar?",
    "Qual a ideia principal disto: {t}",
]
RACIOCINIO = [
    "O usuário quer entender {t}. Vou explicar em português, de forma organizada e correta.",
    "Preciso explicar {t}. Vou dar a definição e os pontos principais, em português.",
    "Pergunta sobre {t}. Responder com clareza, sem enrolar, em português.",
]

SISTEMA_HARMONY = ("You are ChatGPT, a large language model trained by OpenAI.\n"
                   "Knowledge cutoff: 2024-06\nCurrent date: 2026-11-19\n\nReasoning: low\n\n"
                   "# Valid channels: analysis, commentary, final. Channel must be included for every message.")


def conversa_harmony(q, r, f):
    return (f"<|start|>system<|message|>{SISTEMA_HARMONY}<|end|>"
            f"<|start|>user<|message|>{q}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{r}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{f}<|return|>\n")


def conversa_chatml(q, r, f):
    return ("<|im_start|>system\nVocê é um assistente útil.<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{f}<|im_end|>\n")


def tema(par):
    palavras = re.findall(r"\S+", par)[:9]
    return " ".join(palavras).rstrip(",.;:") + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--textos", default="textos")
    a = ap.parse_args()

    arq = gguf.GGUFReader(a.gguf).fields["general.architecture"].contents()
    fazer = conversa_harmony if arq == "gpt-oss" else conversa_chatml
    print(f"Arquitetura {arq}: formato {'harmony (gpt-oss)' if arq == 'gpt-oss' else 'chatml'}")
    rnd = random.Random(0)
    for pasta in sorted(Path(a.textos).iterdir()):
        orig = pasta / "calibracao.txt"
        if not orig.exists():
            continue
        pars = [p.strip() for p in orig.read_text(encoding="utf-8").split("\n") if len(p.strip()) > 200]
        saida = []
        for p in pars:
            t = tema(p)
            saida.append(fazer(rnd.choice(PERGUNTAS).format(t=t), rnd.choice(RACIOCINIO).format(t=t), p))
        (pasta / "calibracao_chat.txt").write_text("".join(saida), encoding="utf-8")
        print(f"{pasta.name:12s} {len(saida)} conversas")


if __name__ == "__main__":
    main()
