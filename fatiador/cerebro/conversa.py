"""
Tokenizador (lido do próprio GGUF), formato de conversa do Qwen e geração de texto.
"""
import time

import numpy as np
import torch
from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders, AddedToken

# expressão de pré-tokenização do Qwen2/Qwen3 (a mesma do tokenizer.json oficial)
QWEN_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def tokenizador_do_gguf(g):
    toks = g.valor("tokenizer.ggml.tokens")
    tipos = g.valor("tokenizer.ggml.token_type")
    merges = g.valor("tokenizer.ggml.merges") or []
    vocab = {t: i for i, t in enumerate(toks)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[tuple(m.split(" ", 1)) for m in merges],
                              ignore_merges=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    especiais = [AddedToken(t, special=True, normalized=False) for t, ty in zip(toks, tipos) if ty in (3, 4)]
    tk.add_special_tokens(especiais)
    return tk


class Cerebro:
    """Junta motor + tokenizador + conversa com memória (o KV cache continua entre as mensagens)."""

    def __init__(self, motor, sistema=None):
        self.m = motor
        self.tk = tokenizador_do_gguf(motor.g)
        self.fim = {self.tk.token_to_id(t) for t in ("<|im_end|>", "<|endoftext|>")} - {None}
        self.sistema = sistema
        self.pendente = []           # tokens que ainda não passaram pelo modelo (ex.: o <|im_end|> final)
        self.historico = []

    def nova_conversa(self):
        self.m.reiniciar()
        self.pendente = []
        self.historico = []

    def _prompt(self, msg):
        t = ""
        if self.m.pos == 0 and not self.pendente and self.sistema:
            t += f"<|im_start|>system\n{self.sistema}<|im_end|>\n"
        if self.m.pos > 0:
            t += "\n"
        t += f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        return t

    def responder(self, msg, max_tokens=1024, temperatura=0.7, top_p=0.8, top_k=20,
                  reorganizar_cada=16, a_cada_token=None):
        """Gera a resposta; chama a_cada_token(texto_novo, info) a cada palavra."""
        ids = self.pendente + self.tk.encode(self._prompt(msg), add_special_tokens=False).ids
        self.pendente = []
        if self.m.pos + len(ids) + max_tokens > self.m.ctx:
            self.nova_conversa()
            ids = self.tk.encode(self._prompt(msg), add_special_tokens=False).ids
        t0 = time.time()
        logits, ativos = self.m.passo(ids)
        self.m.mem.reorganizar()
        t_leitura = time.time() - t0
        gerados, texto = [], ""
        t1 = time.time()
        for i in range(max_tokens):
            tok = amostrar(logits, temperatura, top_p, top_k)
            if tok in self.fim:
                self.pendente = [tok]
                break
            gerados.append(tok)
            novo = self.tk.decode(gerados)
            pedaco, texto = novo[len(texto):], novo
            if a_cada_token:
                a_cada_token(pedaco, {"ativos": ativos, "n": i + 1,
                                      "tok_s": (i + 1) / max(1e-6, time.time() - t1)})
            logits, ativos = self.m.passo([tok])
            if (i + 1) % reorganizar_cada == 0:
                self.m.mem.reorganizar()
        self.m.mem.reorganizar()
        self.historico.append((msg, texto))
        dt = time.time() - t1
        return texto, {"tokens": len(gerados), "tok_s": len(gerados) / max(1e-6, dt),
                       "leitura_s": t_leitura, "pergunta_tokens": len(ids)}


def amostrar(logits, temperatura, top_p, top_k):
    if temperatura <= 0:
        return int(logits.argmax())
    lg = logits.float() / temperatura
    v, i = lg.topk(min(top_k, lg.numel()))
    p = v.softmax(-1)
    acum = p.cumsum(-1)
    manter = acum - p < top_p
    p = p * manter
    p = p / p.sum()
    return int(i[torch.multinomial(p, 1)])
