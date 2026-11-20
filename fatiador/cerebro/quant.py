"""
Desquantização dos pesos do GGUF direto na placa de vídeo (PyTorch).

O GGUF guarda os pesos comprimidos em blocos (Q4_K, Q6_K...). Aqui cada formato é aberto
com operações de tensor do PyTorch, então um especialista inteiro é descomprimido na GPU
em menos de 1 ms. As fórmulas são as mesmas do gguf-py (llama.cpp), só trocando numpy
por torch. Formatos sem versão em torch caem no gguf-py (CPU, mais lento, mas correto).
"""
import numpy as np
import torch
from gguf import GGMLQuantizationType as T, GGML_QUANT_SIZES
from gguf import quants as gq

QK_K = 256
_CONST = {}


def _c(valores, dev, forma):
    """Tensor pequeno constante, criado uma vez por dispositivo (criar a cada chamada trava a GPU)."""
    chave = (valores, str(dev), forma)
    t = _CONST.get(chave)
    if t is None:
        t = _CONST[chave] = torch.tensor(valores, dtype=torch.uint8, device=dev).reshape(forma)
    return t


def _f16(b):
    """bytes (..., 2) -> float32 (..., 1)"""
    return b.contiguous().view(torch.float16).float()


def _q8_0(b):
    d, x = b[:, :2], b[:, 2:]
    return x.contiguous().view(torch.int8).float() * _f16(d)


def _q4_0(b):
    n = b.shape[0]
    d, qs = _f16(b[:, :2]), b[:, 2:]
    q = torch.stack([qs & 0x0F, qs >> 4], dim=1).reshape(n, 32).to(torch.int8) - 8
    return d * q.float()


def _q2_k(b):
    n = b.shape[0]
    scales, qs = b[:, :16], b[:, 16:80]
    d, dmin = _f16(b[:, 80:82]), _f16(b[:, 82:84])
    dl = (d * (scales & 0xF).float()).reshape(n, 16, 1)
    ml = (dmin * (scales >> 4).float()).reshape(n, 16, 1)
    shift = _c((0, 2, 4, 6), b.device, (1, 1, 4, 1))
    q = ((qs.reshape(n, -1, 1, 32) >> shift) & 3).reshape(n, 16, 16).float()
    return (dl * q - ml).reshape(n, QK_K)


def _q3_k(b):
    n = b.shape[0]
    dev = b.device
    hmask, qs, scales, d = b[:, :32], b[:, 32:96], b[:, 96:108], _f16(b[:, 108:110])
    ls, hs = scales[:, :8], scales[:, 8:]
    ls = (ls.reshape(n, 1, 8) >> _c((0, 4), dev, (1, 2, 1))).reshape(n, 16)
    hs = (hs.reshape(n, 1, 4) >> _c((0, 2, 4, 6), dev, (1, 4, 1))).reshape(n, 16)
    sc = ((ls & 0x0F) | ((hs & 0x03) << 4)).to(torch.int16) - 32
    dl = (d * sc.float()).reshape(n, 16, 1)
    ql = qs.reshape(n, -1, 1, 32) >> _c((0, 2, 4, 6), dev, (1, 1, 4, 1))
    qh = hmask.reshape(n, -1, 1, 32) >> _c(tuple(range(8)), dev, (1, 1, 8, 1))
    ql = (ql.reshape(n, 16, 16) & 3).to(torch.int16)
    qh = ((qh.reshape(n, 16, 16) & 1) ^ 1).to(torch.int16)
    q = (ql - (qh << 2)).float()
    return (dl * q).reshape(n, QK_K)


def _scale_min(scales):
    n = scales.shape[0]
    s = scales.reshape(n, 3, 4)
    d, m, md = s[:, 0], s[:, 1], s[:, 2]
    sc = torch.cat([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.float(), mn.float()


def _q4_k(b):
    n = b.shape[0]
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, mn = _scale_min(b[:, 4:16])
    qs = b[:, 16:]
    d = (d * sc).reshape(n, -1, 1)
    dm = (dmin * mn).reshape(n, -1, 1)
    q = qs.reshape(n, -1, 1, 32) >> _c((0, 4), b.device, (1, 1, 2, 1))
    q = (q & 0x0F).reshape(n, -1, 32).float()
    return (d * q - dm).reshape(n, QK_K)


def _q5_k(b):
    n = b.shape[0]
    dev = b.device
    d, dmin = _f16(b[:, :2]), _f16(b[:, 2:4])
    sc, mn = _scale_min(b[:, 4:16])
    qh, qs = b[:, 16:48], b[:, 48:]
    d = (d * sc).reshape(n, -1, 1)
    dm = (dmin * mn).reshape(n, -1, 1)
    ql = qs.reshape(n, -1, 1, 32) >> _c((0, 4), dev, (1, 1, 2, 1))
    qh = qh.reshape(n, -1, 1, 32) >> _c(tuple(range(8)), dev, (1, 1, 8, 1))
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = (qh & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).float()
    return (d * q - dm).reshape(n, QK_K)


def _q6_k(b):
    n = b.shape[0]
    dev = b.device
    ql, qh, scales, d = b[:, :128], b[:, 128:192], b[:, 192:208], _f16(b[:, 208:210])
    d = (d * scales.contiguous().view(torch.int8).float()).reshape(n, 16, 1)
    ql = ql.reshape(n, -1, 1, 64) >> _c((0, 4), dev, (1, 1, 2, 1))
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = qh.reshape(n, -1, 1, 32) >> _c((0, 2, 4, 6), dev, (1, 1, 4, 1))
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.int16) - 32
    return (d * q.reshape(n, 16, -1).float()).reshape(n, QK_K)


FUNCS = {T.Q8_0: _q8_0, T.Q4_0: _q4_0, T.Q2_K: _q2_k, T.Q3_K: _q3_k,
         T.Q4_K: _q4_k, T.Q5_K: _q5_k, T.Q6_K: _q6_k}


def bytes_por_linha(qtype, n_colunas):
    bs, ts = GGML_QUANT_SIZES[qtype]
    return n_colunas // bs * ts


def desquantizar(dados, qtype, forma, device="cpu", dtype=torch.float16):
    """
    dados: tensor uint8 (ou numpy) com os bytes de uma matriz (linhas x bytes_por_linha);
    forma: (linhas, colunas) no formato do PyTorch (saída, entrada).
    Devolve tensor float `dtype` em `device`.
    """
    if isinstance(dados, np.ndarray):
        if qtype in (T.F32, T.F16, T.BF16) and dados.dtype != np.uint8:
            return torch.from_numpy(np.ascontiguousarray(dados)).reshape(forma).to(device=device, dtype=dtype)
        dados = torch.from_numpy(np.ascontiguousarray(dados).view(np.uint8))
    b = dados.to(device, non_blocking=True)
    if qtype == T.F32:
        return b.contiguous().view(torch.float32).reshape(forma).to(dtype)
    if qtype == T.F16:
        return b.contiguous().view(torch.float16).reshape(forma).to(dtype)
    if qtype == T.BF16:
        return b.contiguous().view(torch.bfloat16).reshape(forma).to(dtype)
    bs, ts = GGML_QUANT_SIZES[qtype]
    blocos = b.reshape(-1, ts)
    f = FUNCS.get(qtype)
    if f is None:           # formato raro: usa a implementação de referência (numpy, CPU)
        arr = gq.dequantize(blocos.cpu().numpy(), qtype)
        return torch.from_numpy(arr).reshape(forma).to(device=device, dtype=dtype)
    return f(blocos).reshape(forma).to(dtype)
