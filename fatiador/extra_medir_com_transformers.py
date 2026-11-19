"""
ALTERNATIVA ao passo 1 (via Hugging Face, precisa de GPU grande) — descobre quais especialistas (experts) cada assunto usa, camada por camada.

Passa os textos de calibração de cada assunto pelo modelo e conta, em cada camada,
quantas vezes o roteador escolheu cada especialista. Isso são os "caminhos" de cada assunto.
Resultado: especialistas.json (pequeno, uns poucos MB).

Onde rodar (o Qwen3-30B-A3B completo tem ~61 GB em 16 bits, não cabe no PC):
  * Kaggle (grátis, 2 GPUs T4 = 32 GB), versão já quantizada em 4 bits (~17 GB de download):
        python 1_medir_especialistas.py --modelo unsloth/Qwen3-30B-A3B-bnb-4bit --modo gpu
  * Se esse repositório não existir mais, quantiza na hora (baixa ~61 GB):
        python 1_medir_especialistas.py --modelo Qwen/Qwen3-30B-A3B --modo 4bit
  * Teste rápido na CPU com modelo pequeno: --modo cpu --modelo <pasta>
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def carregar(modelo, modo):
    tok = AutoTokenizer.from_pretrained(modelo)
    if modo == "4bit":
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForCausalLM.from_pretrained(modelo, quantization_config=cfg, device_map="auto")
    elif modo == "gpu":   # usa o formato que o repositório já tem (ex.: versão bnb-4bit pronta)
        m = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.bfloat16, device_map="auto")
    else:
        m = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.float32, device_map="cpu")
    m.eval()
    return tok, m


def info_moe(config):
    cfg = getattr(config, "text_config", config)
    n = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    k = getattr(cfg, "num_experts_per_tok", None)
    if not n or not k:
        raise SystemExit("Esse modelo não parece ser Mixture of Experts.")
    return int(n), int(k)


def instalar_ganchos(modelo, n_exp, top_k, registro):
    """Pendura um 'grampo' em cada roteador pra anotar quais especialistas ele escolheu."""
    ganchos = []
    for nome, mod in modelo.named_modules():
        classe = type(mod).__name__
        eh_roteador = classe.endswith("TopKRouter") or (nome.endswith("mlp.gate") and
                                                        isinstance(mod, torch.nn.Linear))
        if not eh_roteador:
            continue
        camada = int(re.search(r"layers\.(\d+)\.", nome).group(1))

        def gancho(_m, _ent, saida, camada=camada):
            if isinstance(saida, tuple) and len(saida) >= 3:     # transformers 5.x
                idx = saida[2]
            else:                                                # transformers 4.x (logits)
                logits = saida[0] if isinstance(saida, tuple) else saida
                idx = torch.topk(logits.float(), top_k, dim=-1).indices
            cont = torch.bincount(idx.reshape(-1).cpu(), minlength=n_exp).numpy()
            registro[camada] = registro.get(camada, 0) + cont

        ganchos.append(mod.register_forward_hook(gancho))
    if not ganchos:
        raise SystemExit("Não achei os roteadores no modelo (versão do transformers diferente?)")
    return ganchos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="unsloth/Qwen3-30B-A3B-bnb-4bit")
    ap.add_argument("--modo", choices=["gpu", "4bit", "cpu"], default="gpu")
    ap.add_argument("--textos", default="textos")
    ap.add_argument("--tokens", type=int, default=16000, help="tokens de calibração por assunto")
    ap.add_argument("--janela", type=int, default=1024)
    ap.add_argument("--saida", default="especialistas.json")
    a = ap.parse_args()

    tok, modelo = carregar(a.modelo, a.modo)
    n_exp, top_k = info_moe(modelo.config)
    print(f"Modelo: {a.modelo} | {n_exp} especialistas por camada, {top_k} ativos por token")
    disp = next(modelo.parameters()).device

    resultado = {"modelo": a.modelo, "num_especialistas": n_exp, "top_k": top_k, "assuntos": {}}
    for pasta in sorted(Path(a.textos).iterdir()):
        arq = pasta / "calibracao.txt"
        if not arq.exists():
            continue
        ids = tok(arq.read_text(encoding="utf-8"), return_tensors="pt").input_ids[0][: a.tokens]
        registro = {}
        ganchos = instalar_ganchos(modelo, n_exp, top_k, registro)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(ids), a.janela):
                pedaco = ids[i: i + a.janela].unsqueeze(0).to(disp)
                if pedaco.shape[1] > 8:
                    modelo(pedaco)
        for g in ganchos:
            g.remove()
        camadas = [registro[c].tolist() for c in sorted(registro)]
        resultado["assuntos"][pasta.name] = camadas
        usados = np.mean([(np.array(c) > 0).mean() for c in camadas])
        print(f"{pasta.name:12s} {len(ids):6d} tokens em {time.time() - t0:5.0f}s | "
              f"usou {usados:.0%} dos especialistas pelo menos uma vez")

    Path(a.saida).write_text(json.dumps(resultado), encoding="utf-8")
    print(f"\nSalvo em {a.saida}. Agora rode o 2_fatiar_gguf.py no seu PC.")
    resumo(resultado)


def resumo(r):
    """Quanto os assuntos se sobrepõem: se cada um usa caminhos próprios, a ideia funciona."""
    nomes = list(r["assuntos"])
    q = max(r["top_k"], r["num_especialistas"] // 4)
    print(f"\nSobreposição dos {q} especialistas favoritos por camada (100% = mesmos caminhos):")
    print(" " * 13 + "".join(f"{n[:6]:>8s}" for n in nomes))
    tops = {n: [set(np.argsort(c)[::-1][:q]) for c in r["assuntos"][n]] for n in nomes}
    for a in nomes:
        linha = [np.mean([len(x & y) / q for x, y in zip(tops[a], tops[b])]) for b in nomes]
        print(f"{a[:12]:13s}" + "".join(f"{v:8.0%}" for v in linha))


if __name__ == "__main__":
    main()
