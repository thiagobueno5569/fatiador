"""
Passo 2 — cria uma "fatia" do modelo pra cada assunto.

Fatia = NÚCLEO (atenção, embeddings, normalizações: tudo que é comum, nunca é cortado)
      + só os N especialistas favoritos daquele assunto em cada camada.

Não precisa de GPU nem de muita RAM: lê o GGUF original direto do disco.

Uso:
  python 2_fatiar_gguf.py --gguf Qwen3-30B-A3B-Q4_K_M.gguf --manter 32
  python 2_fatiar_gguf.py --gguf ... --manter 48 --assuntos eletronica calculo
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import gguf


# "Super-especialistas": raríssimos, mas sem eles o modelo desmorona (Su et al., 2025,
# "Unveiling Super Experts in MoE LLMs"). Fazem parte do NÚCLEO e nunca são cortados.
SUPER_CONHECIDOS = {("qwen3moe", 128, 48): "1:68,2:92,3:82"}  # Qwen3-30B-A3B


def ler_sempre(texto):
    fixos = {}
    for par in filter(None, (texto or "").split(",")):
        camada, esp = par.split(":")
        fixos.setdefault(int(camada), set()).add(int(esp))
    return fixos


def escolher(contagens, manter, fixos=None):
    """Pra cada camada: os especialistas fixos (núcleo) + os mais usados até completar 'manter'."""
    fixos = fixos or {}
    saida = []
    for camada, c in enumerate(contagens):
        obrig = list(fixos.get(camada, set()))
        resto = [i for i in np.argsort(np.array(c))[::-1] if i not in obrig]
        saida.append(np.sort(np.array(obrig + resto[: manter - len(obrig)], dtype=np.int64)))
    return saida


def fatiar(caminho_gguf, escolhidos, saida):
    leitor = gguf.GGUFReader(caminho_gguf)
    arq = leitor.fields[gguf.Keys.General.ARCHITECTURE].contents()
    chave_n = f"{arq}.expert_count"
    n_orig = int(leitor.fields[chave_n].contents())
    k_usados = int(leitor.fields[f"{arq}.expert_used_count"].contents())
    manter = len(escolhidos[0])
    if manter < k_usados:
        raise SystemExit(f"--manter precisa ser >= {k_usados} (especialistas ativos por token)")

    escritor = gguf.GGUFWriter(saida, arq, endianess=leitor.endianess)
    for campo in leitor.fields.values():
        if campo.name == gguf.Keys.General.ARCHITECTURE or campo.name.startswith("GGUF."):
            continue
        tipo = campo.types[0]
        sub = campo.types[-1] if tipo == gguf.GGUFValueType.ARRAY else None
        valor = manter if campo.name == chave_n else campo.contents()
        if campo.name == "general.name":
            valor = f"{valor} (fatia {Path(saida).stem})"
        escritor.add_key_value(campo.name, valor, tipo, sub_type=sub)

    tensores = []
    for t in leitor.tensors:
        dados = t.data
        m = re.match(r"blk\.(\d+)\.", t.name)
        eh_especialista = m and dados.shape[0] == n_orig and (
            "_exps" in t.name or "ffn_gate_inp" in t.name or "exp_probs" in t.name)
        if eh_especialista:
            dados = np.ascontiguousarray(dados[escolhidos[int(m.group(1))]])
        tensores.append((t, dados))
        escritor.add_tensor_info(t.name, dados.shape, dados.dtype, dados.nbytes, t.tensor_type)

    escritor.write_header_to_file()
    escritor.write_kv_data_to_file()
    escritor.write_ti_data_to_file()
    for t, dados in tensores:
        escritor.write_tensor_data(dados, tensor_endianess=leitor.endianess)
    escritor.close()
    return n_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="modelo original em GGUF")
    ap.add_argument("--especialistas", default="especialistas.json")
    ap.add_argument("--manter", type=int, default=32, help="especialistas por camada na fatia")
    ap.add_argument("--assuntos", nargs="*", help="padrão: todos do json")
    ap.add_argument("--saida", default="fatias")
    ap.add_argument("--sempre", default="auto",
                    help='especialistas que nunca são cortados, ex. "1:68,2:92"; "auto" = conhecidos; "" = nenhum')
    a = ap.parse_args()

    info = json.loads(Path(a.especialistas).read_text(encoding="utf-8"))
    if a.sempre == "auto" and info.get("super") is not None:
        a.sempre = info["super"]
    if a.sempre == "auto":
        lt = gguf.GGUFReader(a.gguf)
        arq = lt.fields[gguf.Keys.General.ARCHITECTURE].contents()
        chave = (arq, int(lt.fields[f"{arq}.expert_count"].contents()),
                 int(lt.fields[f"{arq}.block_count"].contents()))
        a.sempre = SUPER_CONHECIDOS.get(chave, "")
        del lt
    fixos = ler_sempre(a.sempre)
    if fixos:
        print(f"Super-especialistas sempre mantidos (núcleo): {a.sempre}")
    Path(a.saida).mkdir(exist_ok=True)
    tam_orig = Path(a.gguf).stat().st_size / 1e9
    print(f"Original: {tam_orig:.1f} GB")
    for assunto in a.assuntos or list(info["assuntos"]):
        destino = Path(a.saida) / f"{assunto}-{a.manter}esp.gguf"
        n = fatiar(a.gguf, escolher(info["assuntos"][assunto], a.manter, fixos), str(destino))
        tam = destino.stat().st_size / 1e9
        print(f"{assunto:12s} -> {destino}  {tam:.1f} GB  ({a.manter}/{n} especialistas, "
              f"{tam / tam_orig:.0%} do tamanho)")


if __name__ == "__main__":
    main()
