"""
Passo 8 — calibração com MATEMÁTICA ESCRITA (LaTeX), no formato de conversa do modelo.

Por quê: as fatias calibradas com Wikipédia + conversas entendiam o assunto, mas desmontavam
na hora de escrever fórmula (\\frac, ^{}, \\int ficavam quebrados, contas erradas). Os
especialistas que escrevem LaTeX quase não apareciam na medição e eram cortados.

Aqui o computador gera exercícios de cálculo (derivadas, integrais, limites, regra da cadeia,
produto, quociente...) com a resolução passo a passo em LaTeX. Todas as contas são feitas e
conferidas pelo SymPy, então a calibração não ensina nada errado. As conversas novas são
intercaladas com as conversas de texto já existentes (calibracao_chat.txt), pra fatia não
perder o formato de chat.

Uso:
  python 8_calibrar_latex.py --chat textos/calculo/calibracao_chat.txt \\
      --saida textos_latex/calculo/calibracao_latex.txt
"""
import argparse
import random
from pathlib import Path

import sympy as sp

x, h = sp.symbols("x h")
SISTEMA = ("You are ChatGPT, a large language model trained by OpenAI.\n"
           "Knowledge cutoff: 2024-06\nCurrent date: 2026-11-19\n\nReasoning: low\n\n"
           "# Valid channels: analysis, commentary, final. Channel must be included for every message.")


def conversa(q, r, f):
    return (f"<|start|>system<|message|>{SISTEMA}<|end|>"
            f"<|start|>user<|message|>{q}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{r}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{f}<|return|>\n")


def L(e):
    return sp.latex(e, ln_notation=True)


def coef(rnd, a=1, b=9):
    return rnd.choice([i for i in range(-b, b + 1) if abs(i) >= a])


def funcao(rnd):
    """Uma função 'de exercício' variada."""
    a, b, c = coef(rnd), coef(rnd), rnd.randint(1, 9)
    n = rnd.randint(2, 6)
    tipo = rnd.choice(["poli", "exp", "trig", "log", "raiz", "produto", "cadeia", "quociente"])
    if tipo == "poli":
        return tipo, a * x**n + b * x**(n - 1) + c
    if tipo == "exp":
        return tipo, sp.exp(a * x + b)
    if tipo == "trig":
        return tipo, rnd.choice([sp.sin, sp.cos])(a * x) + b * x
    if tipo == "log":
        return tipo, sp.log(abs(a) * x**2 + c)
    if tipo == "raiz":
        return tipo, sp.sqrt(abs(a) * x + c)
    if tipo == "produto":
        return tipo, x**n * rnd.choice([sp.sin(x), sp.exp(x), sp.cos(x)])
    if tipo == "cadeia":
        return tipo, (a * x**2 + b)**n
    return tipo, (a * x + b) / (x**2 + c)


REGRAS = {
    "poli": ("regra da potência", r"\frac{d}{dx}\,x^{n} = n\,x^{n-1}"),
    "exp": ("derivada da exponencial com a regra da cadeia", r"\frac{d}{dx}\,e^{u} = e^{u}\,u'"),
    "trig": ("derivadas de seno e cosseno", r"(\sin u)' = \cos u \cdot u', \quad (\cos u)' = -\sin u \cdot u'"),
    "log": ("derivada do logaritmo com a regra da cadeia", r"\frac{d}{dx}\ln u = \frac{u'}{u}"),
    "raiz": ("derivada da raiz com a regra da cadeia", r"\frac{d}{dx}\sqrt{u} = \frac{u'}{2\sqrt{u}}"),
    "produto": ("regra do produto", r"(f\,g)' = f'\,g + f\,g'"),
    "cadeia": ("regra da cadeia", r"\frac{d}{dx}\,[u(x)]^{n} = n\,[u(x)]^{n-1}\,u'(x)"),
    "quociente": ("regra do quociente", r"\left(\frac{f}{g}\right)' = \frac{f'\,g - f\,g'}{g^{2}}"),
}


def derivada(rnd):
    tipo, f = funcao(rnd)
    d = sp.simplify(sp.diff(f, x))
    nome, regra = REGRAS[tipo]
    q = rnd.choice([f"Qual a derivada de $f(x) = {L(f)}$?",
                    f"Derive f(x) = {sp.sstr(f)} em relação a x. Mostre a conta.",
                    f"Calcule $\\frac{{d}}{{dx}}\\left({L(f)}\\right)$ passo a passo.",
                    f"me ajuda a derivar {sp.sstr(f)}"])
    r = (f"Differentiate f(x) = {sp.sstr(f)} using the {nome}. "
         f"Result: f'(x) = {sp.sstr(d)}. Answer in Portuguese with LaTeX.")
    resp = (f"**Função:**\n\n\\[\nf(x) = {L(f)}\n\\]\n\n"
            f"### 1. Regra usada\n\nVamos usar a **{nome}**:\n\n\\[\n{regra}\n\\]\n\n"
            f"### 2. Aplicando\n\n\\[\n\\begin{{aligned}}\nf'(x) &= \\frac{{d}}{{dx}}\\left[{L(f)}\\right] \\\\\n"
            f"&= {L(sp.diff(f, x))} \\\\\n&= {L(d)}\n\\end{{aligned}}\n\\]\n\n"
            f"## Resultado\n\n\\[\n\\boxed{{f'(x) = {L(d)}}}\n\\]\n")
    return conversa(q, r, resp)


def integral_definida(rnd):
    a, b = rnd.randint(1, 6), rnd.randint(-5, 5)
    n = rnd.randint(1, 4)
    f = a * x**n + b
    ini, fim = rnd.randint(-2, 1), rnd.randint(2, 5)
    F = sp.integrate(f, x)
    v = sp.integrate(f, (x, ini, fim))
    q = rnd.choice([f"Quanto vale a integral de {sp.sstr(f)} de {ini} a {fim}? Mostre a conta.",
                    f"Calcule $\\int_{{{ini}}}^{{{fim}}} \\left({L(f)}\\right)\\,dx$.",
                    f"Calcule a integral definida de y = {sp.sstr(f)} entre x = {ini} e x = {fim}."])
    r = (f"Antiderivative of {sp.sstr(f)} is {sp.sstr(F)}. Evaluate from {ini} to {fim}: "
         f"F({fim}) - F({ini}) = {sp.sstr(F.subs(x, fim))} - ({sp.sstr(F.subs(x, ini))}) = {sp.sstr(v)}.")
    resp = (f"**Integral pedida**\n\n\\[\n\\int_{{{ini}}}^{{{fim}}} \\left({L(f)}\\right)\\,dx\n\\]\n\n"
            f"### 1. Antiderivada\n\nPela regra da potência, "
            f"$\\int x^{{n}}\\,dx = \\frac{{x^{{n+1}}}}{{n+1}} + C$:\n\n"
            f"\\[\nF(x) = {L(F)}\n\\]\n\n"
            f"### 2. Teorema Fundamental do Cálculo\n\n\\[\n\\begin{{aligned}}\n"
            f"\\int_{{{ini}}}^{{{fim}}} \\left({L(f)}\\right)dx &= F({fim}) - F({ini}) \\\\\n"
            f"&= {L(F.subs(x, fim))} - \\left({L(F.subs(x, ini))}\\right) \\\\\n&= {L(v)}\n\\end{{aligned}}\n\\]\n\n"
            f"## Resultado\n\n\\[\n\\boxed{{\\int_{{{ini}}}^{{{fim}}} \\left({L(f)}\\right)dx = {L(v)}}}\n\\]\n")
    return conversa(q, r, resp)


def integral_indefinida(rnd):
    tipo, f = rnd.choice([("poli", coef(rnd) * x**rnd.randint(1, 5) + coef(rnd) * x),
                          ("exp", sp.exp(coef(rnd) * x)),
                          ("trig", coef(rnd) * rnd.choice([sp.sin, sp.cos])(coef(rnd) * x)),
                          ("racional", sp.Integer(coef(rnd)) / x),
                          ("partes", x * rnd.choice([sp.exp(x), sp.sin(x), sp.cos(x)]))])
    F = sp.simplify(sp.integrate(f, x))
    metodo = {"partes": r"integração por partes, $\int u\,dv = u\,v - \int v\,du$",
              "racional": r"$\int \frac{1}{x}\,dx = \ln|x| + C$"}.get(tipo, "as integrais imediatas")
    q = rnd.choice([f"Calcule a integral indefinida de {sp.sstr(f)}.",
                    f"Resolva $\\int {L(f)}\\,dx$.", f"qual a primitiva de {sp.sstr(f)}?"])
    r = f"Integrate {sp.sstr(f)} dx. Result {sp.sstr(F)} + C. Check by differentiating."
    resp = (f"Queremos\n\n\\[\n\\int {L(f)}\\,dx\n\\]\n\nUsando {metodo}:\n\n"
            f"\\[\n\\int {L(f)}\\,dx = {L(F)} + C\n\\]\n\n"
            f"**Conferindo:** derivando o resultado,\n\n\\[\n\\frac{{d}}{{dx}}\\left({L(F)}\\right) = "
            f"{L(sp.simplify(sp.diff(F, x)))}\n\\]\n\nque é exatamente a função original. ✅\n")
    return conversa(q, r, resp)


def limite(rnd):
    a = rnd.randint(1, 5)
    f, ponto = rnd.choice([((x**2 - a**2) / (x - a), a), (sp.sin(a * x) / x, 0),
                           ((sp.exp(a * x) - 1) / x, 0), ((x**3 - a**3) / (x - a), a)])
    v = sp.limit(f, x, ponto)
    q = rnd.choice([f"Calcule o limite de {sp.sstr(f)} quando x tende a {ponto}.",
                    f"Quanto vale $\\lim_{{x \\to {ponto}}} {L(f)}$?"])
    r = f"Limit of {sp.sstr(f)} as x -> {ponto} is indeterminate 0/0; simplify or use L'Hospital. Value {v}."
    resp = (f"Substituindo direto $x = {ponto}$ aparece a forma indeterminada $\\frac{{0}}{{0}}$. "
            f"Usando a regra de L'Hôpital (derivando em cima e embaixo):\n\n"
            f"\\[\n\\lim_{{x \\to {ponto}}} {L(f)} = "
            f"\\lim_{{x \\to {ponto}}} \\frac{{{L(sp.diff(sp.numer(sp.together(f)), x))}}}"
            f"{{{L(sp.diff(sp.denom(sp.together(f)), x))}}} = {L(v)}\n\\]\n\n"
            f"## Resultado\n\n\\[\n\\boxed{{\\lim_{{x \\to {ponto}}} {L(f)} = {L(v)}}}\n\\]\n")
    return conversa(q, r, resp)


def definicao(rnd):
    a, b = coef(rnd), coef(rnd)
    f = a * x**2 + b * x
    dif = sp.expand(f.subs(x, x + h) - f)
    q = rnd.choice(["O que é a derivada de uma função? Dê um exemplo simples.",
                    f"Calcule a derivada de {sp.sstr(f)} pela definição de limite.",
                    "Explique a definição formal de derivada com um exemplo."])
    r = f"Explain derivative as limit of difference quotient; example f(x)={sp.sstr(f)}, f'(x)={sp.sstr(sp.diff(f, x))}."
    resp = (f"A **derivada** de $f$ mede a **taxa de variação instantânea** de $f$: é a inclinação da reta "
            f"tangente ao gráfico. Formalmente:\n\n\\[\nf'(x) = \\lim_{{h \\to 0}} \\frac{{f(x+h) - f(x)}}{{h}}\n\\]\n\n"
            f"### Exemplo: $f(x) = {L(f)}$\n\n\\[\n\\begin{{aligned}}\n"
            f"f(x+h) - f(x) &= {L(dif)} \\\\\n"
            f"\\frac{{f(x+h) - f(x)}}{{h}} &= {L(sp.expand(dif / h))} \\\\\n"
            f"f'(x) = \\lim_{{h \\to 0}} \\left({L(sp.expand(dif / h))}\\right) &= {L(sp.diff(f, x))}\n"
            f"\\end{{aligned}}\n\\]\n\n## Resultado\n\n\\[\n\\boxed{{f'(x) = {L(sp.diff(f, x))}}}\n\\]\n")
    return conversa(q, r, resp)


GERADORES = [derivada] * 4 + [integral_definida] * 3 + [integral_indefinida] * 2 + [limite, definicao]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", help="conversas de texto já existentes, pra intercalar")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--n", type=int, default=400, help="quantos exercícios gerar")
    a = ap.parse_args()

    rnd = random.Random(42)
    exercicios = []
    while len(exercicios) < a.n:
        try:
            exercicios.append(rnd.choice(GERADORES)(rnd))
        except Exception:
            pass           # algum caso degenerado do sorteio: só pula
    texto = []
    if a.chat:
        chat = [c + "<|return|>\n" for c in Path(a.chat).read_text(encoding="utf-8").split("<|return|>\n") if c.strip()]
        # 2 exercícios com LaTeX pra cada conversa de texto: o começo do arquivo (que é o que
        # o llama-imatrix lê) tem os dois tipos misturados
        while exercicios or chat:
            texto += exercicios[:2]; exercicios = exercicios[2:]
            texto += chat[:1]; chat = chat[1:]
    else:
        texto = exercicios
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text("".join(texto), encoding="utf-8")
    print(f"{len(texto)} conversas salvas em {a.saida}")


if __name__ == "__main__":
    main()
