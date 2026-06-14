import sympy as sp

# =========================================================
# 1. Symbols
# =========================================================

# Core parameters
lam, beta, t, R = sp.symbols('lam beta t R', real=True)
theta0, thetaR = sp.symbols('theta0 thetaR', real=True)
alphaT, alphaL0, alphaLR = sp.symbols('alphaT alphaL0 alphaLR', real=True)

# Income structure
mu, yl, yh = sp.symbols('mu yl yh', real=True)

# Marginal costs
cL, cT0, cTR = sp.symbols('cL cT0 cTR', real=True)

# Fixed costs
FT0, FTR, FL0, FLR = sp.symbols('FT0 FTR FL0 FLR', real=True)

# Prices
pT0, pTR, pL0, pLR = sp.symbols('pT0 pTR pL0 pLR', real=True)

# Auxiliary objects
ybar = sp.symbols('ybar', real=True)
dalpha0, dalphaR = sp.symbols('dalpha0 dalphaR', real=True)
K0, KR = sp.symbols('K0 KR', real=True)

# Dummy consumer location symbols
xT, x0, xR = sp.symbols('xT x0 xR', real=True)

# =========================================================
# 2. Definitions
# =========================================================

ybar_def = sp.Eq(ybar, (1 - mu) * yl + mu * yh)
dalpha0_def = sp.Eq(dalpha0, alphaL0 - alphaT)
dalphaR_def = sp.Eq(dalphaR, alphaLR - alphaT)

K0_def = sp.Eq(K0, theta0 + dalpha0 * ybar)
KR_def = sp.Eq(KR, thetaR + dalphaR * ybar)

print("\nDefinitions:")
sp.pprint(ybar_def)
sp.pprint(dalpha0_def)
sp.pprint(dalphaR_def)
sp.pprint(K0_def)
sp.pprint(KR_def)

# =========================================================
# 3. Marginal consumers
# =========================================================

xT_expr = sp.simplify((pTR - pT0 + t * R + beta) / (2 * t))
x0_expr = sp.simplify((pTR - pL0 + t * R + beta + K0) / (2 * t))
xR_expr = sp.simplify((pLR - pT0 + t * R + beta - KR) / (2 * t))

print("\nMarginal consumers:")
print("xT =")
sp.pprint(xT_expr)
print("x0 =")
sp.pprint(x0_expr)
print("xR =")
sp.pprint(xR_expr)

# =========================================================
# 4. Demands
# =========================================================

DT0_expr = sp.simplify((1 - 2 * lam) * xT_expr + lam * xR_expr)
DL0_expr = sp.simplify(lam * x0_expr)
DTR_expr = sp.simplify((1 - 2 * lam) * (R - xT_expr) + lam * (R - x0_expr))
DLR_expr = sp.simplify(lam * (R - xR_expr))

print("\nDemand system:")
print("DT0 =")
sp.pprint(sp.expand(DT0_expr))
print("\nDL0 =")
sp.pprint(sp.expand(DL0_expr))
print("\nDTR =")
sp.pprint(sp.expand(DTR_expr))
print("\nDLR =")
sp.pprint(sp.expand(DLR_expr))

# =========================================================
# 5. Profit functions
# =========================================================

pi0 = sp.expand((pT0 - cT0) * DT0_expr + (pL0 - cL) * DL0_expr - FT0 - FL0)
piR = sp.expand((pTR - cTR) * DTR_expr + (pLR - cL) * DLR_expr - FTR - FLR)

print("\nProfit functions:")
print("pi0 =")
sp.pprint(pi0)
print("\npiR =")
sp.pprint(piR)

# =========================================================
# 6. First-order conditions
# =========================================================

FOC_T0 = sp.simplify(sp.diff(pi0, pT0))
FOC_L0 = sp.simplify(sp.diff(pi0, pL0))
FOC_TR = sp.simplify(sp.diff(piR, pTR))
FOC_LR = sp.simplify(sp.diff(piR, pLR))

print("\nFirst-order conditions:")
print("FOC_T0 = 0:")
sp.pprint(FOC_T0)
print("\nFOC_L0 = 0:")
sp.pprint(FOC_L0)
print("\nFOC_TR = 0:")
sp.pprint(FOC_TR)
print("\nFOC_LR = 0:")
sp.pprint(FOC_LR)

# =========================================================
# 7. Reaction functions
# =========================================================

reaction_L0 = sp.solve(sp.Eq(FOC_L0, 0), pL0)[0]
reaction_LR = sp.solve(sp.Eq(FOC_LR, 0), pLR)[0]
reaction_T0 = sp.solve(sp.Eq(FOC_T0, 0), pT0)[0]
reaction_TR = sp.solve(sp.Eq(FOC_TR, 0), pTR)[0]

print("\nReaction functions:")
print("pL0 =")
sp.pprint(sp.simplify(reaction_L0))
print("\npLR =")
sp.pprint(sp.simplify(reaction_LR))
print("\npT0 =")
sp.pprint(sp.simplify(reaction_T0))
print("\npTR =")
sp.pprint(sp.simplify(reaction_TR))

# =========================================================
# 8. Solve the full system
# =========================================================

sol = sp.solve(
    [
        sp.Eq(pL0, reaction_L0),
        sp.Eq(pLR, reaction_LR),
        sp.Eq(pT0, reaction_T0),
        sp.Eq(pTR, reaction_TR)
    ],
    [pL0, pLR, pT0, pTR],
    dict=True,
    simplify=True,
    rational=True
)[0]

pL0_sol = sp.simplify(sol[pL0])
pLR_sol = sp.simplify(sol[pLR])
pT0_sol = sp.simplify(sol[pT0])
pTR_sol = sp.simplify(sol[pTR])

print("\nEquilibrium prices:")
print("\npT0 =")
sp.pprint(sp.factor(pT0_sol))
print("\npTR =")
sp.pprint(sp.factor(pTR_sol))
print("\npL0 =")
sp.pprint(sp.factor(pL0_sol))
print("\npLR =")
sp.pprint(sp.factor(pLR_sol))

# =========================================================
# 9. Traded-good price gap
# =========================================================

gap = sp.simplify(sp.expand(pT0_sol - pTR_sol))

print("\nTraded-good price gap:")
sp.pprint(sp.factor(gap))

gap_closed = sp.simplify(
    sp.Rational(2, 3) * beta
    + lam * (K0 - KR) / (3 * (2 - 3 * lam))
    + 2 * (1 - lam) * (cT0 - cTR) / (3 * (2 - 3 * lam))
)

print("\nClosed form conjecture:")
sp.pprint(gap_closed)

print("\nCheck gap - conjecture =")
sp.pprint(sp.simplify(gap - gap_closed))

# =========================================================
# 10. Expand K0 and KR
# =========================================================

gap_expanded = sp.simplify(
    gap.subs({
        K0: theta0 + (alphaL0 - alphaT) * ybar,
        KR: thetaR + (alphaLR - alphaT) * ybar
    })
)

gap_income = sp.simplify(
    gap_expanded.subs({
        ybar: (1 - mu) * yl + mu * yh
    })
)

print("\nGap expanded in primitives:")
sp.pprint(sp.factor(gap_expanded))

print("\nGap with income decomposition:")
sp.pprint(sp.factor(gap_income))

# =========================================================
# 11. Comparative statics
# =========================================================

print("\nComparative statics of traded-good prices:")
comparative_statics = {
    "dpT0_dbeta": sp.simplify(sp.diff(pT0_sol, beta)),
    "dpTR_dbeta": sp.simplify(sp.diff(pTR_sol, beta)),
    "dgap_dbeta": sp.simplify(sp.diff(gap, beta)),

    "dpT0_dK0": sp.simplify(sp.diff(pT0_sol, K0)),
    "dpT0_dKR": sp.simplify(sp.diff(pT0_sol, KR)),
    "dpTR_dK0": sp.simplify(sp.diff(pTR_sol, K0)),
    "dpTR_dKR": sp.simplify(sp.diff(pTR_sol, KR)),
    "dgap_dK0": sp.simplify(sp.diff(gap, K0)),
    "dgap_dKR": sp.simplify(sp.diff(gap, KR)),

    "dpT0_dcT0": sp.simplify(sp.diff(pT0_sol, cT0)),
    "dpT0_dcTR": sp.simplify(sp.diff(pT0_sol, cTR)),
    "dpTR_dcT0": sp.simplify(sp.diff(pTR_sol, cT0)),
    "dpTR_dcTR": sp.simplify(sp.diff(pTR_sol, cTR)),
    "dpT0_dcL": sp.simplify(sp.diff(pT0_sol, cL)),
    "dpTR_dcL": sp.simplify(sp.diff(pTR_sol, cL)),
    "dgap_dcT0": sp.simplify(sp.diff(gap, cT0)),
    "dgap_dcTR": sp.simplify(sp.diff(gap, cTR)),
    "dgap_dcL": sp.simplify(sp.diff(gap, cL)),
}

for name, expr in comparative_statics.items():
    print(f"\n{name} =")
    sp.pprint(sp.factor(expr))

print("\nComparative statics wrt t and R:")
print("dpT0_dt =")
sp.pprint(sp.simplify(sp.diff(pT0_sol, t)))
print("\ndpTR_dt =")
sp.pprint(sp.simplify(sp.diff(pTR_sol, t)))
print("\ndpT0_dR =")
sp.pprint(sp.simplify(sp.diff(pT0_sol, R)))
print("\ndpTR_dR =")
sp.pprint(sp.simplify(sp.diff(pTR_sol, R)))
print("\ndgap_dt =")
sp.pprint(sp.simplify(sp.diff(gap, t)))
print("\ndgap_dR =")
sp.pprint(sp.simplify(sp.diff(gap, R)))

gap_income_full = sp.simplify(
    gap.subs({
        K0: theta0 + (alphaL0 - alphaT) * ((1 - mu) * yl + mu * yh),
        KR: thetaR + (alphaLR - alphaT) * ((1 - mu) * yl + mu * yh)
    })
)

print("\nComparative statics wrt income composition:")
print("dgap_dmu =")
sp.pprint(sp.simplify(sp.diff(gap_income_full, mu)))
print("\ndgap_dyl =")
sp.pprint(sp.simplify(sp.diff(gap_income_full, yl)))
print("\ndgap_dyh =")
sp.pprint(sp.simplify(sp.diff(gap_income_full, yh)))

print("\ndgap_dlambda =")
sp.pprint(sp.simplify(sp.diff(gap, lam)))

# =========================================================
# 12. Helpers to export nicer LaTeX
# =========================================================

def latex_split_equation(expr, lhs, label=None, terms_per_line=2):
    """
    Build a LaTeX equation environment using split.
    expr must be expanded.
    lhs is a string, e.g. 'p_{T0}'.
    """
    num, den = sp.fraction(sp.together(sp.expand(expr)))
    terms = sp.Add.make_args(sp.expand(num))

    chunks = []
    for i in range(0, len(terms), terms_per_line):
        chunk = terms[i:i+terms_per_line]
        chunk_str = " + ".join(sp.latex(term) for term in chunk)
        chunk_str = chunk_str.replace("+ -", "- ")
        chunks.append(chunk_str)

    body = []
    body.append(r"\begin{equation}")
    body.append(r"\begin{split}")
    body.append(f"{lhs}")
    body.append(r"=")
    body.append(rf"\frac{{1}}{{{sp.latex(den)}}}")
    body.append(r"\Big[")
    for j, ch in enumerate(chunks):
        if j == 0:
            body.append("&" + ch + r" \\")
        elif j < len(chunks) - 1:
            body.append("&" + ch + r" \\")
        else:
            body.append("&" + ch)
    body.append(r"\Big].")
    body.append(r"\end{split}")
    if label is not None:
        body.append(rf"\label{{{label}}}")
    body.append(r"\end{equation}")
    return "\n".join(body)


def latex_inline(expr):
    return sp.latex(sp.simplify(expr))


# =========================================================
# 13. Print LaTeX blocks ready for Overleaf
# =========================================================

print("\n" + "="*70)
print("LATEX BLOCKS READY TO PASTE")
print("="*70)

print("\n% --- pT0 ---")
print(latex_split_equation(sp.expand(pT0_sol), r"p_{T0}", label="eq:pT0_eq", terms_per_line=2))

print("\n% --- pTR ---")
print(latex_split_equation(sp.expand(pTR_sol), r"p_{TR}", label="eq:pTR_eq", terms_per_line=2))

print("\n% --- pL0 ---")
print(latex_split_equation(sp.expand(pL0_sol), r"p_{L_0}", label="eq:pL0_eq", terms_per_line=2))

print("\n% --- pLR ---")
print(latex_split_equation(sp.expand(pLR_sol), r"p_{L_R}", label="eq:pLR_eq", terms_per_line=2))

print("\n% --- traded gap ---")
print("p_{T0}-p_{TR} =")
print(latex_inline(gap))

# =========================================================
# 14. Save LaTeX blocks to file
# =========================================================

latex_output = []
latex_output.append("% Auto-generated by Sympy\n")
latex_output.append(latex_split_equation(sp.expand(pT0_sol), r"p_{T0}", label="eq:pT0_eq", terms_per_line=2))
latex_output.append("\n")
latex_output.append(latex_split_equation(sp.expand(pTR_sol), r"p_{TR}", label="eq:pTR_eq", terms_per_line=2))
latex_output.append("\n")
latex_output.append(latex_split_equation(sp.expand(pL0_sol), r"p_{L_0}", label="eq:pL0_eq", terms_per_line=2))
latex_output.append("\n")
latex_output.append(latex_split_equation(sp.expand(pLR_sol), r"p_{L_R}", label="eq:pLR_eq", terms_per_line=2))
latex_output.append("\n")
latex_output.append("% Traded-good price gap:\n")
latex_output.append(r"\begin{equation}")
latex_output.append(rf"p_{{T0}}-p_{{TR}} = {sp.latex(sp.simplify(gap))}.")
latex_output.append(r"\label{eq:traded_gap}")
latex_output.append(r"\end{equation}")

with open("model_equations_generated.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex_output))

print("\nSaved LaTeX equations to model_equations_generated.tex")
