# =========================================================
# Symbolic verification of the border model equilibrium
# Local products on both sides of the border
# =========================================================

import sympy as sp
from contextlib import redirect_stdout

# ---------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------

pT0, pTR, pL0, pLR = sp.symbols("pT0 pTR pL0 pLR", real=True)
beta, K0, KR, t, R, lam = sp.symbols("beta K0 KR t R lambda", real=True)
cL, cT0, cTR = sp.symbols("cL cT0 cTR", real=True)

tR = t * R

# ---------------------------------------------------------
# 2. Reaction functions
# ---------------------------------------------------------

eq1 = sp.Eq(pL0, (pTR + cL + tR + beta + K0) / 2)
eq2 = sp.Eq(pLR, (pT0 + cL + tR - beta + KR) / 2)
eq3 = sp.Eq(
    pT0,
    (
        (1 - 2 * lam) * pTR
        + lam * pLR
        + (1 - lam) * (tR + beta)
        - lam * KR
        + (1 - lam) * cT0
    ) / (2 * (1 - lam))
)
eq4 = sp.Eq(
    pTR,
    (
        (1 - 2 * lam) * pT0
        + lam * pL0
        + (1 - lam) * (tR - beta)
        - lam * K0
        + (1 - lam) * cTR
    ) / (2 * (1 - lam))
)

# ---------------------------------------------------------
# 3. Solve the linear system robustly
# ---------------------------------------------------------

eq_system = [
    eq1.lhs - eq1.rhs,
    eq2.lhs - eq2.rhs,
    eq3.lhs - eq3.rhs,
    eq4.lhs - eq4.rhs,
]

A, b = sp.linear_eq_to_matrix(eq_system, [pT0, pTR, pL0, pLR])
sol_vec = A.LUsolve(b)

sol = {
    pT0: sp.simplify(sol_vec[0]),
    pTR: sp.simplify(sol_vec[1]),
    pL0: sp.simplify(sol_vec[2]),
    pLR: sp.simplify(sol_vec[3]),
}

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def run_report():
    section("1. REACTION FUNCTIONS")
    for name, eq in [
        ("Reaction function for local good in market 0", eq1),
        ("Reaction function for local good in market R", eq2),
        ("Reaction function for traded good in market 0", eq3),
        ("Reaction function for traded good in market R", eq4),
    ]:
        print(name + ":")
        sp.pprint(eq)
        print()

    section("2. EQUILIBRIUM PRICES")
    for var in [pT0, pTR, pL0, pLR]:
        print(f"{var} =")
        sp.pprint(sp.factor(sp.together(sol[var])))
        print()

    section("3. CHECK THAT THE SOLUTION SATISFIES THE SYSTEM")
    residuals = {
        "eq1 residual": sp.simplify(eq1.lhs.subs(sol) - eq1.rhs.subs(sol)),
        "eq2 residual": sp.simplify(eq2.lhs.subs(sol) - eq2.rhs.subs(sol)),
        "eq3 residual": sp.simplify(eq3.lhs.subs(sol) - eq3.rhs.subs(sol)),
        "eq4 residual": sp.simplify(eq4.lhs.subs(sol) - eq4.rhs.subs(sol)),
    }
    for name, expr in residuals.items():
        print(name + ":")
        sp.pprint(expr)
        print()

    section("4. TRADED-GOOD PRICE GAP")
    diff = sp.simplify(sol[pT0] - sol[pTR])
    diff_closed = sp.simplify(
        sp.Rational(2, 3) * beta
        + lam * (K0 - KR) / (3 * (2 - 3 * lam))
        + 2 * (1 - lam) * (cT0 - cTR) / (3 * (2 - 3 * lam))
    )
    print("Derived expression for pT0 - pTR:")
    sp.pprint(sp.factor(diff))
    print("\nClosed-form expression:")
    sp.pprint(diff_closed)
    print("\nDifference between both expressions (should be 0):")
    sp.pprint(sp.simplify(diff - diff_closed))

    section("5. COMPARATIVE STATICS OF THE TRADED-GOOD GAP")
    derivs_gap = {
        "d(pT0-pTR)/d beta": sp.simplify(sp.diff(diff, beta)),
        "d(pT0-pTR)/d K0": sp.simplify(sp.diff(diff, K0)),
        "d(pT0-pTR)/d KR": sp.simplify(sp.diff(diff, KR)),
        "d(pT0-pTR)/d cT0": sp.simplify(sp.diff(diff, cT0)),
        "d(pT0-pTR)/d cTR": sp.simplify(sp.diff(diff, cTR)),
        "d(pT0-pTR)/d cL": sp.simplify(sp.diff(diff, cL)),
        "d(pT0-pTR)/d lambda": sp.simplify(sp.diff(diff, lam)),
    }

    expected_gap = {
        "d(pT0-pTR)/d beta": sp.Rational(2, 3),
        "d(pT0-pTR)/d K0": lam / (3 * (2 - 3 * lam)),
        "d(pT0-pTR)/d KR": -lam / (3 * (2 - 3 * lam)),
        "d(pT0-pTR)/d cT0": 2 * (1 - lam) / (3 * (2 - 3 * lam)),
        "d(pT0-pTR)/d cTR": -2 * (1 - lam) / (3 * (2 - 3 * lam)),
        "d(pT0-pTR)/d cL": 0,
        "d(pT0-pTR)/d lambda": 2 * (K0 - KR + cT0 - cTR) / (3 * (2 - 3 * lam) ** 2),
    }

    for name in derivs_gap:
        calc = sp.simplify(derivs_gap[name])
        exp = sp.simplify(expected_gap[name])
        print(name)
        print("Calculated:")
        sp.pprint(calc)
        print("Expected:")
        sp.pprint(exp)
        print("Match:", sp.simplify(calc - exp) == 0)
        print()

    section("6. COMPARATIVE STATICS OF PRICE LEVELS")
    level_checks = {
        "dpT0/dbeta": (
            sp.simplify(sp.diff(sol[pT0], beta)),
            sp.Rational(1, 3),
        ),
        "dpTR/dbeta": (
            sp.simplify(sp.diff(sol[pTR], beta)),
            -sp.Rational(1, 3),
        ),
        "dpT0/dK0": (
            sp.simplify(sp.diff(sol[pT0], K0)),
            -2 * lam * (1 - 2 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpT0/dKR": (
            sp.simplify(sp.diff(sol[pT0], KR)),
            -lam * (4 - 5 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpTR/dK0": (
            sp.simplify(sp.diff(sol[pTR], K0)),
            -lam * (4 - 5 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpTR/dKR": (
            sp.simplify(sp.diff(sol[pTR], KR)),
            -2 * lam * (1 - 2 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpT0/dcT0": (
            sp.simplify(sp.diff(sol[pT0], cT0)),
            (10 * lam**2 - 18 * lam + 8) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpT0/dcTR": (
            sp.simplify(sp.diff(sol[pT0], cTR)),
            (8 * lam**2 - 12 * lam + 4) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpTR/dcT0": (
            sp.simplify(sp.diff(sol[pTR], cT0)),
            (8 * lam**2 - 12 * lam + 4) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpTR/dcTR": (
            sp.simplify(sp.diff(sol[pTR], cTR)),
            (10 * lam**2 - 18 * lam + 8) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpT0/dcL": (
            sp.simplify(sp.diff(sol[pT0], cL)),
            (-9 * lam**2 + 6 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
        "dpTR/dcL": (
            sp.simplify(sp.diff(sol[pTR], cL)),
            (-9 * lam**2 + 6 * lam) / (3 * (4 - 8 * lam + 3 * lam**2)),
        ),
    }

    for name, (calc, exp) in level_checks.items():
        print(name)
        print("Calculated:")
        sp.pprint(calc)
        print("Expected:")
        sp.pprint(exp)
        print("Match:", sp.simplify(calc - exp) == 0)
        print()

    section("7. EFFECT OF t AND R")
    dpT0_dt = sp.simplify(sp.diff(sol[pT0], t))
    dpTR_dt = sp.simplify(sp.diff(sol[pTR], t))
    dpT0_dR = sp.simplify(sp.diff(sol[pT0], R))
    dpTR_dR = sp.simplify(sp.diff(sol[pTR], R))

    print("dpT0/dt =")
    sp.pprint(dpT0_dt)
    print("\ndpTR/dt =")
    sp.pprint(dpTR_dt)
    print("\ndpT0/dR =")
    sp.pprint(dpT0_dR)
    print("\ndpTR/dR =")
    sp.pprint(dpTR_dR)
    print("\nd(pT0-pTR)/dt (should be 0):")
    sp.pprint(sp.simplify(dpT0_dt - dpTR_dt))
    print("\nd(pT0-pTR)/dR (should be 0):")
    sp.pprint(sp.simplify(dpT0_dR - dpTR_dR))

if __name__ == "__main__":
    run_report()

    with open("verification_report.txt", "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_report()

    print("\nSaved full verification output to verification_report.txt")
