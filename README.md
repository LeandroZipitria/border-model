# border-model

This repository contains a symbolic implementation of a border-pricing model with traded and local products on both sides of the border.

## Purpose

The repository is designed to derive, verify, and export the main equilibrium expressions of an extended border model. The framework builds on a Hotelling-style spatial competition setting and is intended to clarify how observed cross-border price gaps may reflect not only border frictions, but also local competitive asymmetries and cost differences.

In particular, the model allows for:

- one traded product sold in both markets,
- one local product in each market,
- heterogeneous substitutability between local and traded products across markets,
- two income types,
- market-specific marginal costs for the traded product,
- a common marginal cost for local products.

The main object of interest is the traded-good price gap across markets and its decomposition into:

1. a pure border component,
2. a local-competition component,
3. a traded-good cost component.

## Repository contents

- `model_sympy.py`: symbolic derivation of the model using Python and Sympy
- `model_equations_generated.tex`: LaTeX equations exported from the symbolic solution

## Model ingredients

The symbolic model includes the following primitives:

- `beta`: border cost
- `t`: transport cost
- `R`: market length
- `lam`: share of consumers attached to local products
- `theta0`, `thetaR`: substitutability parameters between local and traded products across markets
- `mu`, `yl`, `yh`: income distribution parameters
- `cT0`, `cTR`: marginal costs of the traded product in each market
- `cL`: common marginal cost of local products

The model also defines:

- `ybar = (1-mu)*yl + mu*yh`
- `K0 = theta0 + (alphaL0 - alphaT)*ybar`
- `KR = thetaR + (alphaLR - alphaT)*ybar`

## What the script does

The main script:

1. defines the primitives of the model,
2. derives marginal consumers,
3. constructs demand equations,
4. builds profit functions,
5. derives first-order conditions,
6. solves for reaction functions,
7. solves for equilibrium prices,
8. computes the traded-good price gap,
9. derives comparative statics,
10. exports LaTeX-ready equations.

## Requirements

This project requires Python and the `sympy` package.

Install Sympy with:

```bash
pip install sympy
```

## Running the model

Run:

```bash
python model_sympy.py
```

The script prints the symbolic objects in the terminal and writes:

- `model_equations_generated.tex`

which contains LaTeX-ready equilibrium equations.

## Interpretation

The model implies that the observed cross-border price gap in the traded good can reflect three distinct forces:

1. border frictions,
2. asymmetries in local competition,
3. asymmetries in the marginal cost of the traded product.

This is useful for empirical work because it clarifies that regressions omitting local competition or cost shifters may attribute to the border what is in fact driven by local product environments or relative cost differences.

## Scope

This repository is intended as a symbolic derivation and research support tool. It is not a calibrated quantitative model.

## Author

Leandro Zipitria
