# Notebooks

Two hands-on activities for *Intro to Probabilistic Programming*, both built on the
FOPPL evaluator from the lecture. Click a badge to open the notebook in Google Colab
and run it top to bottom. No setup is required.

## Activity 1: Complete the FOPPL evaluator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jburroni/IntroPPLs26/blob/main/notebooks/Jun-19/activity-1-complete-the-evaluator.ipynb)

A FOPPL program is parsed into nested Python lists. You are given the evaluator with
the `if` case left blank. Fill it in, then run the test: likelihood weighting on the
conjugate model `m ~ Normal(0, 1)` with `y = 2` should recover the posterior mean
`E[m | y] = 1`.

## Activity 2: Measure the collapse, then fix the prior

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jburroni/IntroPPLs26/blob/main/notebooks/Jun-19/activity-2-ess-and-proposals.ipynb)

Carry over the `if` line you wrote in Activity 1. Implement the effective sample size
(ESS), measure it for a prior that sits far from the data, then design a better prior
that raises it. A before/after histogram of the weights shows the collapse and the fix.

## Running locally

From the repository root:

```bash
uv run jupyter lab
```

Open either notebook under `lectures/notebooks/` and run the cells top to bottom. The
first cell locates the `minippl` package automatically, so the notebooks run from
wherever you open them.
