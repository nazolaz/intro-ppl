# 🏆 2026 World Cup — Bayesian goals model in PyMC

A teaching demo for **Introduction to Probabilistic Programming Languages** (UBA Exactas).

We build a Bayesian **skill + Poisson-goals** model for the 2026 FIFA World Cup. Each team
has a latent skill `s ~ Normal(μ, σ_s)`; a match between `a` and `b` produces goals

```
λ_a = λ0 · exp(β (s_a − s_b)),   λ_b = λ0 · exp(β (s_b − s_a))
y_a ~ Poisson(λ_a),              y_b ~ Poisson(λ_b)
```

so the **score-line** (not just win/draw/loss) is the data — a 7–1 moves beliefs far more
than a 1–0.

1. **Prior** — latent team *skills*, calibrated so that simulating the tournament reproduces
   the **bookmaker title odds**.
2. **Likelihood** — Poisson **goals** in the **matches played**.
3. **Posterior** — PyMC/NUTS gives the posterior over all 48 skills.
4. **Prediction** — a vectorized Monte-Carlo **tournament simulator** turns skills into
   **championship probabilities**.
5. **Visualization** — a split **violin plot** (`Set2`) of the **prior vs posterior
   distribution of winning**, plus calibration, skill-shift and probability charts (seaborn).


## Run it

```bash
uv sync                      # create the environment
uv run jupyter lab worldcup_trueskill.ipynb
```

Or just re-execute everything headless:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace worldcup_trueskill.ipynb
```

## Files

| file | what |
|------|------|
| `worldcup_trueskill.ipynb` | the notebook (this is the deliverable) |
| `build_notebook.py` | regenerates the notebook from source (`uv run python build_notebook.py`) |
| `pyproject.toml` / `uv.lock` | the `uv`-managed environment |

*Sources: 2026 final draw & fixtures (ESPN), title odds (Yahoo Sports / BetMGM).*
