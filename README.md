# ENEM Performance: What Socioeconomic Data Reveals About Who Reaches University (2019 → 2023)

> An end-to-end data science study on Brazil's national university entrance exam — from
> exploratory analysis (2019 vs 2023) to predictive modeling of the weighted score (MPE).
> Focused less on *running models* and more on *what the numbers actually say* about
> educational inequality.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-3F4F75?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## TL;DR — the findings that matter

1. **The weighted score (MPE) rose ~31 points** between the 2019 and 2023 cohorts (540.8 → 572.1)
   — while **participation dropped ~45%** (1.13M → 628K valid candidates), echoing the
   post-pandemic decline in ENEM turnout.
2. **The strongest predictors of MPE are family income and race**, followed by parental
   education. Being in the lowest income bracket is by far the most influential feature; access to
   a computer at home contributes, but secondarily.
3. **Socioeconomic factors explain about a third of the variation** in a candidate's score
   (R² ≈ 0.33). The other two-thirds live in factors this data can't see — a finding in itself
   about the limits of socioeconomic determinism.

## The question

The ENEM is the main gateway to higher education in Brazil. Did the **profile and performance** of
candidates change between 2019 and 2023, and which socioeconomic characteristics best **predict**
the weighted score (MPE) for Business/Economics programs?

MPE (Business/Economics weighting):

```
MPE = 0.75 × (0.25·LC + 0.40·MT + 0.25·CH + 0.10·CN) + 0.25 × Essay
```

## What's inside

A single notebook (`enem_mpe_analise_predicao.ipynb`) covering the full pipeline:

1. **Cleaning** — load raw microdata (chunked read), keep comparable candidates, compute MPE (vectorized).
2. **Feature prep** — readable labels and ordering applied identically to both years.
3. **Exploratory analysis** — 2019 vs 2023 profiles, comparative charts, a socioeconomic dashboard.
4. **Predictive modeling** — Multiple Linear Regression (with backward p-value selection) vs a
   depth-constrained Random Forest, evaluated with R²/RMSE/MAE and k-fold cross-validation.
5. **Interpretation** — what the results mean, honestly.

## Stage 1 — Exploratory analysis (2019 vs 2023)

- Average MPE **rose ~31 points** (540.8 → 572.1).
- Valid candidates **fell ~45%** (1.13M → 628K) — a sharp participation drop; part of the average
  gain may reflect a more-prepared remaining pool.
- The **same socioeconomic determinants stayed relevant** in both years.

## Stage 2 — Predictive modeling

| Model | R² (test) | R² (train) | Read |
|---|---|---|---|
| **Multiple Linear Regression** | **32.7%** | 32.7% | Near-perfect generalization (train ≈ test) |
| Random Forest (constrained) | 32.1% | 33.0% | ~1-point train–test gap: minimal overfitting |

- Non-significant predictors removed via backward elimination on p-values (`P>|t| > 0.1`).
- Cross-validation confirmed stability: OLS 32.7% ± 0.30%, RF 32.1% ± 0.19%.
- The two models essentially **tie**, with linear regression slightly ahead — the relationship is
  **predominantly linear**, and constraining the forest's depth removed the overfitting it showed
  unconstrained.

**Which variables carry the signal:** by a wide margin, being in the **lowest income bracket**,
then **race** (white), then **parental education**. Computer access registers but is secondary.
The dominant drivers are **income and race** — a direct picture of the socioeconomic inequality
the exam reflects.

## Why ~0.33 R² is honest, not weak

Socioeconomic background shapes ENEM performance substantially, but doesn't determine it. The
unexplained two-thirds is where individual effort, school quality, and other unmeasured factors
live. A model predicting exam scores at 90% from income and race alone would be the suspicious one.

## Data

The raw ENEM microdata are large public files and are **not committed** to this repo. Download them
from the
[INEP open data portal](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem)
(`MICRODADOS_ENEM_2019.csv`, `MICRODADOS_ENEM_2023.csv`). Running the notebook produces a cleaned,
MPE-augmented base (`dados_ENEM_2023.csv`) — small enough to commit for reproducibility.

## Run it

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
jupyter notebook enem_mpe_analise_predicao.ipynb
```

## Limitations & next steps

- Socioeconomic/demographic features only; pedagogical and psychological factors out of scope.
- Try continuous features and regularized regression (Ridge, Lasso).
- Re-run across more ENEM editions to test whether the important variables drift over time.


