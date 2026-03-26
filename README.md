# Bundling-Crop-Insurance-with-Innovative-Production-Technologies
Monte Carlo simulation evaluating bundled weather-index insurance, input credit, and drought-tolerant seed scenarios for smallholder farmers in Guatemala's Dry Corridor. MSc coursework, Imperial Business School 2026.


# Guatemala Dry Corridor — Bundled Insurance & Input Finance Model

**Course:** Risk Management and Climate Change  
**Programme:** MSc Climate Change, Management and Finance  
**Institution:** Imperial Business School  
**Group:** O — Brunel, Li, Gigler, Pfister, Louali, Pearce, Tolo  
**Version:** v8 (final submission)

---

## What this model does

This is an original Python simulation built to evaluate product design 
choices for a bundled financial instrument targeting smallholder maize 
farmers in Guatemala's Dry Corridor. The bundle combines:

- Weather-index crop insurance (parametric, rainfall-triggered)
- An interlinked input credit facility
- Drought-tolerant (DT) maize seeds

The model tests **10 design scenarios** across **6 output dimensions** 
simultaneously, allowing direct quantitative comparison of trade-offs 
that the empirical literature can only inform indirectly.

It supports Section 6 (Policy Implications and Design Recommendations) 
of the group report.

---

## The 10 scenarios

| ID  | Description |
|-----|-------------|
| S1  | ERA5 index - current MAGA baseline |
| S2  | CHIRPS index |
| S3  | Composite index (NDVI + SPI) |
| S4  | ERA5 + Basis Risk Fund (BRF) |
| S5  | CHIRPS + conditional agronomic audits |
| S6  | CHIRPS + 50% premium subsidy |
| S7  | CHIRPS + full premium subsidy |
| S8  | CHIRPS + conventional seeds (no DT) |
| S9  | High-stress scenario (drought probability = 50%) |
| S10 | CHIRPS + audits + 50% subsidy - **recommended design** |

---

## The 6 output dimensions

1. **Basis risk** — false-negative rate (FN) and false-positive rate (FP)
2. **Farmer income gain** — net USD/ha/yr relative to uninsured baseline
3. **Tail risk** — Value-at-Risk (VaR 5%) and Conditional VaR (CVaR 5%)
4. **Insurer viability** — loss ratio at fair pricing
5. **Credit enhancement** — interest rate reduction and saving (informational)
6. **DT seed adoption** — trajectory over 5 seasons

---

## Model architecture

The model has two engines that cross-validate each other.

**Analytical engine** (`analytical_kpis`)  
Computes exact expected income using a bivariate normal probability 
model. Deterministic and the primary source for income gain, FN/FP 
rates, NGO cost, and loss ratio figures.

**Monte Carlo engine** (`monte_carlo`)  
Simulates N = 5,000 farmers over T = 200 seasons using Cholesky 
decomposition. Provides VaR 5%, CVaR 5%, and default probabilities. 
Verified stable at T = 200 (seed-to-seed VaR spread < $18).

---

## Key parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Drought probability | 0.30 | CHIRPS / World Bank DRIFCA 2024 |
| Severe El Niño frequency | 0.08 | ENSO ONI index |
| Base loan rate | 22% | BANRURAL 2022 |
| Premium loading factor | 35% | Commercial standard |
| DT input cost | $180/ha | WFP / CIMMYT Dry Corridor |
| Conventional input cost | $80/ha | WFP / CIMMYT Dry Corridor |
| Farm-scale ρ discount | −15pp | Biffis & Chavez (2017) |
| No-shock disadoption rate | 0.18 | Boucher et al. (2024) |
| Monte Carlo farmers (N) | 5,000 | — |
| Monte Carlo seasons (T) | 200 | Stability threshold |

---

## Literature embedded in the model

Four papers are directly parameterised in the code:

- **Carter, Cheng & Sarris (2016, JDE)** — portfolio equilibrium 
  multiplier (1.39×); meso-level interlinkage flattens lender loan 
  supply curve
- **Mobarak & Rosenzweig (2013, AER P&P)** — trust recovery parameter 
  (trust_rec = 0.10); informal network offset for peer-level diffusion
- **Biffis & Chavez (2017, Risk Analysis)** — farm-scale ρ discount 
  (−15pp); minimum viable ρ concept
- **Biffis et al. (2022, Geneva Risk)** — no credit double-counting; 
  affordability channel for subsidies

---

## Known limitations

1. **Single representative farmer** — no wealth heterogeneity or 
   variation in risk aversion. The model cannot reproduce distributional 
   adoption dynamics across farm sizes.

2. **Adoption parameters transferred from other countries** — 
   disadoption rates are calibrated on Tanzania, Ethiopia, and India 
   data. No Guatemala-specific calibration exists. Year-5 adoption 
   figures (30–42%) are illustrative projections, not forecasts.

3. **ρ values derived from Mozambique** — the correlation coefficient 
   ρ = 0.78 for CHIRPS is a literature transfer from Biffis & Chavez 
   (2017), not an empirically estimated value for the Dry Corridor.

4. **No climate non-stationarity** — drought probability is fixed at 
   0.30. S9 is a stress test, not a trajectory. A structural shift to 
   pd = 0.50 raises lifetime default probability to 100%.

5. **No peril basis risk** — the model only captures weather-index 
   spatial mismatch. Losses from fall armyworm, Roya, or localised 
   flooding are not modelled. True false-negative rates are higher than 
   reported.

6. **Maize only** — yield parameters, drought probability, and input 
   costs are maize-specific. Extension to coffee or sugarcane requires 
   full re-parameterisation.

7. **No moral hazard, no spatial heterogeneity, no price risk.**

---

## How to run
```bash
# Install dependencies
pip install numpy pandas matplotlib scipy

# Run the model (generates results table, recommendation text, 
# verification checks, and 5 figures)
python guatemala_final_v8.py
```

Output:
- Printed results table (all 10 scenarios × 6 dimensions)
- Section 6 recommendation text
- Verification checks (all should pass)
- `fig1_scenarios.png` — 6-panel scenario dashboard
- `fig2_distributions.png` — Monte Carlo income distributions
- `fig3_sensitivity.png` — sensitivity tornado chart
- `fig4_tradeoff.png` — NGO cost vs income gain scatter
- `fig5_adoption.png` — adoption trajectories over 5 seasons

---

## Dependencies
```
numpy
pandas
matplotlib
scipy
```

Python 3.8+ required. No proprietary packages.

---

## AI disclaimer

Consistent with the group report's AI disclaimer, GenAI tools (ChatGPT, 
Gemini, Perplexity) were used during development to clarify technical 
concepts and cross-check implementations. All model logic, parameter 
choices, and outputs were reviewed and validated by the group.
