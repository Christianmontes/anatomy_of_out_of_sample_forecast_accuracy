# Referee Figures: Three DGPs

This figure set is generated from the final canonical CSV outputs under
`Simulation/outputs/sim1_master_out/`. It does not re-run simulations.

## Folder Structure

Each DGP folder contains:

- `combined/`: multi-page PDFs with all selected models.
- `by_model/<model>/`: one-page PDFs for a single model.
- `referee_model_summary.csv`: one row per model for that DGP.

Top-level files:

- `referee_model_summary_all_dgps.csv`: combined summary across DGPs.
- `manifest.json`: source folders and model list used for this figure set.

## DGPs

- `01_dgp1_structural_break__break_b08_expanding`: Headline structural-break DGP.
  Source: `revision\simulations\outputs\sim1_master_out\break_b08_expanding`
- `02_dgp2_persistent_noise__persistent_rho099_rolling20`: Persistent-noise / spurious-predictor DGP.
  Source: `revision\simulations\outputs\sim1_master_out\persistent_rho099_rolling20`
- `03_dgp3_proxy_break__proxy_break_reverse_noisyx1_expanding`: Proxy-break DGP: stale proxy reverses after the break.
  Source: `revision\simulations\outputs\sim1_master_out\proxy_break_reverse_noisyx1_expanding`

## Models

`ols`, `enet`, `rf`
