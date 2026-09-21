# Simulation studies (Internet Appendix Sections A.4 and A.7)

Self-contained scripts: synthetic data, seeded generators, the installed `anatomy` package. Run from this folder
inside the pinned environment (or through `tools/run_all_tier1.py --tier S [--smoke]` from the package root).

| Script (`scripts/revision/`) | Exhibit | Output folder | Runtime |
|---|---|---|---|
| `run_gp_bsv_convergence_bands.py [--smoke] [--out-dir]` | IA Section A.4, Figures A.1 to A.6: GPBSV convergence in the number of Monte Carlo permutations, three DGPs (Friedman #1, polynomial, threshold), expanding and rolling windows, with replicate bands | `outputs/convergence_bands/` (`convergence_bands_overlay_<dgp>_<window>.png` are the published panels; per-model figures and the raw/summary CSVs are also written) | hours (`--smoke`: minutes) |
| `run_sim1_trap_dgps.py [--n-seeds N] [--output-root]` | IA Section A.7, Figures A.7 to A.9: in-sample importance vs GPBSV for the structural-break, persistent-noise and proxy-break DGPs, OLS / elastic net / random forest, 25 seeds | `outputs/sim1_master_out/<scenario>/` | up to a day |
| `postprocess_sim1_paper.py` | paper-facing tables and figures from the sim1 raw output | `outputs/sim1_master_out/` | minutes |
| `build_sim1_referee_three_dgp_figures.py` | the three-DGP figure set used in the appendix (`referee_outputs/three_dgp_figures/<dgp>/by_model/<model>/fig_importance_vs_gpbsv_avg_<model>.pdf`) | `outputs/sim1_master_out/referee_outputs/` | minutes |
| `postprocess_sim1_referee.py` | referee-facing summaries (MAS distributions, quadrant plots) | `outputs/sim1_master_out/*/referee_outputs/` | minutes |

Supplementary (part of the revision analysis, not behind a published exhibit): `run_sim2_ape_oob_comparison.py`
(GPBSV vs average partial effects vs out-of-bag importance; `outputs/sim2/`), `run_sim1_insample_gpbsv.py` and
`run_sim1_mas_is_vs_oos_gpbsv.py` (in-sample vs out-of-sample GPBSV in the simulations; `outputs/insample_gpbsv/`),
`simulation_2/Main.py` and `pbsv_convergence_with_bands.py` (earlier versions of the studies).

The committed `outputs/` are the reference results behind the appendix. Rerunning a script rewrites its output
folder in place; use `git diff` / `git checkout -- outputs` to compare with or restore the reference.
