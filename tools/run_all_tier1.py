#!/usr/bin/env python3
"""Regenerate every table and figure of the paper and the Internet Appendix from the saved artifacts.

Run from the package root, inside the pinned environment (see README, "Setup"):

    poetry run python tools/run_all_tier1.py                # Tier 0 + Tier 1 (minutes)
    poetry run python tools/run_all_tier1.py --tier 0       # only verify the published numbers
    poetry run python tools/run_all_tier1.py --tier S --smoke   # quick end-to-end run of the simulations
    poetry run python tools/run_all_tier1.py --tier S       # full simulation runs (hours, see README)
    poetry run python tools/run_all_tier1.py --tier 1.5     # MAS Monte Carlo recompute (about 1 hour)
    poetry run python tools/run_all_tier1.py --audit        # also log every file the run opens

Everything is written under outputs/ (tables/, Figures/, runtimes.txt, summary.json, file_usage.txt).
Figures are written under the exact relative names the LaTeX sources include, so
outputs/Figures can be dropped next to Anatomy.tex / AnatomyInternetAppendix.tex as is.
The tracked artifacts under Results/ are never modified by Tiers 0 and 1: the plotting
functions' savefig calls are redirected into outputs/ and the MAS table is assembled from
the cached Monte Carlo results (iml_rev.rankdiffs2_w) without rewriting the MAS workbooks.
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import warnings

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(PKG, "Code", "V002")
RES = os.path.join(PKG, "Results", "Updated CPI 3")
SIM = os.path.join(PKG, "revision", "simulations")
MASREF = os.path.join(PKG, "revision", "mas_referee_response")

# ---------------------------------------------------------------------------------------------
# Exhibit map: tex include path -> (producer, raw file name written by the producer)
# ---------------------------------------------------------------------------------------------
MODELS_TEX = {  # iml_rev model key -> tex row label
    "pca": "Principal component regression", "enet": "Elastic net", "rf_cv": "Random forest",
    "xgb_cv": "XGBoost", "nn_comb": "Neural network", "lin_comb": "Ensemble-linear",
    "nonlin_comb": "Ensemble-nonlinear", "all_comb": "Ensemble-all",
}
PERF_KEYS = {"pca": "pca_y_hat", "enet": "enet_y_hat", "rf_cv": "rf_cv_y_hat", "xgb_cv": "xgb_cv_y_hat",
             "nn_comb": "nn_comb", "lin_comb": "lin_comb", "nonlin_comb": "nonlin_comb", "all_comb": "all_comb"}
HS = [1, 3, 6, 12]
PBSV_MODELS = ["nonlin-comb", "all-comb", "enet", "lin-comb", "nn-comb", "pca", "rf-cv", "xgb-cv"]

FIGURES = {}  # tex path -> (step name, source path relative to a root, root)
for m in PBSV_MODELS:
    FIGURES["Figures/PBSV-vs-TSVI/oospbsv-vs-isvi-%s-rmse.pdf" % m] = (
        "plot_pbsv_vs_isvi", "Figures/PBSV-iShapley/oospbsv-vs-isvi-%s-rmse.pdf" % m, "raw")
FIGURES["Figures/MAS/quadrant_plot_all_mas2_w.pdf"] = ("quadrant_plot_w", "Figures/MAS/quadrant_plot_all_mas2_w.pdf", "raw")
for m in PBSV_MODELS:
    FIGURES["Figures/cumsse-%s-vs-base-upd.pdf" % m] = ("plot_cumsse", "Figures/CUMSSE/cumsse-%s-vs-base_upd.pdf" % m, "raw")
for i, dgp in enumerate(["friedman1", "polynomial", "threshold"], start=1):
    for w in ["expanding", "rolling"]:
        FIGURES["Figures/simulation/convergence_dgp%d_%s.png" % (i, w)] = (
            "simulations", "outputs/convergence_bands/convergence_bands_overlay_%s_%s.png" % (dgp, w), "sim")
SIM1 = {"break": "01_dgp1_structural_break__break_b08_expanding",
        "persistent": "02_dgp2_persistent_noise__persistent_rho099_rolling20",
        "proxy": "03_dgp3_proxy_break__proxy_break_reverse_noisyx1_expanding"}
for k, d in SIM1.items():
    for m in ["ols", "enet", "rf"]:
        FIGURES["Figures/simulation/sim1-importance-vs-gpbsv-avg-%s-%s.pdf" % (k, m)] = (
            "simulations", "outputs/sim1_master_out/referee_outputs/three_dgp_figures/%s/by_model/%s/fig_importance_vs_gpbsv_avg_%s.pdf" % (d, m, m), "sim")

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
class Run:
    def __init__(self, out):
        self.out = out
        self.raw = os.path.join(out, "_raw")
        self.results = []
        self.saved = []      # every file written through the savefig redirect

    def step(self, name, fn):
        t0 = time.time()
        try:
            msg = fn() or ""
            status = "PASS"
        except Exception as e:  # keep going, report at the end
            msg = "%s: %s" % (type(e).__name__, e)
            status = "FAIL"
        dt = time.time() - t0
        self.results.append({"step": name, "status": status, "seconds": round(dt, 1), "note": str(msg)})
        print("[%s] %-28s %7.1fs  %s" % (status, name, dt, msg), flush=True)


def stars(p):
    return "***" if p <= 0.01 else "**" if p <= 0.05 else "*" if p <= 0.10 else ""


def write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_csv(path, lineterminator="\n")
    except TypeError:
        df.to_csv(path, line_terminator="\n")


def tex_cell(v, s):
    """r@{.}l split cell: '$ 0$&$56^{***}$'."""
    txt = "%.2f" % v
    a, b = txt.split(".")
    return "$%s$&$%s^{%s}$" % (a, b, s) if s else "$%s$&$%s$" % (a, b)


def write_tex_table(path, header, rows, colspec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["\\begin{tabular}{%s}" % colspec, "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# Tier 0 / Tier 1 steps (run with cwd = Code/V002, the convention of the original code)
# ---------------------------------------------------------------------------------------------
def make_steps(run, args):
    import numpy as np
    import pandas as pd
    import pickle

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PYTHONBREAKPOINT"] = "0"          # iml_rev.normed_rank_diffs_tlb2 ends in breakpoint()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # Arial may be absent
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    sys.path.insert(0, CODE)
    os.chdir(CODE)
    import iml_rev  # noqa: E402  (the authors' module; imports the anatomy package)

    # redirect every plt.savefig(...) of the original code into outputs/_raw/<Results-relative path>
    _savefig = plt.savefig

    def savefig_redirect(fname, *a, **k):
        rel = os.path.normpath(str(fname)).replace("\\", "/")
        marker = "Results/Updated CPI 3/"
        rel = rel[rel.index(marker) + len(marker):] if marker in rel else os.path.basename(rel)
        dest = os.path.join(run.raw, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        run.saved.append(rel)
        return _savefig(dest, *a, **k)

    plt.savefig = savefig_redirect
    tables = os.path.join(run.out, "tables")

    # ---- Tier 0 -----------------------------------------------------------------------------
    def tier0_numbers():
        perf = pd.read_pickle(os.path.join(RES, "perf_raw.pickle"))
        w = pd.read_excel(os.path.join(RES, "MAS", "wmas_a0.67.xlsx"), index_col=0, engine="openpyxl")
        lines = ["Published numbers read from the saved artifacts (Tier 0)", ""]
        lines.append("RMSE ratios vs the AR benchmark (perf_raw.pickle):")
        for k, pk in PERF_KEYS.items():
            lines.append("  %-30s " % MODELS_TEX[k] + "  ".join("h=%-2d %.3f" % (h, perf.loc[(h, pk), "rmse_ratio"]) for h in HS))
        lines.append("")
        lines.append("MAS (alpha = 2/3), MAS/wmas_a0.67.xlsx:")
        for k in PERF_KEYS:
            lines.append("  %-30s " % MODELS_TEX[k] + "  ".join("h=%-2d %.3f" % (h, w.loc[k, h]) for h in HS))
        os.makedirs(tables, exist_ok=True)
        with open(os.path.join(run.out, "tier0_published_numbers.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        r12 = {k: perf.loc[(12, pk), "rmse_ratio"] for k, pk in PERF_KEYS.items()}
        assert abs(r12["rf_cv"] - 0.818) < 5e-4 and abs(r12["nonlin_comb"] - 0.822) < 5e-4, "h=12 RMSE ratios changed"
        return "RF h=12 ratio %.3f, ensemble-nonlinear %.3f, MAS RF h=12 %.2f" % (r12["rf_cv"], r12["nonlin_comb"], w.loc["rf_cv", 12])

    # ---- Table 1 (paper): MAS with significance stars ----------------------------------------
    def table1_mas():
        vals, pvals = {}, {}
        for h in HS:  # cached 1M-draw Monte Carlo results (override=False); Tier 1.5 recomputes them
            d = iml_rev.rankdiffs2_w(h=h, a=2 / 3, n_mc_sims=1000000, override=False)
            vals[h] = d["ishapley_vs_pbsv_mas_rnd_norm"]
            pvals[h] = d["ishapley_vs_pbsv_mas_rnd_norm_p"]
        keys = list(MODELS_TEX)
        V = pd.concat([vals[h].rename(h) for h in HS], axis=1).loc[keys]
        P = pd.concat([pvals[h].rename(h) for h in HS], axis=1).loc[keys]
        V["avg"] = V[HS].mean(axis=1)
        # consistency with the shipped workbooks (what the paper was typeset from)
        w = pd.read_excel(os.path.join(RES, "MAS", "wmas_a0.67.xlsx"), index_col=0, engine="openpyxl")
        wp = pd.read_excel(os.path.join(RES, "MAS", "wmas_pval_a0.67.xlsx"), index_col=0, engine="openpyxl")
        assert np.allclose(V.values, w.loc[keys, list(V.columns)].values, atol=1e-12), "MAS values differ from wmas_a0.67.xlsx"
        assert np.allclose(P.values, wp.loc[keys, HS].values, atol=1e-12), "MAS p-values differ from wmas_pval_a0.67.xlsx"
        out = pd.DataFrame({"forecast": [MODELS_TEX[k] for k in keys]}, index=keys)
        for h in HS:
            out["mas_h%d" % h] = V[h].round(4)
            out["pval_h%d" % h] = P[h].round(4)
            out["stars_h%d" % h] = [stars(p) for p in P[h]]
        write_csv(out, os.path.join(tables, "table1_mas.csv"))
        rows = ["%-30s & " % MODELS_TEX[k] + " & ".join(tex_cell(V.loc[k, h], stars(P.loc[k, h])) for h in HS) for k in keys]
        write_tex_table(os.path.join(tables, "table1_mas.tex"),
                        "Forecast & \\multicolumn{2}{c}{$h=1$} & \\multicolumn{2}{c}{$h=3$} & \\multicolumn{2}{c}{$h=6$} & \\multicolumn{2}{c}{$h=12$}",
                        rows, "lr@{.}lr@{.}lr@{.}lr@{.}l")
        return "32 cells, values and p-values identical to the shipped MAS workbooks"

    # ---- Table A.2 (IA, tab:rmse): RMSE ratios with DM stars ---------------------------------
    def tableA2_rmse():
        perf = pd.read_pickle(os.path.join(RES, "perf_raw.pickle"))
        rows_csv, rows_tex = [], []
        ar = ["Autoregressive benchmark RMSE"] + ["%.2f%%" % (100 * perf.loc[(h, "ar_y_hat"), "rmse"]) for h in HS]
        rows_csv.append(ar)
        rows_tex.append("%-30s & " % ar[0] + " & ".join("$%s$&$%s\\%%$" % tuple(("%.2f" % (100 * perf.loc[(h, "ar_y_hat"), "rmse"])).split(".")) for h in HS))
        for k, pk in PERF_KEYS.items():
            r = [perf.loc[(h, pk), "rmse_ratio"] for h in HS]
            p = [perf.loc[(h, pk), "dm_pval"] for h in HS]
            rows_csv.append([MODELS_TEX[k]] + ["%.4f%s" % (v, stars(q)) for v, q in zip(r, p)])
            rows_tex.append("%-30s & " % MODELS_TEX[k] + " & ".join(tex_cell(v, stars(q)) for v, q in zip(r, p)))
        df = pd.DataFrame(rows_csv, columns=["forecast"] + ["h=%d" % h for h in HS]).set_index("forecast")
        write_csv(df, os.path.join(tables, "tableA2_rmse.csv"))
        write_tex_table(os.path.join(tables, "tableA2_rmse.tex"),
                        "Forecast & \\multicolumn{2}{c}{$h=1$} & \\multicolumn{2}{c}{$h=3$} & \\multicolumn{2}{c}{$h=6$} & \\multicolumn{2}{c}{$h=12$}",
                        rows_tex, "lr@{.}lr@{.}lr@{.}lr@{.}l")
        return "AR RMSE row + 32 ratio cells with Diebold-Mariano stars"

    # ---- Table A.3 (IA, tab:inGpbsv): MAS from TS-Shapley-VI vs in-sample GPBSV ---------------
    def tableA3_insample_gpbsv():
        raw = pd.read_csv(os.path.join(MASREF, "outputs", "mas_is_oos_gpbsv_v5_a0.67_mc1000000.csv"))
        pv = pd.read_csv(os.path.join(MASREF, "outputs", "mas_is_oos_gpbsv_v5_pvalues_a0.67_mc1000000.csv"))
        TS = "Current: IS Shapley VI vs OOS GPBSV"
        IS = "IS GPBSV robustness: abs(IS GPBSV) vs saved OOS GPBSV"
        NAME = [("PCR", "PCA"), ("Elastic net", "ENet"), ("Random forest", "RF"), ("XGBoost", "XGBoost"),
                ("Neural network", "Neural net"), ("Ensemble-linear", "Linear combination"),
                ("Ensemble-nonlinear", "Nonlinear combination"), ("Ensemble-all", "All models combination")]
        rec = []
        for tex_name, lab in NAME:
            row = {"forecast": tex_name}
            for h in HS:
                for block, tag in [(TS, "tsvi"), (IS, "isgpbsv")]:
                    v = float(raw[(raw.comparison_label == block) & (raw.model_label == lab)][str(h)].iloc[0])
                    p = float(pv[(pv.comparison_label == block) & (pv.model_label == lab)][str(h)].iloc[0])
                    row["mas_%s_h%d" % (tag, h)] = round(v, 4)
                    row["stars_%s_h%d" % (tag, h)] = stars(p)
            rec.append(row)
        df = pd.DataFrame(rec).set_index("forecast")
        avg = {c: df[c].mean() for c in df.columns if c.startswith("mas_")}
        df.loc["Average"] = pd.Series({**{c: round(v, 4) for c, v in avg.items()}, **{c: "" for c in df.columns if c.startswith("stars_")}})
        write_csv(df, os.path.join(tables, "tableA3_insample_gpbsv_mas.csv"))
        shutil.copy2(os.path.join(MASREF, "outputs", "mas_is_oos_gpbsv_v5_a0.67_mc1000000.tex"),
                     os.path.join(tables, "tableA3_insample_gpbsv_mas_shipped.tex"))
        return "64 cells + averages from the shipped in-sample GPBSV MAS results"

    # ---- Table A.1 (IA, tab:fredmd): predictor list (documentation) --------------------------
    def tableA1_predictors():
        fm = pickle.load(open(os.path.join(CODE, "_cache", "get_fred_md.bin"), "rb"))
        soc = pickle.load(open(os.path.join(CODE, "_cache", "get_soc.bin"), "rb"))
        cols = [c for c in fm.columns] + list(soc.columns)
        df = pd.DataFrame({"predictor": cols, "source": ["FRED-MD"] * len(fm.columns) + ["Michigan SOC"] * len(soc.columns)})
        df.index.name = "n"
        write_csv(df, os.path.join(tables, "tableA1_predictor_list.csv"))
        used = sorted(set(pd.read_pickle(os.path.join(RES, "shapleys_h1_upd.bin"))[0][0].columns) - {"base_contribution"})
        with open(os.path.join(tables, "tableA1_predictors_in_models.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(used) + "\n")
        return "%d FRED-MD + %d SOC series in the cache; %d grouped predictors enter the Shapley decompositions" % (len(fm.columns), len(soc.columns), len(used))

    # ---- Figures from the saved Shapley/GPBSV bundles ----------------------------------------
    def fig_pbsv_vs_isvi():
        iml_rev.plot_pbsv_vs_isvi()
        plt.close("all")
        n = sum(1 for s in run.saved if s.startswith("Figures/PBSV-iShapley/"))
        return "%d PDFs written (8 full-horizon + zoom variants)" % n

    def fig_quadrant():
        iml_rev.quadrant_plot_w()
        plt.close("all")
        return "MAS quadrant plots written"

    def fig_cumsse():
        missing = [h for h in [1, 6, 12] if not os.path.exists(os.path.join(RES, "y_base_nn_h%d.csv" % h))
                   and not os.path.exists(os.path.join(RES, "ishapley_h%d_upd.bin" % h))]
        if missing:
            raise FileNotFoundError("y_base_nn_h%s.csv missing (and no ishapley bins): run tools/extract_y_base.py on the estimation box" % missing)
        iml_rev.plot_cumsse()
        plt.close("all")
        n = sum(1 for s in run.saved if s.startswith("Figures/CUMSSE/"))
        return "%d CDSE PDFs written" % n

    # ---- collect everything under the tex names ---------------------------------------------
    def collect():
        missing, done = [], 0
        for tex, (producer, src_rel, root) in FIGURES.items():
            src = os.path.join(run.raw, src_rel) if root == "raw" else os.path.join(SIM, src_rel)
            dst = os.path.join(run.out, tex)
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                done += 1
            else:
                missing.append(tex)
        # keep the zoom / single variants produced by the original code
        extra = os.path.join(run.out, "Figures", "extra")
        used = {v[1] for v in FIGURES.values() if v[2] == "raw"}
        for rel in run.saved:
            if rel not in used:
                os.makedirs(extra, exist_ok=True)
                shutil.copy2(os.path.join(run.raw, rel), os.path.join(extra, os.path.basename(rel)))
        if missing:
            raise FileNotFoundError("%d of %d figures missing: %s" % (len(missing), len(FIGURES), missing))
        return "%d figure files under outputs/Figures with the LaTeX include names" % done

    steps = []
    if args.tier in ("0", "1", "all"):
        steps.append(("tier0_numbers", tier0_numbers))
    if args.tier in ("1", "all"):
        steps += [("table1_mas", table1_mas), ("tableA2_rmse", tableA2_rmse),
                  ("tableA3_insample_gpbsv", tableA3_insample_gpbsv), ("tableA1_predictors", tableA1_predictors),
                  ("fig_pbsv_vs_isvi", fig_pbsv_vs_isvi), ("fig_quadrant", fig_quadrant),
                  ("fig_cumsse", fig_cumsse), ("collect", collect)]
    return steps


# ---------------------------------------------------------------------------------------------
# Tier 1.5 and Tier S: external scripts (they write their own default output locations)
# ---------------------------------------------------------------------------------------------
def external_steps(run, args):
    py = sys.executable

    def sh(cmd, cwd):
        print("   $", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)

    def mas_recompute():  # independent 1M-draw recompute of Table 1 (seeded) + referee variants
        out = os.path.join(run.out, "mas_referee_variants")
        os.makedirs(out, exist_ok=True)
        sh([py, "mas_referee_variants.py", "--n-sims", "1000" if args.smoke else "1000000", "--output-dir", out,
            "--include-sensitivity", "--include-insample-gpbsv"], MASREF)
        return "outputs/mas_referee_variants (Current MAS block = Table 1)"

    def sims():
        # Non-destructive: everything goes to outputs/simulations/ and the committed reference outputs in
        # revision/simulations/outputs/ stay untouched. build_sim1_referee_three_dgp_figures.py reads the
        # scenario folders from the committed location (hard-coded in the script), so its figure set here is
        # rebuilt from the reference scenario outputs; to regenerate the figures from a fresh full run, run the
        # scripts in place as the authors did (README, Section 8).
        so = os.path.join(run.out, "simulations")
        os.makedirs(so, exist_ok=True)
        s = ["--smoke"] if args.smoke else []
        sh([py, os.path.join("scripts", "revision", "run_gp_bsv_convergence_bands.py"), "--out-dir", os.path.join(so, "convergence_bands")] + s, SIM)
        sh([py, os.path.join("scripts", "revision", "run_sim1_trap_dgps.py"), "--output-root", os.path.join(so, "sim1_master_out")]
           + (["--n-seeds", "2"] if args.smoke else []), SIM)
        sh([py, os.path.join("scripts", "revision", "postprocess_sim1_paper.py"), "--output_root", os.path.join(so, "sim1_master_out")], SIM)
        sh([py, os.path.join("scripts", "revision", "build_sim1_referee_three_dgp_figures.py"), "--output-root",
            os.path.join(so, "three_dgp_figures")], SIM)
        return "simulation outputs written to outputs/simulations (reference outputs untouched)"

    steps = []
    if args.tier in ("1.5",):
        steps.append(("mas_recompute", mas_recompute))
    if args.tier in ("S",):
        steps.append(("simulations", sims))
    return steps


def install_audit(run):
    opened = set()
    pkg = os.path.normcase(os.path.abspath(PKG))

    def hook(event, a):
        if event == "open" and isinstance(a[0], str):
            p = os.path.normcase(os.path.abspath(a[0]))
            mode = a[1] or "r"
            if "r" in mode and not p.startswith(os.path.normcase(os.path.abspath(sys.prefix))):
                opened.add(os.path.relpath(p, pkg) if p.startswith(pkg) else "EXTERNAL " + p)
    sys.addaudithook(hook)
    return opened


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="all", choices=["0", "1", "all", "1.5", "S"])
    ap.add_argument("--out", default=os.path.join(PKG, "outputs"))
    ap.add_argument("--smoke", action="store_true", help="fast configurations for Tiers 1.5 and S")
    ap.add_argument("--audit", action="store_true", help="log every file opened for reading to outputs/file_usage.txt")
    args = ap.parse_args()

    run = Run(os.path.abspath(args.out))
    os.makedirs(run.out, exist_ok=True)
    opened = install_audit(run) if args.audit else None
    t0 = time.time()
    steps = external_steps(run, args) if args.tier in ("1.5", "S") else make_steps(run, args)
    for name, fn in steps:
        run.step(name, fn)
    total = time.time() - t0

    with open(os.path.join(run.out, "runtimes.txt"), "a", encoding="utf-8", newline="\n") as fh:
        fh.write("%s tier=%s smoke=%s python=%s total=%.1fs\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), args.tier, args.smoke, sys.version.split()[0], total))
        for r in run.results:
            fh.write("  %-28s %-4s %8.1fs  %s\n" % (r["step"], r["status"], r["seconds"], r["note"]))
    with open(os.path.join(run.out, "summary_%s.json" % args.tier.replace(".", "_")), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tier": args.tier, "smoke": args.smoke, "total_seconds": round(total, 1), "steps": run.results}, fh, indent=2)
    if opened is not None:
        with open(os.path.join(run.out, "file_usage.txt"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(sorted(opened)) + "\n")
        print("file usage log: %d files -> outputs/file_usage.txt" % len(opened))
    failed = [r["step"] for r in run.results if r["status"] == "FAIL"]
    print("\n%s: %d steps, %d failed, %.1fs" % ("FAILED" if failed else "ALL PASSED", len(run.results), len(failed), total))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
