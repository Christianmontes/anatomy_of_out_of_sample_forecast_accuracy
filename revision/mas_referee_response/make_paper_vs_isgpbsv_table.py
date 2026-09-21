"""
Build a focused MAS comparison table for the horizons currently available, to
show a collaborator. Read-only on the MAS outputs; writes new artifact files
only. Does NOT run any MAS or touch the pipeline.

Both columns are "(in-sample ranking) vs out-of-sample GPBSV":
  IS-Shapley-VI = "Current: IS Shapley VI vs OOS GPBSV"   (reproduces manuscript Table 1 MAS)
  IS-GPBSV      = "IS GPBSV robustness: abs(IS GPBSV) vs saved OOS GPBSV"
                  (the referee's symmetric in-sample-vs-out-of-sample GPBSV check)
The shared "vs OOS-GPBSV" right-hand side is stated once in the caption.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
SNAP = OUT / "mas_snapshots" / "through_h12"
FORMATTED = SNAP / "mas_is_oos_gpbsv_v5_formatted_a0.67_mc1000000.csv"
RAW = SNAP / "mas_is_oos_gpbsv_v5_a0.67_mc1000000.csv"
if not FORMATTED.exists():
    FORMATTED = OUT / "mas_is_oos_gpbsv_v5_formatted_a0.67_mc1000000.csv"
    RAW = OUT / "mas_is_oos_gpbsv_v5_a0.67_mc1000000.csv"

PAPER = "Current: IS Shapley VI vs OOS GPBSV"
ISGPBSV = "IS GPBSV robustness: abs(IS GPBSV) vs saved OOS GPBSV"
L_PAPER = "IS-Shapley-VI"   # column label for the paper's MAS
L_IS = "IS-GPBSV"           # column label for the new robustness MAS
MODEL_ORDER = ["PCA", "ENet", "RF", "XGBoost", "Neural net",
               "Linear combination", "Nonlinear combination", "All models combination"]
SHORT = {"Linear combination": "Linear comb.", "Nonlinear combination": "Nonlinear comb.",
         "All models combination": "All-models comb."}
CAPTION = ("Model Agreement Score (MAS) of each in-sample importance ranking with the "
           "out-of-sample GPBSV ranking (higher = closer agreement; ***/**/* = p<=0.01/0.05/0.10). "
           "IS-Shapley-VI reproduces the paper's Table 1 MAS; IS-GPBSV is the symmetric "
           "in-sample-vs-out-of-sample GPBSV robustness check.")


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c) for c in df.columns]
    return df


def main() -> None:
    fmt, raw = load(FORMATTED), load(RAW)
    horizons = sorted([c for c in fmt.columns if c not in ("comparison_label", "model_label")], key=int)
    print(f"Source: {FORMATTED}\nHorizons available: {horizons}")

    def block(df, comp):
        return df[df["comparison_label"] == comp].set_index("model_label").reindex(MODEL_ORDER)

    fmt_paper, fmt_is = block(fmt, PAPER), block(fmt, ISGPBSV)
    raw_paper, raw_is = block(raw, PAPER), block(raw, ISGPBSV)

    cols = {}
    for h in horizons:
        cols[(f"h={h}", L_PAPER)] = fmt_paper[h].values
        cols[(f"h={h}", L_IS)] = fmt_is[h].values
    table = pd.DataFrame(cols, index=[SHORT.get(m, m) for m in MODEL_ORDER])
    table.index.name = "Model"
    mean_row = {}
    for h in horizons:
        mean_row[(f"h={h}", L_PAPER)] = f"{raw_paper[h].astype(float).mean():.3f}"
        mean_row[(f"h={h}", L_IS)] = f"{raw_is[h].astype(float).mean():.3f}"
    table.loc["Mean"] = pd.Series(mean_row)
    table.columns = pd.MultiIndex.from_tuples(table.columns)

    htag = "-".join(horizons)
    csv_path = OUT / f"mas_paper_vs_isgpbsv_h{htag}.csv"
    xlsx_path = OUT / f"mas_paper_vs_isgpbsv_h{htag}.xlsx"
    tex_path = OUT / f"mas_paper_vs_isgpbsv_h{htag}.tex"
    md_path = OUT / f"mas_paper_vs_isgpbsv_h{htag}.md"

    table.to_csv(csv_path)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name="MAS_paper_vs_ISGPBSV")

    # LaTeX: table float, per-horizon multicolumn groups, caption + note.
    lines = [r"\begin{table}[htbp]", r"\centering",
             rf"\caption{{{CAPTION}}}", r"\label{tab:mas_isgpbsv}",
             r"\begin{tabular}{l" + "cc" * len(horizons) + "}", r"\toprule"]
    lines.append(" & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{$h={h}$}}" for h in horizons]) + r" \\")
    lines.append(" ".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(horizons))))
    lines.append("Model & " + " & ".join([f"{L_PAPER} & {L_IS}"] * len(horizons)) + r" \\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        row = [SHORT.get(m, m)]
        for h in horizons:
            row += [str(fmt_paper.loc[m, h]), str(fmt_is.loc[m, h])]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    mrow = ["Mean"]
    for h in horizons:
        mrow += [f"{raw_paper[h].astype(float).mean():.3f}", f"{raw_is[h].astype(float).mean():.3f}"]
    lines.append(" & ".join(mrow) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    # Markdown.
    md = ["| Model | " + " | ".join(f"h={h} {L_PAPER} | h={h} {L_IS}" for h in horizons) + " |",
          "|" + "---|" * (1 + 2 * len(horizons))]
    for m in MODEL_ORDER:
        cells = [SHORT.get(m, m)]
        for h in horizons:
            cells += [str(fmt_paper.loc[m, h]), str(fmt_is.loc[m, h])]
        md.append("| " + " | ".join(cells) + " |")
    mcells = ["**Mean**"]
    for h in horizons:
        mcells += [f"**{raw_paper[h].astype(float).mean():.3f}**", f"**{raw_is[h].astype(float).mean():.3f}**"]
    md.append("| " + " | ".join(mcells) + " |")
    md_text = "\n".join(md) + f"\n\n*{CAPTION}*\n"
    md_path.write_text(md_text, encoding="utf-8")

    print("\nWrote: " + ", ".join(p.name for p in (csv_path, xlsx_path, tex_path, md_path)))
    print("\n--- MARKDOWN ---\n")
    print(md_text)


if __name__ == "__main__":
    sys.exit(main())
