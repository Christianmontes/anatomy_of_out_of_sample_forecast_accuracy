#!/usr/bin/env python3
"""Compare the tables and figure includes of the LaTeX sources with the regenerated outputs.

The manuscript sources are not part of this package. Point the script at them:

    poetry run python tools/verify_tables.py --paper path/to/Anatomy.tex --ia path/to/AnatomyInternetAppendix.tex

Run tools/run_all_tier1.py first (it writes outputs/tables/*.csv and outputs/Figures/...).
Checks (each reports PASS/FAIL, exit code 1 if any fails):
  1. Table 1 (tab:mas, paper): 32 cells, value rounded to 2 decimals + significance stars
  2. Table A.2 (tab:rmse, IA): AR RMSE row (percent) + 32 ratio cells with Diebold-Mariano stars
  3. Table A.3 (tab:inGpbsv, IA): 64 cells + the average row
  4. Table A.1 (tab:fredmd, IA): listed abbreviations vs the FRED-MD columns in the data cache (report only)
  5. every \\includegraphics target of both files exists under outputs/Figures (commented lines ignored)
"""
import argparse
import os
import re
import sys

import pandas as pd

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, "outputs")

CELL = re.compile(r"\$\s*(?:\\quad|\\ |\s)*(\d)\$&\$(\d+)(\\%)?(?:\^\{([* ]*)\})?\s*\$")
ROWS_MAIN = ["Principal component regression", "Elastic net", "Random forest", "XGBoost",
             "Neural network", "Ensemble-linear", "Ensemble-nonlinear", "Ensemble-all"]
ROWS_A3 = ["PCR", "Elastic net", "Random forest", "XGBoost", "Neural network",
           "Ensemble-linear", "Ensemble-nonlinear", "Ensemble-all", "Average"]
HS = [1, 3, 6, 12]


def strip_comments(tex):
    return "\n".join(l for l in tex.splitlines() if not l.lstrip().startswith("%"))


def table_block(tex, label):
    i = tex.find("\\label{%s}" % label)
    if i < 0:
        raise KeyError("label %s not found" % label)
    start = tex.rfind("\\begin{tabular}", 0, i)
    return tex[start:i]


def parse_rows(block, names):
    rows = {}
    for line in block.splitlines():
        for n in names:
            if line.startswith(n) and "&" in line:
                rows.setdefault(n, []).append([(m.group(1) + "." + m.group(2), bool(m.group(3)), (m.group(4) or "").strip())
                                               for m in CELL.finditer(line)])
    return rows


class Report:
    def __init__(self):
        self.fails = 0

    def item(self, ok, msg):
        print("[%s] %s" % ("PASS" if ok else "FAIL", msg))
        self.fails += 0 if ok else 1


def check_table1(tex, rep):
    rows = parse_rows(table_block(tex, "tab:mas"), ROWS_MAIN)
    art = pd.read_csv(os.path.join(OUT, "tables", "table1_mas.csv"), index_col=0, keep_default_na=False)
    bad, n = [], 0
    for name in ROWS_MAIN:
        a = art[art.forecast == name].iloc[0]
        for h, (v, _, s) in zip(HS, rows[name][0]):
            n += 1
            want_v, want_s = "%.2f" % float(a["mas_h%d" % h]), str(a["stars_h%d" % h])
            if v != want_v or s != want_s:
                bad.append("%s h=%d tex %s%s vs artifact %s%s" % (name, h, v, s, want_v, want_s))
    rep.item(not bad and n == 32, "Table 1 (tab:mas): %d/32 cells match" % (n - len(bad)) + ("; " + "; ".join(bad) if bad else ""))


def check_tableA2(tex, rep):
    block = table_block(tex, "tab:rmse")
    rows = parse_rows(block, ROWS_MAIN + ["Autoregressive benchmark RMSE"])
    art = pd.read_csv(os.path.join(OUT, "tables", "tableA2_rmse.csv"), index_col=0, keep_default_na=False)
    bad, n = [], 0
    for h, (v, pct, _) in zip(HS, rows["Autoregressive benchmark RMSE"][0]):
        n += 1
        want = art.loc["Autoregressive benchmark RMSE", "h=%d" % h]
        if ("%s%%" % v) != want or not pct:
            bad.append("AR RMSE h=%d tex %s%% vs %s" % (h, v, want))
    for name in ROWS_MAIN:
        for h, (v, _, s) in zip(HS, rows[name][0]):
            n += 1
            want = art.loc[name, "h=%d" % h]            # e.g. 0.8176***
            m = re.match(r"([0-9.]+)(\**)", want)
            want_v, want_s = "%.2f" % float(m.group(1)), m.group(2)
            if v != want_v or s != want_s:
                bad.append("%s h=%d tex %s%s vs artifact %s%s" % (name, h, v, s, want_v, want_s))
    rep.item(not bad and n == 36, "Table A.2 (tab:rmse): %d/36 cells match" % (n - len(bad)) + ("; " + "; ".join(bad) if bad else ""))


def check_tableA3(tex, rep):
    rows = parse_rows(table_block(tex, "tab:inGpbsv"), ROWS_A3)
    art = pd.read_csv(os.path.join(OUT, "tables", "tableA3_insample_gpbsv_mas.csv"), index_col=0, keep_default_na=False)
    bad, n = [], 0
    for name in ROWS_A3:
        panels = rows[name]                        # [h1/h3 panel, h6/h12 panel]
        spec = [(1, "tsvi"), (1, "isgpbsv"), (3, "tsvi"), (3, "isgpbsv"), (6, "tsvi"), (6, "isgpbsv"), (12, "tsvi"), (12, "isgpbsv")]
        cells = panels[0] + panels[1]
        for (h, tag), (v, _, s) in zip(spec, cells):
            n += 1
            want_v = "%.2f" % float(art.loc[name, "mas_%s_h%d" % (tag, h)])
            want_s = "" if name == "Average" else str(art.loc[name, "stars_%s_h%d" % (tag, h)])
            if v != want_v or s != want_s:
                bad.append("%s h=%d %s tex %s%s vs artifact %s%s" % (name, h, tag, v, s, want_v, want_s))
    rep.item(not bad and n == 72, "Table A.3 (tab:inGpbsv): %d/72 cells match (values, stars, averages)" % (n - len(bad)) + ("; " + "; ".join(bad) if bad else ""))


def check_tableA1(tex, rep):
    # the table starts at \caption{FRED-MD variables} and continues in "(continued)" tables after the label
    start = tex.find("\\caption{FRED-MD variables")
    end = tex.find("\\label{tab:fredmd}")
    while True:
        j = tex.find("(continued)}", end)
        k = tex.find("\\end{table}", end)
        if j < 0 or j > tex.find("\\section", end) > 0 and tex.find("\\section", end) < j:
            break
        end = tex.find("\\end{table}", j)
    block = tex[start:end]
    norm = lambda x: re.sub(r"[\\\s:]", "", x.lower())   # 's\&p:divyield' and 's&p div yield' -> 's&pdivyield'
    listed = [norm(m.group(1)) for m in re.finditer(r"\{\\tt\s+([^}]+)\}", block)]
    cache = pd.read_csv(os.path.join(OUT, "tables", "tableA1_predictor_list.csv"))
    fred = [norm(c) for c in cache[cache.source == "FRED-MD"].predictor]
    not_in_cache = sorted(set(listed) - set(fred))
    not_listed = sorted(set(fred) - set(listed))
    print("[INFO] Table A.1 (tab:fredmd): %d abbreviations listed, %d of them in the 2024-10-05 FRED-MD cache (%d columns)"
          % (len(listed), len(listed) - len(not_in_cache), len(fred)))
    print("       listed but absent from the cache vintage: %s" % (not_in_cache or "none"))
    print("       cache columns not listed (target and series not used as predictors): %s" % (not_listed or "none"))


def check_includes(tex, rep, tag):
    inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", strip_comments(tex))
    missing = [p for p in inc if not os.path.exists(os.path.join(OUT, p))]
    rep.item(not missing, "%s: %d/%d \\includegraphics targets present under outputs/" % (tag, len(inc) - len(missing), len(inc))
             + ("; missing: " + ", ".join(missing) if missing else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True, help="Anatomy.tex")
    ap.add_argument("--ia", required=True, help="AnatomyInternetAppendix.tex")
    args = ap.parse_args()
    paper = open(args.paper, encoding="utf-8").read()
    ia = open(args.ia, encoding="utf-8").read()
    rep = Report()
    check_table1(paper, rep)
    check_tableA2(ia, rep)
    check_tableA3(ia, rep)
    check_tableA1(ia, rep)
    check_includes(paper, rep, "Anatomy.tex")
    check_includes(ia, rep, "AnatomyInternetAppendix.tex")
    print("\n%s" % ("ALL CHECKS PASSED" if rep.fails == 0 else "%d CHECK(S) FAILED" % rep.fails))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
