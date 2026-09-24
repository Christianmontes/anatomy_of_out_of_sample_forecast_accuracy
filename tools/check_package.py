#!/usr/bin/env python3
"""Mechanical integrity checks for the replication package (standard library only).

    python tools/check_package.py                 # all checks
    python tools/check_package.py --load-artifacts  # also unpickle every artifact (needs the pinned env)

Checks (PASS/FAIL each, exit code 1 if any fails):
  required      every file the exhibit pipeline needs is present (data caches, Shapley bundles, MAS
                results, simulation reference outputs, on-path scripts, environment files)
  paths         no absolute machine paths (drive letters, /home, /Users, Dropbox), no /FAST_STORE
                outside comments and the documented Tier-3 script, no legacy 'Updated CPI 2' reads
                on the exhibit path
  attribution   no AI co-authorship text anywhere except the README acknowledgement; if the package
                is a git repository, a single human author and no Co-Authored-By trailers
  secrets       no API keys, tokens or passwords in text files
  encoding      text files are UTF-8; readme.txt and Data/csv/*.csv are plain ASCII with LF endings
  syntax        every .py file compiles
  sizes         no file above 50 MB (GitHub limit 100 MB), total size reported
  manifest      MANIFEST_sha256.txt (if present) matches the files on disk
  file_usage    if outputs/file_usage.txt exists (run_all_tier1.py --audit), no file outside the
                package was read; lists package files the exhibit run never opened (information)
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_EXT = {".py", ".ps1", ".md", ".txt", ".toml", ".cff", ".json", ".csv", ".tex", ".ini", ".yml", ".yaml", ".cfg", ".lock", ".r", ".log"}
SKIP_DIRS = {".git", ".venv", "__pycache__", ".idea", "build"}   # plus the top-level outputs/ (see walk)

REQUIRED = [
    "README.md", "readme.txt", "LICENSE", "CITATION.cff", "pyproject.toml", "poetry.lock", "poetry.toml",
    "env/requirements.txt", "env/requirements_iml39.txt", "env/pyproject-refit-py37.toml",
    "packages/anatomy/setup.py", "packages/anatomy/anatomy/__init__.py", "packages/anatomy/anatomy/_algorithm.py",
    "packages/anatomy/anatomy/_base.py", "packages/anatomy/anatomy/_mas.py", "packages/anatomy/anatomy/_models.py",
    "packages/anatomy/anatomy/_subsets.py", "packages/anatomy/LICENSE",
    "Code/V002/data.py", "Code/V002/data_cacher.py", "Code/V002/models.py", "Code/V002/library.py",
    "Code/V002/shap_algos.py", "Code/V002/pred_rev.py", "Code/V002/iml_rev.py", "Code/V002/iml_rev_ishapley.py",
    "Code/V002/rec.xlsx", "Code/V002/_runs/20241005_091334/config.json",
    "Data/Variable_list_CPI.xlsx",
    "Data/csv/fred_md_raw_2024-10-05.csv", "Data/csv/fred_md_transformed_2024-10-05.csv", "Data/csv/soc_2024-10-05.csv",
    "Results/Updated CPI 3/perf_raw.pickle", "Results/Updated CPI 3/ys_upd.bin",
    "Results/Updated CPI 3/MAS/wmas_a0.67.xlsx", "Results/Updated CPI 3/MAS/wmas_pval_a0.67.xlsx",
    "revision/mas_referee_response/mas_referee_variants.py", "revision/mas_referee_response/compute_insample_gpbsv_v5.py",
    "revision/mas_referee_response/make_paper_vs_isgpbsv_table.py",
    "revision/mas_referee_response/outputs/mas_is_oos_gpbsv_v5_a0.67_mc1000000.csv",
    "revision/mas_referee_response/outputs/mas_is_oos_gpbsv_v5_pvalues_a0.67_mc1000000.csv",
    "revision/mas_referee_response/outputs/mas_is_oos_gpbsv_v5_a0.67_mc1000000.tex",
    "revision/simulations/scripts/revision/run_gp_bsv_convergence_bands.py",
    "revision/simulations/scripts/revision/run_sim1_trap_dgps.py",
    "revision/simulations/scripts/revision/postprocess_sim1_paper.py",
    "revision/simulations/scripts/revision/build_sim1_referee_three_dgp_figures.py",
    "tools/run_all_tier1.py", "tools/verify_tables.py", "tools/export_data_csv.py", "tools/extract_y_base.py",
    "Models/README.md",
]
REQUIRED += ["Code/V002/_cache/%s.bin" % f for f in
             ["get_fred_md", "get_fred_md_raw", "get_soc", "get_epu", "get_goyal_welch", "get_ism", "get_jwurgler",
              "get_nrtz", "get_sludvigson", "get_soen"]]
for h in [1, 3, 6, 12]:
    REQUIRED += ["Results/Updated CPI 3/shapleys_h%d_upd.bin" % h,
                 "Results/Updated CPI 3/MAS/rankdiffs2_w_h%d_mc1000000_a0.67.pickle" % h,
                 "revision/mas_referee_response/outputs/insample_gpbsv_v5/insample_gpbsv_h%d_v5.bin" % h]
for dgp in ["friedman1", "polynomial", "threshold"]:
    for w in ["expanding", "rolling"]:
        REQUIRED.append("revision/simulations/outputs/convergence_bands/convergence_bands_overlay_%s_%s.png" % (dgp, w))
for d in ["01_dgp1_structural_break__break_b08_expanding", "02_dgp2_persistent_noise__persistent_rho099_rolling20",
          "03_dgp3_proxy_break__proxy_break_reverse_noisyx1_expanding"]:
    for m in ["ols", "enet", "rf"]:
        REQUIRED.append("revision/simulations/outputs/sim1_master_out/referee_outputs/three_dgp_figures/%s/by_model/%s/fig_importance_vs_gpbsv_avg_%s.pdf" % (d, m, m))
RECOMMENDED = ["Results/Updated CPI 3/y_base_nn_h%d.csv" % h for h in [1, 3, 6, 12]]

# absolute-path patterns; the allowlist names files where a documented hit is acceptable
DRIVE = r"(?<![A-Za-z0-9_])[A-Za-z]:\\(?![nrt0\\'\"])"   # E:\au, C:\Users; not the \n of "s:\n" in a string
DRIVE_FWD = r"(?<![A-Za-z0-9_])[A-Za-z]:/[A-Za-z]"   # C:/Users/... in R or forward-slash Python paths
PATH_PATTERNS = [DRIVE, DRIVE_FWD, r"/home/[a-z]", r"/Users/", r"Dropbox", r"/FAST_STORE", r"Updated CPI 2", r"sander-tr"]
PATH_ALLOW = {
    "Code/V002/pred_rev.py": {r"/FAST_STORE"},                     # Tier-3 refit script (documented)
    "Code/V002/iml_rev.py": {r"/FAST_STORE", r"Updated CPI 2"},    # comments + legacy functions off the exhibit path
    "Code/V002/iml_rev_ishapley.py": {r"/FAST_STORE"},             # comment
    "README.md": {r"/FAST_STORE", r"Updated CPI 2", DRIVE},  # documentation of the above and Windows examples
    "readme.txt": {r"/FAST_STORE", r"Updated CPI 2", DRIVE},
    "revision/mas_referee_response/README.md": {DRIVE},
    "revision/mas_referee_response/run_pipeline_h3h6h12.ps1": {DRIVE},
    "Models/README.md": {r"/FAST_STORE", DRIVE},
}
SELF = {"tools/check_package.py"}   # contains the patterns themselves
ATTRIB_PATTERNS = [r"co-authored-by", r"anthropic", r"claude"]
SECRET_PATTERNS = [r"(api_key|apikey|api-key|token|password|passwd|secret)\s*[=:]\s*['\"][A-Za-z0-9_\-]{8,}['\"]",
                   r"AKIA[0-9A-Z]{16}", r"ghp_[A-Za-z0-9]{20,}", r"sk-[A-Za-z0-9]{20,}"]


def walk():
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")
                   and not (root == PKG and d == "outputs")]   # only the top-level outputs/ is generated
        for f in files:
            p = os.path.join(root, f)
            yield os.path.relpath(p, PKG).replace(os.sep, "/"), p


def is_text(rel):
    return os.path.splitext(rel)[1].lower() in TEXT_EXT or os.path.basename(rel) in ("LICENSE", ".gitignore", ".gitattributes", ".gitkeep")


def read_text(p):
    with open(p, "rb") as fh:
        return fh.read().decode("utf-8")


class Report:
    def __init__(self):
        self.fails = 0

    def item(self, ok, name, msg):
        print("[%s] %-12s %s" % ("PASS" if ok else "FAIL", name, msg))
        self.fails += 0 if ok else 1

    def info(self, name, msg):
        print("[INFO] %-12s %s" % (name, msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-artifacts", action="store_true")
    args = ap.parse_args()
    rep = Report()
    files = list(walk())
    rels = {r for r, _ in files}

    # required files
    missing = [r for r in REQUIRED if r not in rels]
    rep.item(not missing, "required", "%d/%d required files present" % (len(REQUIRED) - len(missing), len(REQUIRED)) + ("; missing: " + ", ".join(missing) if missing else ""))
    rec_missing = [r for r in RECOMMENDED if r not in rels]
    rep.item(not rec_missing, "y_base", "y_base_nn_h*.csv present (needed for the CDSE figures without the 1.7 GB ishapley bins)" if not rec_missing
             else "missing %s: run tools/extract_y_base.py on the machine holding ishapley_h*_upd.bin" % rec_missing)

    # text scans
    path_hits, attrib_hits, secret_hits, bad_utf8 = [], [], [], []
    for rel, p in files:
        if not is_text(rel) or rel in SELF:
            continue
        try:
            t = read_text(p)
        except UnicodeDecodeError:
            bad_utf8.append(rel)
            continue
        low = t.lower()
        for pat in PATH_PATTERNS:
            if pat in PATH_ALLOW.get(rel, set()):
                continue
            for i, line in enumerate(t.splitlines(), 1):
                if re.search(pat, line):
                    if pat == r"/FAST_STORE" and line.lstrip().startswith("#"):
                        continue
                    path_hits.append("%s:%d (%s)" % (rel, i, pat))
                    break
        for pat in ATTRIB_PATTERNS:
            if re.search(pat, low):
                if rel in ("README.md", "readme.txt") and pat in ("anthropic", "claude"):
                    continue   # the acknowledgement sentence
                attrib_hits.append("%s (%s)" % (rel, pat))
        for pat in SECRET_PATTERNS:
            if re.search(pat, t, flags=re.I):
                secret_hits.append("%s (%s)" % (rel, pat[:20]))
    rep.item(not path_hits, "paths", "no machine-specific absolute paths on the exhibit path" if not path_hits else "; ".join(path_hits[:15]))
    rep.item(not attrib_hits, "attribution", "no AI attribution text outside the README acknowledgement" if not attrib_hits else "; ".join(attrib_hits[:15]))
    rep.item(not secret_hits, "secrets", "no credentials in text files" if not secret_hits else "; ".join(secret_hits))
    rep.item(not bad_utf8, "encoding", "all text files decode as UTF-8" if not bad_utf8 else "not UTF-8: " + ", ".join(bad_utf8))
    ascii_files = ["readme.txt"] + [r for r in rels if r.startswith("Data/csv/") and r.endswith(".csv")]
    non_ascii = []
    for rel in ascii_files:
        if rel in rels:
            b = open(os.path.join(PKG, rel), "rb").read()
            if any(c > 127 for c in b) or b"\r\n" in b:
                non_ascii.append(rel)
    rep.item(not non_ascii, "ascii", "readme.txt and Data/csv are plain ASCII with LF endings" if not non_ascii else "not plain ASCII/LF: " + ", ".join(non_ascii))

    # git identity (only if this folder is a git repository)
    if os.path.isdir(os.path.join(PKG, ".git")):
        try:
            authors = subprocess.check_output(["git", "-C", PKG, "log", "--format=%an <%ae>%n%cn <%ce>"], text=True).splitlines()
            bodies = subprocess.check_output(["git", "-C", PKG, "log", "--format=%B"], text=True).lower()
            ident = sorted(set(a for a in authors if a.strip()))
            trailers = [l for l in bodies.splitlines() if "co-authored-by" in l or "anthropic" in l or "claude" in l]
            bots = [i for i in ident if re.search(r"anthropic|claude|copilot|\[bot\]|openai|gemini", i, re.I)]
            rep.item(not bots and not trailers, "git",
                     "no AI or bot identity in the history and no attribution trailers; identities: %s" % ident
                     if not bots and not trailers else "AI/bot identities %s; trailer lines %s" % (bots, trailers[:5]))
        except Exception as e:  # git not available
            rep.info("git", "could not read the git history (%s)" % e)
    else:
        rep.info("git", "not a git repository yet (run after git init to check the commit identity)")

    # syntax
    bad_py = []
    for rel, p in files:
        if rel.endswith(".py"):
            try:
                compile(open(p, "rb").read(), p, "exec")
            except Exception as e:
                bad_py.append("%s: %s" % (rel, str(e).splitlines()[0]))
    rep.item(not bad_py, "syntax", "%d Python files compile" % sum(1 for r, _ in files if r.endswith(".py")) if not bad_py else "; ".join(bad_py))

    # sizes
    big = [(r, os.path.getsize(p)) for r, p in files if os.path.getsize(p) > 50 * 2 ** 20]
    total = sum(os.path.getsize(p) for _, p in files)
    rep.item(not big, "sizes", "%d files, %.1f MB total, largest %.1f MB" % (len(files), total / 1e6, max(os.path.getsize(p) for _, p in files) / 1e6)
             + ("; over 50 MB: %s" % big if big else ""))

    # manifest
    man = os.path.join(PKG, "MANIFEST_sha256.txt")
    if os.path.exists(man):
        bad, n, absent = [], 0, []
        for line in read_text(man).splitlines():
            digest, rel = line.split(" *", 1)
            p = os.path.join(PKG, rel)
            if not os.path.exists(p):
                absent.append(rel)
                continue
            n += 1
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                blob = fh.read()
            if is_text(rel):
                blob = re.sub(rb"\r+\n", b"\n", blob)   # manifest hashes LF bytes; tolerate CRLF checkouts
            h.update(blob)
            if h.hexdigest() != digest:
                bad.append(rel)
        rep.item(not bad and not absent, "manifest", "%d files match MANIFEST_sha256.txt" % n + ("; changed: %s" % bad[:10] if bad else "") + ("; listed but absent: %s" % absent[:10] if absent else ""))
        unlisted = sorted(r for r in rels if r not in {l.split(" *", 1)[1] for l in read_text(man).splitlines()} and r != "MANIFEST_sha256.txt" and not r.startswith("outputs/"))
        if unlisted:
            rep.info("manifest", "%d files not in the manifest (added after export): %s" % (len(unlisted), unlisted[:12]))
    else:
        rep.info("manifest", "no MANIFEST_sha256.txt")

    # file usage from run_all_tier1.py --audit
    fu = os.path.join(PKG, "outputs", "file_usage.txt")
    if os.path.exists(fu):
        lines = read_text(fu).splitlines()
        ext = [l for l in lines if l.startswith("EXTERNAL ") and not re.search(r"site-packages|/usr/|python3|\.cache|/dev/|/etc/|/proc/|\\\\Python|AppData|\.pyc|matplotlib|fonts", l)]
        rep.item(not ext, "file_usage", "the exhibit run read no file outside the package" if not ext else "external reads: " + "; ".join(ext[:10]))
        opened = {l for l in lines if not l.startswith("EXTERNAL")}
        never = sorted(r for r in rels if r not in opened and not r.endswith((".py", ".md", ".txt", ".toml", ".lock", ".cff", ".ps1", ".gitignore", ".gitattributes", ".gitkeep"))
                       and not r.startswith(("outputs/", "Data/csv/", "tools/", "env/", "packages/")))
        rep.info("file_usage", "%d data/artifact files were not opened by the Tier 0/1 run (reference outputs, Tier 1.5/2/3/S inputs); "
                 "see outputs/file_usage.txt" % len(never))
    else:
        rep.info("file_usage", "no outputs/file_usage.txt (run tools/run_all_tier1.py --audit to produce it)")

    # artifact loadability (pinned env)
    if args.load_artifacts:
        import pickle
        bad = []
        n = 0
        for rel, p in files:
            if rel.endswith((".bin", ".pickle")) and not rel.startswith("outputs/"):
                n += 1
                try:
                    pickle.load(open(p, "rb"))
                except Exception as e:
                    bad.append("%s: %s" % (rel, type(e).__name__))
        rep.item(not bad, "artifacts", "%d pickled artifacts load" % n if not bad else "; ".join(bad[:10]))

    print("\n%s" % ("ALL CHECKS PASSED" if rep.fails == 0 else "%d CHECK(S) FAILED" % rep.fails))
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
