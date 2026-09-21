#!/usr/bin/env python3
"""Export the pickled input-data caches to plain-text CSV (archival copies).

The forecasting code reads the pickled pandas DataFrames in Code/V002/_cache/ (the
2024-10-05 data vintage). These CSVs are the same data in a format that does not depend
on a pandas version, as required by the JAE Data Archive. They are not read by the code.

Run from the package root (pinned environment):
    poetry run python tools/export_data_csv.py            # write Data/csv/*.csv
    poetry run python tools/export_data_csv.py --check    # verify CSV == pickle, exit 1 on mismatch
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(PKG, "Code", "V002", "_cache")
OUT = os.path.join(PKG, "Data", "csv")

# cache file -> CSV name (vintage date in the name), description
EXPORTS = {
    "get_fred_md_raw.bin": ("fred_md_raw_2024-10-05.csv", "FRED-MD monthly panel, raw levels as downloaded (McCracken and Ng, 2016), 2024-10-05 vintage"),
    "get_fred_md.bin": ("fred_md_transformed_2024-10-05.csv", "FRED-MD monthly panel after the McCracken-Ng transformation codes (data.py: get_fred_md)"),
    "get_soc.bin": ("soc_2024-10-05.csv", "University of Michigan Surveys of Consumers: ICS, ICE, ICC indices (data.py: get_soc), retrieved 2024-10-05"),
}


def to_csv(df, path):
    kw = {"float_format": "%.17g", "index_label": "date"}
    try:
        df.to_csv(path, lineterminator="\n", **kw)
    except TypeError:  # pandas < 1.5 spells it line_terminator
        df.to_csv(path, line_terminator="\n", **kw)


def load_csv(path):
    # float_precision="round_trip": pandas' default fast parser is not bit-exact
    return pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")


def same(df, back):
    if list(df.columns) != list(back.columns):
        return False, "columns differ"
    if len(df.index) != len(back.index) or not (pd.DatetimeIndex(df.index) == back.index).all():
        return False, "index differs"
    a, b = df.to_numpy(dtype=float), back.to_numpy(dtype=float)
    if not (np.isnan(a) == np.isnan(b)).all():
        return False, "NaN pattern differs"
    m = ~np.isnan(a)
    if not np.array_equal(a[m], b[m]):
        return False, "values differ (max abs diff %.3g)" % np.nanmax(np.abs(a - b))
    return True, "identical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify existing CSVs against the pickles")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ok = True
    for bin_name, (csv_name, desc) in EXPORTS.items():
        df = pickle.load(open(os.path.join(CACHE, bin_name), "rb"))
        path = os.path.join(OUT, csv_name)
        if not args.check:
            to_csv(df, path)
        good, msg = same(df, load_csv(path))
        ok &= good
        print("%-40s %-6s %s  (%d rows x %d cols)" % (csv_name, "OK" if good else "FAIL", msg, *df.shape))
    # the other caches are None placeholders (loaders not used in the paper run)
    for f in sorted(os.listdir(CACHE)):
        if f.endswith(".bin") and f not in EXPORTS:
            obj = pickle.load(open(os.path.join(CACHE, f), "rb"))
            print("%-40s placeholder: %s" % (f, type(obj).__name__))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
