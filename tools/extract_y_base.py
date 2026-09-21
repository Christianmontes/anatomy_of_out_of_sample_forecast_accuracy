#!/usr/bin/env python3
"""Extract y_base (average base value of the two neural nets per out-of-sample period)
from the multi-GB ishapley_h*_upd.bin files into small CSVs that ship with the package.

Run ONCE on the machine that holds the ishapley bins (Windows, Poetry env, 64 GB RAM):
    cd <repo root>
    poetry run python tools\extract_y_base.py

Writes Results/Updated CPI 3/y_base_nn_h{1,3,6,12}.csv (index = ys_upd.bin index as text,
column y_base, full double precision). iml_rev.plot_cumsse() reads these CSVs and falls back
to the bins only when a CSV is absent.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "Results", "Updated CPI 3")


def main():
    ys = pickle.load(open(os.path.join(RES, "ys_upd.bin"), "rb"))
    for h in [1, 3, 6, 12]:
        src = os.path.join(RES, "ishapley_h%i_upd.bin" % h)
        out = os.path.join(RES, "y_base_nn_h%i.csv" % h)
        if not os.path.exists(src):
            print("SKIP h=%i: %s not found" % (h, src))
            continue
        print("h=%i: loading %s ..." % (h, src), flush=True)
        A = pickle.load(open(src, "rb"))
        n = ys[h].shape[0]
        y_base = pd.Series(
            [(A["nn_deep"][i]["base_value"][0] + A["nn_shallow"][i]["base_value"][0]) / 2 for i in range(n)],
            index=ys[h].index.astype(str), name="y_base",
        )
        try:
            y_base.to_csv(out, float_format="%.17g", index_label="period", lineterminator="\n")
        except TypeError:  # pandas < 1.5 (the pinned 1.1.5) spells it line_terminator
            y_base.to_csv(out, float_format="%.17g", index_label="period", line_terminator="\n")
        del A
        # float_precision="round_trip": pandas' default float parser is not bit-exact
        chk = pd.read_csv(out, index_col=0, float_precision="round_trip")["y_base"]
        assert len(chk) == n and np.array_equal(chk.values, y_base.values, equal_nan=True), "round-trip check failed"
        print("  wrote %s (%d rows)" % (out, n))
    print("done")


if __name__ == "__main__":
    sys.exit(main())
