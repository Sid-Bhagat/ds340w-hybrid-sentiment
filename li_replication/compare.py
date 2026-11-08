"""Reproducibility checks.

  python compare.py september RESULTS_DIR [--old-dir old/outputs]
      Compare a run on the September cache (old/data) with the September
      outputs. HMM state labels are arbitrary, so each ticker's states are
      matched to the September states by the permutation that fits best; the
      Eq. 16 state effects are compared as state intercepts (const + dummy).

  python compare.py identical DIR_A DIR_B
      Check that two runs produced byte-identical files.
"""

import argparse
import filecmp
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-8  # relative tolerance for "match"


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def _intercepts(eq16):
    """State intercepts from Eq. 16: state 0 = const, state k = const + c_k."""
    const = eq16.loc[eq16.term == "const", "coef"].iloc[0]
    out = {0: const}
    for _, r in eq16[eq16.term.str.startswith("state_")].iterrows():
        out[int(r.term.split("_")[1])] = const + r.coef
    return out


def compare_september(new_dir, old_dir):
    old = pd.read_csv(old_dir / "regression_summary.csv")
    eq16 = pd.read_csv(new_dir / "eq16.csv")
    eq17 = pd.read_csv(new_dir / "eq17.csv")
    sel = pd.read_csv(new_dir / "hmm_state_selection.csv")
    rows = []

    for t in eq16.ticker.unique():
        o16 = old[(old.ticker == t) & (old.model == "stock_level (eq16)")]
        n16 = eq16[eq16.ticker == t]
        o17 = old[(old.ticker == t) & old.model.str.endswith("(eq17)")].copy()
        o17["state"] = o17.model.str.extract(r"state_(\d+)")[0].astype(int)
        n17 = eq17[eq17.ticker == t]

        # Best permutation new state -> old state, judged on Eq. 17 coefficients.
        new_states = sorted(n17.state.unique())
        old_states = sorted(o17.state.unique())
        best, best_err = None, np.inf
        if len(new_states) == len(old_states):
            for perm in itertools.permutations(old_states):
                m = dict(zip(new_states, perm))
                err = 0.0
                for s in new_states:
                    a = n17[n17.state == s].set_index("term").coef
                    b = o17[o17.state == m[s]].set_index("term").coef
                    err = max(err, _rel(a, b.reindex(a.index)).max())
                if err < best_err:
                    best, best_err = m, err
        identity = best is not None and all(k == v for k, v in best.items())

        # Label-free Eq. 16 terms.
        for term in ["sp500_return", "momentum", "volume"]:
            a = n16.set_index("term").loc[term]
            b = o16.set_index("term").loc[term]
            rows.append((t, "eq16", term, a.coef, b.coef, a.pvalue, b.pvalue))
        # Eq. 16 state intercepts under the permutation.
        if best is not None:
            ni, oi = _intercepts(n16), _intercepts(o16.assign(term=o16.term))
            for s in new_states:
                rows.append((t, "eq16", f"intercept_state{s}->old{best[s]}",
                             ni[s], oi[best[s]], np.nan, np.nan))
            for s in new_states:
                a = n17[n17.state == s].set_index("term")
                b = o17[o17.state == best[s]].set_index("term")
                for term in a.index:
                    rows.append((t, f"eq17 state{s}->old{best[s]}", term,
                                 a.loc[term, "coef"], b.loc[term, "coef"],
                                 a.loc[term, "pvalue"], b.loc[term, "pvalue"]))
        print(f"{t}: state mapping new->old {best} "
              f"({'same labels' if identity else 'relabelled'}), "
              f"max rel diff eq17 coefs {best_err:.2e}")

        # AIC/BIC table.
        os_ = pd.read_csv(old_dir / f"state_selection_{t}.csv")
        ns = sel[sel.ticker == t].reset_index(drop=True)
        for c in ["logL", "AIC", "BIC"]:
            for k, a, b in zip(ns.n_states, ns[c], os_[c]):
                rows.append((t, "state_selection", f"{c}_k{k}", a, b, np.nan, np.nan))

    df = pd.DataFrame(rows, columns=["ticker", "block", "term", "coef_new", "coef_old",
                                     "p_new", "p_old"])
    df["rel_diff_coef"] = _rel(df.coef_new, df.coef_old)
    df["abs_diff_p"] = (df.p_new - df.p_old).abs()
    ok = (df.rel_diff_coef < TOL) & ((df.abs_diff_p < TOL) | df.abs_diff_p.isna())
    df["match"] = ok
    print(f"\n{ok.sum()}/{len(df)} values match within rel tol {TOL:g}; "
          f"max rel coef diff {df.rel_diff_coef.max():.2e}, "
          f"max abs p diff {df.abs_diff_p.max():.2e}")
    if not ok.all():
        print(df[~ok].to_string(index=False))

    # Figures: compare decoded pixels (PNG metadata differs).
    import matplotlib.image as mpimg
    figs_same, figs = 0, sorted((old_dir).glob("*.png"))
    for f in figs:
        a, b = mpimg.imread(new_dir / "figures" / f.name), mpimg.imread(f)
        figs_same += a.shape == b.shape and np.array_equal(a, b)
    print(f"figures pixel-identical: {figs_same}/{len(figs)}")
    return df


def compare_identical(a, b):
    fa = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    fb = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    diff = [str(p) for p in fa if p.name != "run_info.json"
            and not filecmp.cmp(a / p, b / p, shallow=False)]
    print(f"{len(fa)} files in A, {len(fb)} in B; same file list: {fa == fb}")
    print(f"byte-identical (excluding run_info.json): {len(fa) - 1 - len(diff)}/{len(fa) - 1}")
    for d in diff:
        print("  differs:", d)
    return fa == fb and not diff


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("september")
    s.add_argument("results_dir", type=Path)
    s.add_argument("--old-dir", type=Path, default=Path(__file__).parent / "old" / "outputs")
    s.add_argument("--save", type=Path, help="write the comparison table to this CSV")
    i = sub.add_parser("identical")
    i.add_argument("dir_a", type=Path)
    i.add_argument("dir_b", type=Path)
    args = ap.parse_args()

    if args.cmd == "september":
        df = compare_september(args.results_dir, args.old_dir)
        if args.save:
            from lirep.io_utils import save_csv
            save_csv(df, args.save, index=False)
        sys.exit(0 if df.match.all() else 1)
    sys.exit(0 if compare_identical(args.dir_a, args.dir_b) else 1)


if __name__ == "__main__":
    main()
