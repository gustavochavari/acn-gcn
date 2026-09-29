"""
Recomputes the paired significance tests of the paper (ACN-GCN vs. every
baseline) from the saved per-run accuracies, without retraining.

The 50 accuracies of every model are aligned by (split, seed), so the tests are
paired: Wilcoxon signed-rank (reported in the paper) and paired t-test.

ACN-GCN is taken with the normalization main.py selected by mean VALIDATION
accuracy. Validation accuracies are not stored in the JSON files, so the
selection is read from the run log and fixed below.
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SELECTED_NORM = {
    "Texas":              "deep_sym",
    "Wisconsin":          "deep_sym",
    "Cornell":            "deep_sym",
    "chameleon-filtered": "src",
    "squirrel-filtered":  "src",
}

BASELINE_KEYS = ["GCN", "GAT", "GraphSAGE", "H2GCN", "GPRGNN", "APPNP",
                 "MixHop", "CurvGN", "FAGCN", "ACM-GCN", "A2GCN"]


def paired_significance(a, b):
    """Same computation as paired_significance() in main.py."""
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    if np.allclose(diff, 0):
        return dict(mean_diff=0.0, p_ttest=1.0, p_wilcoxon=1.0)
    out = dict(mean_diff=float(diff.mean()),
               p_ttest=float(ttest_rel(a, b).pvalue))
    try:
        out["p_wilcoxon"] = float(wilcoxon(a, b).pvalue)
    except ValueError:
        out["p_wilcoxon"] = float("nan")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results_dir", default=os.path.join(REPO_ROOT, "results_paper"))
    p.add_argument("--norm", default=None,
                   help="Normalization of ACN-GCN to test on every dataset "
                        "(default: the one selected by validation, see SELECTED_NORM)")
    args = p.parse_args()

    for ds, sel in SELECTED_NORM.items():
        norm = args.norm or sel
        base = os.path.join(args.results_dir, ds)
        with open(os.path.join(base, "norm_sweep", "all_norms.json"), encoding="utf-8") as f:
            baselines = json.load(f)["baselines"]
        with open(os.path.join(base, "ablation", "chain_sym.json"), encoding="utf-8") as f:
            head = json.load(f)[norm]["ACN-GCN"]

        print(f"\n{ds}  -  ACN-GCN + {norm}  "
              f"({100 * np.mean(head):.1f} +- {100 * np.std(head, ddof=1):.1f}, n={len(head)})")
        for key in BASELINE_KEYS:
            b = baselines.get(key, [])
            if len(b) != len(head):
                print(f"  {key:<10}: incompatible number of runs, skipped")
                continue
            s = paired_significance(head, b)
            star = "*" if s["p_wilcoxon"] < 0.05 else " "
            print(f"  {key:<10}: d_acc={100 * s['mean_diff']:+5.1f} pp  "
                  f"p_wilcoxon={s['p_wilcoxon']:.4f}{star}  p_t={s['p_ttest']:.4f}")


if __name__ == "__main__":
    main()
