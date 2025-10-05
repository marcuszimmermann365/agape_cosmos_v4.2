
import json, argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def agg(rows, label):
    df = pd.DataFrame([r for r in rows if r["label"]==label])
    def mean_ci(series):
        m = series.mean()
        se = series.std(ddof=1)/math.sqrt(len(series)) if len(series)>1 else 0.0
        ci = 1.96*se
        return f"{m:.2f} ± {ci:.2f}"
    return {"L_pp": mean_ci(df["L_pp"]), "Gini": mean_ci(df["Gini"]), "J4": mean_ci(df["J4"]), "SCM": mean_ci(df["SCM"])}, df

def ttest(full_df, ab_df, col):
    if len(full_df)<2 or len(ab_df)<2:
        return float('nan'), float('nan')
    t, p = stats.ttest_ind(full_df[col], ab_df[col], equal_var=False)
    return t, p

def make_table(results_path: Path, out_md: Path, alpha: float):
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    labels = ["V4.3_baseline","H1_fixed_teleology","H2_no_ahimsa","H3_static_reality"]
    agg_map = {}; df_map = {}
    for lbl in labels:
        agg_map[lbl], df_map[lbl] = agg(rows, lbl)

    md = []
    md.append("Metric (Final Value) | Baseline V4.3 | Ablation H1 (Fixed Teleology) | Ablation H2 (No Ahimsa) | Ablation H3 (Static Reality)")
    md.append("---|---:|---:|---:|---:")
    md.append(f"Avg. Suffering (L_pp) | {agg_map['V4.3_baseline']['L_pp']} | {agg_map['H1_fixed_teleology']['L_pp']} | {agg_map['H2_no_ahimsa']['L_pp']} | {agg_map['H3_static_reality']['L_pp']}")
    md.append(f"Suffering Gini Coeff. | {agg_map['V4.3_baseline']['Gini']} | {agg_map['H1_fixed_teleology']['Gini']} | {agg_map['H2_no_ahimsa']['Gini']} | {agg_map['H3_static_reality']['Gini']}")
    md.append(f"Avg. Empowerment (J4) | {agg_map['V4.3_baseline']['J4']} | {agg_map['H1_fixed_teleology']['J4']} | {agg_map['H2_no_ahimsa']['J4']} | {agg_map['H3_static_reality']['J4']}")
    md.append(f"Social Coherence | {agg_map['V4.3_baseline']['SCM']} | {agg_map['H1_fixed_teleology']['SCM']} | {agg_map['H2_no_ahimsa']['SCM']} | {agg_map['H3_static_reality']['SCM']}")
    out_md.write_text("\n".join(md), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, default="out/results.json")
    ap.add_argument("--make-table", type=str, default="table1.md")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    make_table(Path(args.results), Path(args.make_table), args.alpha)
    print("Wrote:", args.make_table)

if __name__ == "__main__":
    main()
