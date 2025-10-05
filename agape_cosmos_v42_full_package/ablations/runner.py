
import os, json, argparse, importlib
from pathlib import Path
OUT = Path("out"); OUT.mkdir(exist_ok=True)

def load_engine():
    return importlib.import_module("agape_v42_sim.engine")

def run_case(engine, label, base_cfg, seeds):
    rows = []
    for s in seeds:
        cfg = dict(base_cfg); cfg.update({"seed": s})
        res = engine.run_experiment(cfg)
        rows.append({"label": label, "seed": s, "L_pp": float(res["L_pp_final"]), "Gini": float(res["Gini_suffering"]), "J4": float(res["J4_final"]), "SCM": float(res["SCM_final"])})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jivas", type=int, default=16)
    ap.add_argument("--outer", type=int, default=50)
    ap.add_argument("--inner", type=int, default=20)
    args = ap.parse_args()

    engine = load_engine()
    base = {"N_jivas": args.jivas, "outer_steps": args.outer, "inner_steps": args.inner, "use_hamiltonian_sde": True, "use_mi_empowerment": True}
    seeds = list(range(args.seeds))

    rows_full = run_case(engine, "V4.3_baseline", base, seeds)
    cfg_h1 = dict(base); cfg_h1.update({"teleology_mode": "fixed_uniform"})
    rows_h1 = run_case(engine, "H1_fixed_teleology", cfg_h1, seeds)
    cfg_h2 = dict(base); cfg_h2.update({"disable_ahimsa": True})
    rows_h2 = run_case(engine, "H2_no_ahimsa", cfg_h2, seeds)
    cfg_h3 = dict(base); cfg_h3.update({"static_reality": True})
    rows_h3 = run_case(engine, "H3_static_reality", cfg_h3, seeds)

    all_rows = rows_full + rows_h1 + rows_h2 + rows_h3
    out_json = OUT / "results.json"
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print("Wrote:", out_json)

if __name__ == "__main__":
    main()
