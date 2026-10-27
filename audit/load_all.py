"""Load every run pickle under data/runs into tidy DataFrames (read-only), cached to audit/cache.pkl."""
# Pickles are the project's own experiment outputs (data/runs), not third-party files.
import pickle
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "runs"
CACHE = Path(__file__).resolve().parent / "cache.pkl"

NAME_RE = re.compile(
    r"^(?P<preset>[^-]+)-(?P<dataset>cifar\.10|mnist)-(?P<strategy>[a-z_]+)-(?P<outlier>True|False)-"
    r"(?P<act>\d+)-(?P<noise>[0-9.]+)-(?P<mal_round>[^-]+)-(?P<mal_noise>[^-]+)-(?P<run>\d+)-\{(?P<uuid>[0-9a-f-]+)\}$"
)


def load():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    runs, tables = [], {k: [] for k in ["global", "users", "votes", "receipts", "contributions", "contributions_mad", "punishments", "warnings"]}
    for f in sorted(ROOT.rglob("*.pkl")):
        m = NAME_RE.match(f.stem)
        assert m, f.stem
        p = pickle.load(open(f, "rb"))
        md = p["metadata"]
        info = dict(
            eid=p["experiment_id"], folder=f.parent.name, dataset=m["dataset"], strategy=m["strategy"],
            act=int(m["act"]), noise=float(m["noise"]), run=int(m["run"]),
            md_strategy=md.get("contribution_score_strategy"), md_act=md.get("freerider_start_round"),
            md_noise=md.get("freerider_noise_scale"), md_mal_round=md.get("malicious_start_round"),
            md_mal_noise=md.get("malicious_noise_scale"), timestamp=md.get("timestamp"),
            gpu=p.get("setup", {}).get("hardware", {}).get("gpu_name"),
            total_time=p.get("setup", {}).get("total_experiment_time"),
            epochs=md.get("epochs"), batch=md.get("batch_size"), rounds=md.get("minimum_rounds"),
            good=md.get("number_of_good_contributors"), bad=md.get("number_of_bad_contributors"),
            fr=md.get("number_of_freerider_contributors"), punish=md.get("punish_factor"),
            fee=md.get("first_round_fee"), reward=md.get("reward"), buyin=md.get("min_buy_in"),
            outlier=md.get("use_outlier_detection"), meta_keys=sorted(md.keys()),
        )
        runs.append(info)
        for k, df in p["tables"].items():
            if k in tables and isinstance(df, pd.DataFrame) and not df.empty:
                df = df.copy()
                df["eid"] = p["experiment_id"]
                tables[k].append(df)
    runs = pd.DataFrame(runs)
    out = {"runs": runs, **{k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in tables.items()}}
    pickle.dump(out, open(CACHE, "wb"))
    return out


if __name__ == "__main__":
    d = load()
    r = d["runs"]
    print(len(r), "runs")
    print(r.groupby(["dataset", "noise", "strategy", "act"]).size().unstack())
    for k, v in d.items():
        print(k, getattr(v, "shape", None))
