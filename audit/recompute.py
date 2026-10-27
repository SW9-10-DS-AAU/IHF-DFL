"""Recompute the paper's reported numbers from data/runs (via audit/load_all.py cache)."""
import numpy as np
import pandas as pd

from load_all import load

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
d = load()
runs, users = d["runs"], d["users"]
S = ["naive", "accuracy_only", "loss_only", "accuracy_loss"]


def section(t: str) -> None:
    print(f"\n{'=' * 10} {t} {'=' * 10}")


section("Config sanity per folder")
print(runs.groupby("folder")[["gpu", "timestamp", "epochs", "batch", "rounds", "good", "bad", "fr", "punish", "fee", "outlier"]]
      .agg(lambda s: sorted(set(map(str, s)))).to_string())
print("metadata vs filename mismatches:",
      int(((runs.md_strategy != runs.strategy) | (runs.md_act != runs.act) | (runs.md_noise != runs.noise)).sum()))
print("malicious config values:", runs[["md_mal_round", "md_mal_noise"]].drop_duplicates().to_dict("records"))
print("any commit hash in metadata keys:", [k for k in runs.meta_keys.iloc[0] if "commit" in k or "git" in k or "hash" in k])

section("Did the malicious participant ever behave maliciously?")
u = users.merge(runs[["eid", "dataset", "strategy", "act", "noise", "run"]], on="eid")
print(u.groupby("role")["behavior"].value_counts().unstack(fill_value=0))
print("attack_type by role:\n", u.groupby("role")["attack_type"].value_counts(dropna=False).unstack(fill_value=0))

section("Free-rider behavior switch round (first round with behavior=freerider) minus act")
fr = u[(u.role == "freerider") & (u.behavior == "freerider")].groupby("eid")["round"].min().rename("first_fr")
chk = runs.set_index("eid").join(fr)
print((chk.first_fr - chk.act).value_counts(dropna=False))

section("Permanent removals")
dq = u[u.state == "disqualified"].groupby(["eid", "user_id"])["round"].min().rename("removed_round").reset_index()
roles = u[u["round"] == 0][["eid", "user_id", "role"]]
roles = roles.merge(dq, on=["eid", "user_id"], how="left").merge(runs[["eid", "dataset", "strategy", "act", "noise"]], on="eid")
roles["removed"] = roles.removed_round.notna()
tab = roles.groupby(["dataset", "role", "strategy"]).agg(removed=("removed", "sum"), n=("removed", "size")).unstack("strategy")
print(tab.to_string())

section("Free-rider removal by dataset/noise/strategy/act (count and median delay)")
frr = roles[roles.role == "freerider"].copy()
frr["delay_minus_act"] = frr.removed_round - frr.act
frr["delay_minus_act_plus1"] = frr.removed_round - frr.act + 1
g = frr.groupby(["dataset", "noise", "strategy", "act"]).agg(
    removed=("removed", "sum"), med_r_minus_act=("delay_minus_act", "median"),
    med_r_minus_act_p1=("delay_minus_act_plus1", "median"), med_removed_round=("removed_round", "median"))
print(g.to_string())
print("\nLow-noise totals:", frr[frr.noise == 0.01].groupby("strategy")["removed"].sum().to_dict())
print("High-noise totals:", frr[frr.noise == 0.1].groupby("strategy")["removed"].sum().to_dict())

section("Contribution-score sums and ranges")
c = d["contributions"].merge(runs[["eid", "dataset", "strategy"]], on="eid")
sums = c.groupby(["eid", "round"])["contribution_score"].sum()
print("max |sum-1|:", float((sums - 1).abs().max()))
print(c.groupby(["dataset", "strategy"])["contribution_score"].agg(["min", "max"]).to_string())
# within-round variance, mean per run
v = c.groupby(["eid", "round", "dataset", "strategy"])["contribution_score"].var(ddof=0).groupby(["eid", "dataset", "strategy"]).mean()
print(v.groupby(["dataset", "strategy"]).describe()[["min", "50%", "max"]].to_string())
eq = c.groupby(["eid", "round", "strategy"])["contribution_score"].agg(lambda s: np.allclose(s, s.iloc[0]))
print("rounds with all-equal scores, by strategy:\n", eq.groupby("strategy").agg(["sum", "size"]))

section("Final accuracy")
gl = d["global"].merge(runs[["eid", "dataset", "strategy", "noise", "rounds"]], on="eid")
fin = gl[gl["round"] == gl["rounds"]]
print(fin.groupby(["dataset", "strategy"])["objective_global_accuracy"].agg(["mean", "std", "count"]).to_string())

section("Round-one gas")
rc = d["receipts"].merge(runs[["eid", "dataset", "strategy"]], on="eid")
r1 = rc[rc["round"] == 1]
tot = r1.groupby(["eid", "dataset", "strategy"])["gas_used"].sum().groupby(["dataset", "strategy"]).mean()
print(tot.unstack())
for ds in ["mnist", "cifar.10"]:
    base = tot[(ds, "naive")]
    print(ds, {s: round(100 * (tot[(ds, s)] / base - 1), 1) for s in S})
print(r1.groupby(["dataset", "strategy", "tx_type"])["gas_used"].sum().div(
    r1.groupby(["dataset", "strategy"])["eid"].nunique()).unstack().round(0).to_string())
