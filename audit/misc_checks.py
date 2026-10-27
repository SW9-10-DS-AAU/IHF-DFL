"""MAD filtering rate, loss cap hits, round-1 initialisation, seeding, and significance of Loss vs Accuracy."""
from math import comb

import pandas as pd

from load_all import load


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n1, n2, k = a + b, c + d, a + c
    p = lambda x: comb(n1, x) * comb(n2, k - x) / comb(n1 + n2, k)
    p0 = p(a)
    return sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= p0 * (1 + 1e-9))


pd.set_option("display.width", 250)
d = load()
runs = d["runs"][["eid", "dataset", "strategy", "act", "noise", "run"]]
cm = d["contributions_mad"].merge(runs, on="eid")
n_ex, n_ac = cm.current_excluded_values.apply(len), cm.current_accepted_values.apply(len)
cm["excl_frac"] = n_ex / (n_ex + n_ac)
print("MAD tau=1.1: mean fraction of peer reports discarded per (user, round):")
print(cm.groupby(["dataset", "strategy", "metric"]).excl_frac.mean().round(3).to_string())
v = d["votes"].merge(runs, on="eid")
print("\nvote_loss at uint16 cap 65535:", int((v.vote_loss >= 65535).sum()), "of", len(v))
u = d["users"].merge(runs, on="eid")
g = d["global"].merge(runs, on="eid")
print("\nround-1 mean honest local val acc:", u[(u["round"] == 1) & (u.role == "good")].groupby("dataset").subjective_personal_accuracy.mean().round(3).to_dict())
for r in [0, 1, 2]:
    print(f"round-{r} global acc:", g[g["round"] == r].groupby("dataset").objective_global_accuracy.mean().round(3).to_dict())
z = g[g["round"] == 0].groupby(["dataset", "run"]).objective_global_accuracy.nunique()
print("distinct round-0 global accuracies per (dataset, run_id) across its 24 configs: min", z.min(), "max", z.max())
users = d["users"]
dq = users[users.state == "disqualified"][["eid", "user_id"]].drop_duplicates()
fr = users[(users["round"] == 0) & (users.role == "freerider")][["eid", "user_id"]]
frr = fr.merge(dq, on=["eid", "user_id"]).eid
for ds in ["mnist", "cifar.10", None]:
    def count(s: str) -> tuple[int, int]:
        rr = runs[(runs.strategy == s) & (runs.noise == 0.01) & ((runs.dataset == ds) if ds else True)]
        return int(rr.eid.isin(frr).sum()), len(rr)
    (a, n), (b, m) = count("loss_only"), count("accuracy_only")
    print(f"{ds or 'both'}: loss {a}/{n} vs accuracy {b}/{m}  Fisher two-sided p={fisher_two_sided(a, n - a, b, m - b):.4f}")
