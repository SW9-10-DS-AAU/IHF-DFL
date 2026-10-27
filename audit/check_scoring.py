"""Which normalization produced the stored scores, and what mechanism removed honest participants?"""
import numpy as np
import pandas as pd

from load_all import load

pd.set_option("display.width", 250)
d = load()
runs = d["runs"][["eid", "dataset", "strategy", "act", "noise"]]


def eq7(diffs: list[float]) -> list[float]:
    s = sum(diffs)
    return [1 / len(diffs)] * len(diffs) if s == 0 else [x / s for x in diffs]


def new_norm(vals: list[float], prev: float, metric: str) -> list[float]:
    """Copy of contracts/contribution.py:normalize_contribution_scores_new on main."""
    vals = [v - prev for v in vals]
    if metric == "loss":
        vals = [-v for v in vals]
    s = sum(vals)
    mx = max(vals)
    if mx == 0:
        vals = [1 if v == 0 else v for v in vals]
    elif mx < 0:
        vals = [s / v for v in vals]
    if min(vals) < -1:
        div = -min(vals)
        vals = [v / div if v < 0 else v for v in vals]
    s = sum(vals)
    if not s == 1:
        if min(vals) >= 0:
            vals = [v / s for v in vals]
        else:
            sp = sum(v for v in vals if v > 0)
            ex = s - 1
            vals = [v + (v / sp) * -ex if v > 0 else v for v in vals]
    return vals


cm = d["contributions_mad"].merge(runs, on="eid")
c = d["contributions"][["eid", "round", "user_id", "contribution_score"]]
rows = []
for strat, metric in [("accuracy_only", "accuracy"), ("loss_only", "loss")]:
    sub = cm[(cm.strategy == strat) & (cm.metric == metric)].merge(c, on=["eid", "round", "user_id"])
    for (eid, rnd), g in sub.groupby(["eid", "round"]):
        g = g.sort_values("user_id")
        if len(g) <= 3:
            continue
        prev = g.prev_avg_round_val_after_mad.iloc[0]
        vals = g.user_mad_avg.tolist()
        diffs = [v - prev for v in vals] if metric == "accuracy" else [prev - v for v in vals]
        stored = g.contribution_score.to_numpy()
        rows.append(dict(strategy=strat, eid=eid, round=rnd,
                         err_new=np.max(np.abs(stored - new_norm(vals, prev, metric))),
                         err_eq7=np.max(np.abs(stored - eq7(diffs))),
                         sign_flip=sum(diffs) < 0))
r = pd.DataFrame(rows)
print("Rounds checked:", len(r))
print(r.groupby("strategy").agg(match_new=("err_new", lambda e: (e < 1e-9).mean()),
                                match_eq7=("err_eq7", lambda e: (e < 1e-9).mean()),
                                rounds_with_negative_sum=("sign_flip", "sum")).to_string())

# Accuracy+Loss: does it match Eq. 7-10 (average of the two signed-sum normalizations)?
rows = []
sub = cm[cm.strategy == "accuracy_loss"].merge(c, on=["eid", "round", "user_id"])
for (eid, rnd), g in sub.groupby(["eid", "round"]):
    a = g[g.metric == "accuracy"].sort_values("user_id")
    l = g[g.metric == "loss"].sort_values("user_id")
    if len(a) <= 3 or len(a) != len(l):
        continue
    na = eq7([v - a.prev_avg_round_val_after_mad.iloc[0] for v in a.user_mad_avg])
    nl = eq7([l.prev_avg_round_val_after_mad.iloc[0] - v for v in l.user_mad_avg])
    pred = (np.array(na) + np.array(nl)) / 2
    rows.append(np.max(np.abs(a.contribution_score.to_numpy() - pred)))
print("accuracy_loss rounds matching Eq.10 with Eq.7:", np.mean(np.array(rows) < 1e-9), "of", len(rows))

# Removal mechanism for every permanently removed participant
u = d["users"]
p = d["punishments"].merge(runs, on="eid")
roles = u[u["round"] == 0][["eid", "user_id", "role"]]
dis = p[p.punishment_type.str.contains("disq", case=False)].merge(roles, on=["eid", "user_id"])
print("\npunishment types:", d["punishments"].punishment_type.value_counts().to_dict())
cs = c.rename(columns={"contribution_score": "score_at_removal"})
dis = dis.merge(cs, on=["eid", "round", "user_id"], how="left")
grs_prev = u[["eid", "round", "user_id", "grs"]].copy()
grs_prev["round"] += 1
dis = dis.merge(grs_prev.rename(columns={"grs": "grs_before"}), on=["eid", "round", "user_id"], how="left")
dis["cause"] = np.select(
    [dis.score_at_removal.isna(), dis.score_at_removal < -3, dis.score_at_removal < 0],
    ["RRS<0 (vote)", "score<-3 (instant)", "negative score, GRS<=1/3"], "other")
print(dis.groupby(["dataset", "strategy", "role", "cause"]).size().unstack(fill_value=0).to_string())
print("\nGRS (ETH) before an instant |c|>3 disqualification:")
print((dis[dis.cause == "score<-3 (instant)"].groupby(["dataset", "role"]).grs_before.describe() / 1e18)[["count", "min", "50%", "max"]])
