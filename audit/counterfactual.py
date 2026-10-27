"""Offline replay of on-chain settlement under (a) the current contract rules and (b) the paper's Eq. 7 + Eq. 11.

Holds each run's observed merge sets, filtered peer reports and passive punishments fixed and recomputes
scores, GRS and removals. Rule (a) must reproduce the observed removals; that validates the replay.
Approximation: in the real system, changed GRS changes vote weights (RRS sign) and removals change who is
evaluated later. A participant that survives in the replay after it was removed in the real run has no
further data; it is counted as "survives, censored" from that round on.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from load_all import load

ETH = 1e18
THRESH = 1 / 3  # min_collateral / punish_factor, in ETH
FEE = 0.5


def eq7(diffs: list[float]) -> list[float]:
    s = sum(diffs)
    return [1 / len(diffs)] * len(diffs) if s == 0 else [x / s for x in diffs]


def new_norm(vals: list[float], prev: float, metric: str) -> list[float]:
    """Copy of contribution.py:normalize_contribution_scores_new on main."""
    vals = [v - prev for v in vals]
    if metric == "loss":
        vals = [-v for v in vals]
    s, mx = sum(vals), max(vals)
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
            vals = [v + (v / sp) * -(s - 1) if v > 0 else v for v in vals]
    return vals


def scores_for(strategy: str, rule: str, ids: list[int], mad: dict, stored: dict) -> dict[int, float]:
    if strategy in ("naive", "accuracy_loss") or len(ids) <= 3:
        return {i: stored[i] for i in ids}  # these already follow the paper (Eq. 7 / 1/N / equal-share fallback)
    metric = "loss" if strategy == "loss_only" else "accuracy"
    vals = [mad[(i, metric)][0] for i in ids]
    prev = mad[(ids[0], metric)][1]
    if rule == "current":
        sc = new_norm(vals, prev, metric)
    else:
        sc = eq7([prev - v for v in vals] if metric == "loss" else [v - prev for v in vals])
    return dict(zip(ids, sc))


def replay(run: pd.Series, rounds: dict, rule: str) -> dict[int, tuple[str, int | None, float]]:
    grs = dict(rounds["start_grs"])
    alive, out = set(grs), {}
    reward_per_round = 1.0 / run.rounds
    for r in range(1, run.rounds + 1):
        rd = rounds.get(r)
        if rd is None:
            break
        for i in list(alive):
            if i not in rd["observed_active"]:  # removed in the real run, alive in replay: no more data
                out[i] = ("censored", r, grs[i])
                alive.discard(i)
        merged = [i for i in rd["merged"] if i in alive]
        if not merged:
            continue
        total = 0.0
        if r == 1:
            for i in alive:
                grs[i] -= FEE
        for i in alive - set(merged):  # RRS < 0
            if grs[i] > THRESH:
                p = grs[i] / 3
                grs[i] -= p
                total += p
            else:
                total += grs[i]
                out[i] = ("removed", r, grs[i])
        alive -= {i for i, (k, rr, _) in out.items() if k == "removed" and rr == r}
        if r == 1:
            for i in alive:
                if i in merged:
                    grs[i] += FEE
                else:
                    total += FEE
        votes = {i: (0 if i in rd["passive"] else len(rd["observed_active"]) - 1) for i in merged}
        sc = scores_for(run.strategy, rule, merged, rd["mad"], rd["stored"])
        R = reward_per_round + total
        w = {i: votes[i] * sc[i] for i in merged}
        pos = sum(x for x in w.values() if x > 0)
        if rule == "current":
            for i in merged:
                if sc[i] < 0:
                    p = grs[i] / 3 * abs(sc[i])
                    if grs[i] <= THRESH or p > grs[i]:
                        R += grs[i]
                        out[i] = ("removed", r, grs[i])
                    else:
                        grs[i] -= p
                        R += p
            alive -= {i for i, (k, rr, _) in out.items() if k == "removed" and rr == r}
            for i in merged:
                if i in alive and sc[i] >= 0 and pos > 0:
                    grs[i] += R * max(w[i], 0) / pos
        else:  # paper Eq. 11 + exclusion when GRS < min_collateral / punish_factor
            for i in merged:
                if pos > 0:
                    grs[i] += R * w[i] / pos
            for i in merged:
                if grs[i] < THRESH:
                    out[i] = ("removed", r, grs[i])
                    alive.discard(i)
    for i in alive:
        out.setdefault(i, ("survived", None, grs[i]))
    return out


def build_inputs(d: dict) -> dict:
    users, contrib, mad, pun = d["users"], d["contributions"], d["contributions_mad"], d["punishments"]
    inp = defaultdict(dict)
    for eid, g in users[users["round"] == 0].groupby("eid"):
        inp[eid]["start_grs"] = {int(u): grs / ETH for u, grs in zip(g.user_id, g.grs)}
    act = users[users.state == "active"]
    prev_active = act.assign(round=act["round"] + 1).groupby(["eid", "round"]).user_id.apply(set)
    for (eid, r), ids in prev_active.items():
        inp[eid].setdefault(r, {})["observed_active"] = ids
    for (eid, r), g in contrib.groupby(["eid", "round"]):
        rd = inp[eid].setdefault(r, {})
        rd["merged"] = sorted(g.user_id.astype(int))
        rd["stored"] = dict(zip(g.user_id.astype(int), g.contribution_score))
    for (eid, r), g in mad.groupby(["eid", "round"]):
        inp[eid].setdefault(r, {})["mad"] = {(int(u), m): (a, p) for u, m, a, p in
                                             zip(g.user_id, g.metric, g.user_mad_avg, g.prev_avg_round_val_after_mad)}
    passive = pun[pun.punishment_type == "passive"].groupby(["eid", "round"]).user_id.apply(set)
    for eid, rounds in inp.items():
        for r, rd in rounds.items():
            if isinstance(r, int):
                rd.setdefault("mad", {})
                rd.setdefault("merged", [])
                rd.setdefault("stored", {})
                rd.setdefault("observed_active", set())
                rd["passive"] = passive.get((eid, r), set())
    return inp


def main() -> None:
    d = load()
    runs = d["runs"]
    inp = build_inputs(d)
    roles = d["users"][d["users"]["round"] == 0].set_index(["eid", "user_id"]).role
    dq = d["users"][d["users"].state == "disqualified"].groupby(["eid", "user_id"])["round"].min()
    rows = []
    for _, run in runs.iterrows():
        for rule in ["current", "paper"]:
            for uid, (kind, rr, g_end) in replay(run, inp[run.eid], rule).items():
                rows.append(dict(eid=run.eid, dataset=run.dataset, strategy=run.strategy, noise=run.noise, act=run.act,
                                 rule=rule, user_id=uid, role=roles[(run.eid, uid)], outcome=kind, round=rr, grs_at=g_end,
                                 observed_removed=(run.eid, uid) in dq.index))
    res = pd.DataFrame(rows)
    cur = res[res.rule == "current"]
    agree = ((cur.outcome == "removed") == cur.observed_removed).mean()
    print(f"Validation: current-rule replay matches observed removal status for {agree:.1%} of participants")
    print(pd.crosstab([cur.observed_removed], cur.outcome))
    res["role"] = res.role.replace({"bad": "good (labelled malicious)"})
    for noise in [0.01, 0.1]:
        print(f"\n=== Free-rider outcomes, noise {noise} (removed / censored-survived / survived, of 30 per dataset) ===")
        fr = res[(res.role == "freerider") & (res.noise == noise)]
        print(fr.groupby(["dataset", "strategy", "rule"]).outcome.value_counts().unstack(fill_value=0).to_string())
    print("\n=== Honest participants (incl. inactive 'malicious') removed ===")
    h = res[res.role != "freerider"]
    print(h.groupby(["dataset", "strategy", "rule"]).outcome.value_counts().unstack(fill_value=0).to_string())
    print("\n=== GRS (ETH) at censoring round, paper rule ===")
    c = res[(res.rule == "paper") & (res.outcome == "censored")]
    print(c.groupby(["dataset", "role"]).grs_at.describe()[["count", "min", "50%", "max"]].round(3).to_string())
    res.to_csv("counterfactual_outcomes.csv", index=False)


if __name__ == "__main__":
    main()
