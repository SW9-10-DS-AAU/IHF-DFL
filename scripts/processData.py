import os
from pathlib import Path
from typing import Optional

from matplotlib import cm, pyplot as plt
import numpy as np
from parser import *

RESULTDATAFOLDER = Path(__file__).resolve().parents[1].joinpath("experiment/data/experimentData")

print(RESULTDATAFOLDER)

def plot_from_parsed(
    rounds: list[Round],
    participants: dict[int, "Participant"],
    output_folder_path: str,
    filename: str = "simulation.pdf",
    title_prefix: Optional[str] = None,
) -> plt.Figure:
    # --- prepare rounds (sorted) ---
    rounds_sorted = sorted(rounds, key=lambda r: getattr(r, "nr", 0))
    if not rounds_sorted:
        raise ValueError("No rounds to plot.")

    # --- accuracy & loss arrays (use available fields) ---
    accuracy = []
    loss = []
    for r in rounds_sorted:
        # globalAcc/globalLoss fallback handling
        acc = getattr(r, "globalAcc", None)
        lossv = getattr(r, "globalLoss", None)
        # try alternatives if attributes missing
        if acc is None:
            acc = 0.0
        if lossv is None:
            lossv = 0.0
        accuracy.append(float(acc))
        loss.append(float(lossv))

    rounds_idx = list(range(len(accuracy)))

    # --- participants list (stable ordering) ---
    # We'll map participants to a stable index order by sorted id
    participants_list = [participants[k] for k in sorted(participants.keys())]

    # assign colors
    cmap = plt.colormaps["tab10"]
    colors = [cmap(i % 10) for i in range(len(participants_list))]

    # --- derive per-participant GRS series ---
    # Each Round.GRS is expected to be a list of values (one per participant, in same order)
    # We'll try to read GRS for each participant index; if mismatch, fill with np.nan
    grs_rounds = list(range(len(rounds_sorted)))
    num_rounds = len(rounds_sorted)

    # Determine expected number of participants from first round GRS if present
    first_grs_len = None
    for r in rounds_sorted:
        if getattr(r, "GRS", None):
            try:
                first_grs_len = len(r.GRS)
                break
            except Exception:
                first_grs_len = None
    # Build per-user GRS by index, mapping by participant index in participants_list
    per_user_grs: list[list[float]] = []
    for p_idx, p in enumerate(participants_list):
        series = []
        for r in rounds_sorted:
            grs = getattr(r, "GRS", None)
            if grs and isinstance(grs, (list, tuple)) and len(grs) > p_idx:
                try:
                    series.append(float(grs[p_idx]))
                except Exception:
                    series.append(np.nan)
            else:
                series.append(np.nan)  # missing
        per_user_grs.append(series)

    # --- compute reward and punishment series per round ---
    # Reward: prefer conctractBalanceRewards (if list sum or numeric), else sum of rewards
    reward_balance_per_round = []
    punish_sum_per_round = []
    for r in rounds_sorted:
        # reward balance
        cb = getattr(r, "conctractBalanceRewards", None)
        if isinstance(cb, (list, tuple)):
            try:
                reward_balance_per_round.append(sum(float(x) for x in cb))
            except Exception:
                reward_balance_per_round.append(0.0)
        else:
            # maybe single numeric
            try:
                reward_balance_per_round.append(float(cb) if cb is not None else 0.0)
            except Exception:
                # fallback to sum of r.rewards if available
                rr = getattr(r, "rewards", None)
                if isinstance(rr, (list, tuple)):
                    try:
                        reward_balance_per_round.append(sum(float(x) for x in rr))
                    except Exception:
                        reward_balance_per_round.append(0.0)
                else:
                    reward_balance_per_round.append(0.0)

        # punishments: try to sum the "amount" from each tuple in r.punishments
        pun_list = getattr(r, "punishments", None)
        if isinstance(pun_list, (list, tuple)):
            s = 0.0
            for tup in pun_list:
                # expected format: (address:str, amount:int, maybe_bigint, user_id:int)
                # try to take the 2nd element as amount, fallback to any numeric found
                if isinstance(tup, (list, tuple)):
                    if len(tup) >= 2:
                        try:
                            s += float(tup[1])
                        except Exception:
                            # scan for any numeric inside tuple
                            for v in tup:
                                if isinstance(v, (int, float)):
                                    s += float(v)
                                    break
                else:
                    # if it's a plain number
                    if isinstance(tup, (int, float)):
                        s += float(tup)
            punish_sum_per_round.append(s)
        else:
            punish_sum_per_round.append(0.0)

    # prepare plotting arrays
    # for nicer step look, replicate points as needed (original code had step option)
    x_reward = list(range(len(reward_balance_per_round)))
    # If you want step-like doubling for visual steps, duplicate:
    x_reward_step = [item for sublist in zip(x_reward, (np.array(x_reward)+1).tolist()) for item in sublist]
    x_reward_step[-1] = len(reward_balance_per_round) - 1

    y_reward_raw = reward_balance_per_round
    y_pun_raw = punish_sum_per_round

    # duplicate for step plotting as original code did
    if len(x_reward) > 0:
        y_reward = [item for sublist in zip(y_reward_raw, y_reward_raw) for item in sublist]
        y_reward_plus_pun = [item + pun for item, pun in zip(y_reward_raw, y_pun_raw)]
        y2_reward = [item for sublist in zip(y_reward_plus_pun, y_reward_plus_pun) for item in sublist]
    else:
        y_reward = []
        y2_reward = []

    # ------------------ Begin plotting ------------------
    f, axs = plt.subplots(
        1,
        4,
        figsize=(16, 3),
        gridspec_kw={"width_ratios": [0.8, 2, 2, 2], "height_ratios": [1]},
    )

    # color choices for main plots
    palette = ["#00629b", "#629b00", "#000000", "#d93e6a"]

    # participants combined: show all participants (including disqualified) in participants_list
    # plotting options
    use_step_grs = False

    # --- panel (b): accuracy & loss ---
    axs[1].clear()
    acc_line = axs[1].plot(rounds_idx, accuracy, color=palette[0], linewidth=2.5, label="Avg. Accuracy")[0]
    twin = axs[1].twinx()
    loss_line = twin.plot(rounds_idx, loss, color=palette[1], linewidth=2.5, linestyle="--", label="Avg. Loss")[0]

    # --- panel (c): GRS per participant ---
    axs[2].clear()
    grs_ticks = grs_rounds
    if use_step_grs:
        grs_x = [item for sublist in zip(grs_rounds, (np.array(grs_rounds) + 1).tolist()) for item in sublist]
        for p_idx, series in enumerate(per_user_grs):
            # create step y by duplicating values
            series_step = [item for sublist in zip(series, series) for item in sublist]
            axs[2].plot(grs_x, series_step, linewidth=2.5, color=colors[p_idx % len(colors)])
    else:
        for p_idx, series in enumerate(per_user_grs):
            # filter out nan for plotting continuity
            xs = [i for i, v in enumerate(series) if not (v is None or (isinstance(v, float) and np.isnan(v)))]
            ys = [series[i] for i in xs]
            if not xs:
                continue
            axs[2].plot(xs, ys, linewidth=2.0, color=colors[p_idx % len(colors)], alpha=0.9, marker="o", markersize=4)

    # --- panel (d): rewards & punishments ---
    axs[3].clear()
    if x_reward_step:
        axs[3].plot(x_reward_step, y_reward, label="reward", color=palette[0], linewidth=2.5)
        axs[3].plot(x_reward_step, y2_reward, label="reward + punishments", color=palette[1], linewidth=2.5)
        axs[3].fill_between(x_reward_step, y_reward, y2_reward, alpha=0.2, hatch=r"//", color=palette[1])

    # --- panel (a): meta text ---
    axs[0].clear()
    # attempt to extract some meta info from data
    dataset_name = getattr(rounds_sorted[0], "dataset", None) or title_prefix or "dataset"
    rounds_count = len(rounds_sorted)
    print(rounds_count)
    users_count = len(participants_list)
    # malicious / freerider counts: try to infer from participant states attitutes
    malicious = 0
    freerider = 0
    try:
        for p in participants_list:
            # check most recent state if available
            if getattr(p, "states", None):
                last_state = p.states[-1]
                att = getattr(last_state, "attitude", None)
                if att is not None and getattr(att, "name", "").upper() == "BAD":
                    malicious += 1
                if att is not None and getattr(att, "name", "").upper() == "FREERIDER":
                    freerider += 1
    except Exception:
        pass

    info_text = (
        f"dataset={dataset_name}\n"
        f"rounds={rounds_count}\n"
        f"users={users_count}\n"
        f"malicious={malicious}\n"
        f"copycat={freerider}"
    )
    axs[0].text(0, 0.1, info_text, fontsize=12)
    axs[0].set_axis_off()

    # labels, grids, ticks
    axs[1].set_xlabel("rounds\n(a)", fontsize=12)
    axs[2].set_xlabel("rounds\n(b)", fontsize=12)
    axs[3].set_xlabel("rounds\n(c)", fontsize=12)

    axs[1].set_ylabel("Avg. Accuracy", fontsize=12)
    twin.set_ylabel("Avg. Loss", fontsize=12)
    axs[2].set_ylabel("GRS", fontsize=12)
    axs[3].set_ylabel("Contract Balance", fontsize=12)

    axs[1].tick_params(axis="both", which="major", labelsize=12)
    axs[2].tick_params(axis="both", which="major", labelsize=12)
    axs[3].tick_params(axis="both", which="major", labelsize=12)

    # xticks
    if len(rounds_idx) > 20:
        axs[1].set_xticks([i for i in rounds_idx if i % 5 == 0 or i == 0])
    else:
        axs[1].set_xticks(rounds_idx)

    if len(grs_ticks) > 20:
        axs[2].set_xticks([i for i in grs_ticks if i % 5 == 0 or i == 0])
    else:
        axs[2].set_xticks(grs_ticks)

    if len(x_reward_step) > 20:
        axs[3].set_xticks([i for i in x_reward_step if i % 5 == 0 or i == 0])
    else:
        axs[3].set_xticks(x_reward_step)

    if rounds_idx:
        axs[1].set_xlim(0, max(rounds_idx))

    axs[2].yaxis.get_offset_text().set_fontsize(12)
    axs[3].yaxis.get_offset_text().set_fontsize(12)

    axs[1].grid()
    axs[2].grid()
    axs[3].grid()

    # Legend for accuracy/loss
    twin_lines = [acc_line, loss_line]
    axs[1].legend(twin_lines, [l.get_label() for l in twin_lines], loc="lower right", fontsize=9)

    if x_reward_step:
        lgnd = axs[3].legend(fontsize=9)

    plt.tight_layout(pad=1)

    os.makedirs(output_folder_path, exist_ok=True)
    savepath = os.path.join(output_folder_path, filename)
    plt.savefig(savepath, bbox_inches="tight")

    return plt

def processData(rounds: list[Round], participants: list[Participant], gasStats: GasStats, outDir):
  #print(rounds)
  #print(participants)
  plot_from_parsed(rounds, participants, outDir, "plot.pdf", "TestPlot")

runProcessor(RESULTDATAFOLDER, processData)