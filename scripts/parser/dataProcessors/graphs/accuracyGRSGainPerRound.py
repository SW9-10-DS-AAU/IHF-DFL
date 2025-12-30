import numpy as np
from parser.helpers.mehods import Method
from parser.types.participant import Attitude
from parser.parseExports import runProcessor
from parser.dataProcessors.plotter import line_graph

# gains[user_type][round] = [values...]
grs_gain = {}

def prepare_accuracy_grs_gain(
    rounds,
    participants,
    experiment_specs,
    gasStats,
    outDir,
):
    # Only accuracy scoring
    if experiment_specs.contribution_score_strategy != "accuracy":
        return

    method = Method.ACCURACY

    # Precompute participant groups
    groups = {
        Attitude.GOOD: set(),
        Attitude.FREERIDER: set(),
        Attitude.MALICIOUS: set(),
    }

    for pid, p in participants.items():
        if p.futureAttitude in groups:
            groups[p.futureAttitude].add(pid)

    # Track previous round GRS
    prev_grs = {}

    for r in rounds:
        # round_gains = []
        for att, ids in groups.items():
            if not ids:
                continue

            gains = []
            for pid in ids:
                curr = r.GRS[pid]
                prev = prev_grs.get(pid)

                # if prev is not None:
                #     gains.append(curr - prev)
                if prev is not None:
                    delta = curr - prev
                    gains.append(delta)
                    # round_gains.append(delta)

                prev_grs[pid] = curr

            if gains:
                grs_gain.setdefault(att, {}).setdefault(r.nr, []).extend(gains)
        # if round_gains:
        #     volume = sum(abs(g) for g in round_gains)
        #     grs_gain.setdefault("VOLUME", {}).setdefault(r.nr, []).append(volume)


def accuracyGRSGainGraph(
    title: str,
    usePreviousTests: bool,
    windowAndFileName: str,
    RESULTDATAFOLDER,
):
    grs_gain.clear()

    runProcessor(
        RESULTDATAFOLDER,
        usePreviousTests,
        lambda rounds, participants, experimentConfig, gasCosts, outdir:
            prepare_accuracy_grs_gain(
                rounds,
                participants,
                experimentConfig,
                gasCosts,
                outdir,
            )
    )

    # Average per round
    # data = {
    #     att.name: {r: np.mean(vals) for r, vals in rounds.items()}
    #     for att, rounds in grs_gain.items()
    # }
    data = {}

    for key, rounds in grs_gain.items():
        label = key
        data[label] = {r: np.mean(vals) for r, vals in rounds.items()}

    line_graph(
        data,
        False,
        x_label="Round",
        y_label="Average GRS Gained",
        title=title,
        windowAndFileName=windowAndFileName
    )
