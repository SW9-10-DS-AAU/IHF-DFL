import numpy as np
from parser.helpers.mehods import Method
from parser.types.participant import Attitude
from parser.parseExports import runProcessor
from parser.dataProcessors.plotter import line_graph

# grs[method][round] = [values...]
grs = {}

def prepare_dotproduct_grs(
    rounds,
    participants,
    experiment_specs,
    gasStats,
    outDir,
    target_attitude,
):
    # Only dot product
    if experiment_specs.contribution_score_strategy != "dotproduct":
        return

    # Only activation round 3
    if experiment_specs.freerider_start_round != 3:
        return

    method = Method.from_string(
        experiment_specs.contribution_score_strategy,
        experiment_specs.use_outlier_detection,
    )

    # Filter users by attitude
    target_ids = {
        pid for pid, p in participants.items()
        if p.futureAttitude == target_attitude
    }
    if not target_ids:
        return

    grs.setdefault(method, {})

    for r in rounds:
        vals = [r.GRS[pid] for pid in target_ids]
        grs[method].setdefault(r.nr, []).extend(vals)


def dotProductOutlierGRSGraph(
    target_attitude: Attitude,
    title: str,
    usePreviousTests: bool, 
    windowAndFileName: str,
    RESULTDATAFOLDER,
    useShortName: bool = False
):
    grs.clear()

    runProcessor(
        RESULTDATAFOLDER,
        usePreviousTests,
        lambda rounds, participants, experimentConfig, gasCosts, outdir:
            prepare_dotproduct_grs(
                rounds,
                participants,
                experimentConfig,
                gasCosts,
                outdir,
                target_attitude,
            )
    )

    # Average per round
    data = {
        m: {r: np.mean(vals) for r, vals in rounds.items()}
        for m, rounds in grs.items()
    }

    line_graph(
        data,
        useShortName,
        x_label="Round",
        y_label="Average GRS",
        title=title,
        vline=3,
        windowAndFileName=windowAndFileName
    )
