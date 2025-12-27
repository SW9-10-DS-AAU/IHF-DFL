import numpy as np
from openfl.ml.pytorch_model import Participant
from parser.helpers.mehods import Method
from parser.participant import Attitude, MetaAttitude
from parser.dataProcessors.plotter import line_graph
from parser.parseExports import runProcessor
from parser.experiment_specs import ExperimentSpec
from parser.gasCosts import GasStats
from parser.round import Round


grs_by_method_round = {}

def get_target_user_ids(participants, target_attitude):
    return {
        pid for pid, p in participants.items()
        if p.futureAttitude == target_attitude
    }

def prepare_grs_by_round(
    rounds: list[Round], 
    participants: dict[int, Participant], 
    experiment_specs: ExperimentSpec, 
    gasStats: GasStats, 
    outDir, 
    target_attitude: Attitude,  # FREERIDER or BAD only
):
    method = Method.from_string(
        experiment_specs.contribution_score_strategy,
        experiment_specs.use_outlier_detection,
    )

    target_ids = get_target_user_ids(participants, target_attitude)
    if not target_ids:
        return

    grs_by_method_round.setdefault(method, {})

    for r in rounds:
        vals = [r.GRS[pid] for pid in target_ids]

        grs_by_method_round[method].setdefault(r.nr, []).extend(vals)

def get_grs_lines():
    out = {}

    for method, rounds in grs_by_method_round.items():
        out[method] = {
            r: np.mean(vals) for r, vals in rounds.items()
        }

    return out

def grsGraph(
    target_attitude: Attitude,
    title: str,
    freeridingRoundStart: int,
    RESULTDATAFOLDER: str
):
    grs_by_method_round.clear()

    runProcessor(
        RESULTDATAFOLDER,
        lambda rounds, participants, experimentConfig, gasCosts, outdir: \
            prepare_grs_by_round(
                rounds,
                participants,
                experimentConfig,
                gasCosts,
                outdir,
                target_attitude,
            )
    )

    data = {
        m.name: v for m, v in get_grs_lines().items()
    }

    line_graph(
        data,
        x_label="Round",
        y_label="GRS",
        title=title,
        vline=freeridingRoundStart,
    )