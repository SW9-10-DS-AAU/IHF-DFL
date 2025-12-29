import numpy as np
from parser.dataProcessors.graphs.kickedUsersExperiment import format_for_grouped_bar, prepare_data_for_graph
from parser.helpers.mehods import Method
from parser.parseExports import runProcessor
from parser.types.participant import MetaAttitude
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance
from parser.helpers.varianceCalculator import getVariances


roundkicked_by_method_noise = {}

def new_noise_bucket():
    return {}

def prepare_data_for_graph_method_noise(
    rounds,
    participants,
    experiment_specs,
    gasStats,
    outDir,
    freeRiderRound,
):
    if experiment_specs.freerider_start_round != freeRiderRound:
        return

    method = Method.from_string(
        experiment_specs.contribution_score_strategy,
        experiment_specs.use_outlier_detection,
    )
    noise = experiment_specs.freerider_noise_scale

    roundkicked_by_method_noise.setdefault(method, {})
    roundkicked_by_method_noise[method].setdefault(noise, [])

    for r in rounds:
        for p in r.disqualifiedUsers:
            if p.futureAttitude == MetaAttitude.FREERIDER:
                roundkicked_by_method_noise[method][noise].append(r.nr)

def get_round_kicked_method_noise():
    out = {}

    for method, noises in roundkicked_by_method_noise.items():
        out.setdefault(method, {})

        for noise, values in noises.items():
            if not values:
                continue
            variences = getVariances(values)
            out[method][noise] = getVariances(values)

    return out

def format_for_grouped_bar_method_noise(data):
    methods = list(data.keys())
    noises = sorted({n for m in data.values() for n in m.keys()})

    labels = [m.display_name for m in methods]
    group_names = [f"noise={n}" for n in noises]

    means = []
    variances = []
    missing = []

    for noise in noises:
        noise_means = []
        noise_vars = []
        noise_missing = []

        for method in methods:
            s = data.get(method, {}).get(noise)
            is_missing = s is None or s["NoValues"]
            noise_missing.append(is_missing)
            if s is None:
                noise_means.append(0)
                noise_vars.append([0, 0])
            else:
                avg = s["avg"]
                noise_means.append(avg)
                noise_vars.append([
                    max(0, s["low"]),
                    max(0, s["high"]),
                ])

        means.append(noise_means)
        variances.append(noise_vars)
        missing.append(noise_missing)

    return labels, means, variances, group_names, missing


def kickedGraphMethodNoise(freeriderRound: int, title: str, usePreviousTests: bool, RESULTDATAFOLDER):
    runProcessor(
        RESULTDATAFOLDER,
        usePreviousTests, 
        lambda rounds, participants, experimentConfig, gasCosts, outdir: \
            prepare_data_for_graph_method_noise(
                rounds,
                participants,
                experimentConfig,
                gasCosts,
                outdir,
                freeriderRound,
            )
    )

    labels, means, variances, group_names, missing = \
        format_for_grouped_bar_method_noise(
            get_round_kicked_method_noise()
        )

    grouped_bar_with_variance(
        labels,
        means,
        variances,
        group_names,
        missing,
        ylabel="Round Freerider Kicked",
        title=title,
    )