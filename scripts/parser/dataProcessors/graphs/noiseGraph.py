import numpy as np
from parser.dataProcessors.graphs.kickedUsersExperiment import format_for_grouped_bar, prepare_data_for_graph
from parser.helpers.mehods import Method
from parser.parseExports import runProcessor
from parser.participant import MetaAttitude
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance


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

            out[method][noise] = {
                "avg": np.mean(values),
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75),
            }

    return out

def format_for_grouped_bar_method_noise(data):
    methods = list(data.keys())
    noises = sorted({n for m in data.values() for n in m.keys()})

    labels = [m.name for m in methods]
    group_names = [f"noise={n}" for n in noises]

    means = []
    variances = []

    for noise in noises:
        noise_means = []
        noise_vars = []

        for method in methods:
            s = data.get(method, {}).get(noise)
            if s is None:
                noise_means.append(0)
                noise_vars.append([0, 0])
            else:
                avg = s["avg"]
                noise_means.append(avg)
                noise_vars.append([
                    max(0, avg - s["p25"]),
                    max(0, s["p75"] - avg),
                ])

        means.append(noise_means)
        variances.append(noise_vars)

    return labels, means, variances, group_names


def kickedGraphMethodNoise(freeriderRound: int, title: str, RESULTDATAFOLDER):
    runProcessor(
        RESULTDATAFOLDER,
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

    labels, means, variances, group_names = \
        format_for_grouped_bar_method_noise(
            get_round_kicked_method_noise()
        )

    grouped_bar_with_variance(
        labels,
        means,
        variances,
        group_names,
        ylabel="Round Freerider Kicked",
        title=title,
    )