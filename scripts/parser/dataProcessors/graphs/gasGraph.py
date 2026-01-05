from collections import defaultdict

from parser.parseExports import runProcessor
from parser.helpers.mehods import Method
from parser.helpers.varianceCalculator import getVariances
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance
from parser.types.round import GasType, Round
from parser.experiment_specs import ExperimentSpec
from parser.gasCosts import GasStats
from parser.types.participant import Participant
from parser.helpers.setLegendLocation import LegendPosition

gas_by_method = defaultdict(list)

def prepare_data_for_graph_gas_methods(
    rounds: list[Round], 
    participants: dict[int, Participant], 
    experiment_specs: ExperimentSpec, 
    gasStats: GasStats, 
    outDir: str, 
):
    method = Method.from_string(
        experiment_specs.contribution_score_strategy,
        experiment_specs.use_outlier_detection,
    )

    for r in rounds:
        gas = {
            t: sum(g.amount for g in r.gasTransactions if g.type == t)
            for t in GasType
        }

        contrib_plus_feedback = (
            gas[GasType.contrib] + gas[GasType.feedback]
        )

        other_gas = sum(
            v for t, v in gas.items()
            if t not in (GasType.contrib, GasType.feedback)
        )

        total_gas = sum(gas.values())

        gas_by_method[method].append({
            "contrib_feedback": contrib_plus_feedback,
            "other": other_gas,
            "total": total_gas,
        })

        assert abs(total_gas - (contrib_plus_feedback + other_gas)) == 0

def get_gas_by_method():
    out = {}

    for method, rounds in gas_by_method.items():
        if not rounds:
            continue

        out[method] = {}

        for key in ("contrib_feedback", "other", "total"):
            vals = [r[key] for r in rounds]

            if all(v == 0 for v in vals):
                continue

            out[method][key] = getVariances(vals)

    return out

def format_for_grouped_bar_gas_methods(data):
    methods = list(data.keys())
    groups = ["contrib_feedback", "other", "total"]

    labels = [m.display_name for m in methods]
    group_names = [
        "Contrib + Feedback",
        "Other Gas Costs",
        "Total Gas",
    ]

    means = []
    variances = []
    missing = []

    for group in groups:
        group_means = []
        group_vars = []
        group_missing = []

        for method in methods:
            s = data.get(method, {}).get(group)
            is_missing = s is None or s.get("NoValues", False)
            group_missing.append(is_missing)

            if s is None:
                group_means.append(0)
                group_vars.append([0, 0])
            else:
                avg = s["avg"]
                group_means.append(avg)
                group_vars.append([
                    max(0, s["low"]),
                    max(0, s["high"]),
                ])

        means.append(group_means)
        variances.append(group_vars)
        missing.append(group_missing)

    return labels, means, variances, group_names, missing

def gasCostGraphMethods(title: str, usePreviousTests: bool, windowAndFileName: str, legend_position: LegendPosition,RESULTDATAFOLDER):
    runProcessor(
        RESULTDATAFOLDER,
        usePreviousTests,
        prepare_data_for_graph_gas_methods
    )

    labels, means, variances, group_names, missing = \
        format_for_grouped_bar_gas_methods(
            get_gas_by_method()
        )

    grouped_bar_with_variance(
        labels,
        means,
        variances,
        group_names,
        missing,
        windowAndFileName,
        legend_position,
        ylabel="Gas used per round (WEI)",
        title=title,
    )