from parser.dataProcessors.gasCostExtractor import print_gas, get_totals
from parser.helpers.mehods import Method
from parser.parseExports import runProcessor
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance
import numpy as np

gas_by_method = {}

def gasCostGraph(title, RESULTDATAFOLDER):
    gas_by_method.clear()

    runProcessor(
        RESULTDATAFOLDER,
        lambda rounds, participants, experimentConfig, gasCosts, outdir: (
            print_gas(rounds, participants, experimentConfig, gasCosts, outdir),
            gas_by_method.setdefault(
                Method.from_string(
                    experimentConfig.contribution_score_strategy,
                    experimentConfig.use_outlier_detection,
                ),
                []
            ).append(get_totals()[-1])
        )
    )

    labels = [m.name for m in gas_by_method]
    means = [[np.mean(v) for v in gas_by_method.values()]]
    variances = [[
        [np.mean(v) - np.percentile(v, 25), np.percentile(v, 75) - np.mean(v)]
        for v in gas_by_method.values()
    ]]

    grouped_bar_with_variance(
        labels,
        means,
        variances,
        ["Total Gas Used"],
        ylabel="Total Gas Used",
        title=title,
    )
