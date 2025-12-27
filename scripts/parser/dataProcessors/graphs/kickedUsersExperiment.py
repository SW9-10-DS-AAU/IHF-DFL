import jsonpickle
import numpy as np
from parser import *
from parser.experiment_specs import ExperimentSpec
from parser.helpers.mehods import Method
from parser.participant import MetaAttitude
from parser.parseExports import runProcessor
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance

def new_counter():
    return {
        MetaAttitude.GOOD: [],
        MetaAttitude.BAD: [],
        MetaAttitude.FREERIDER: [],
        MetaAttitude.BOTH: []
    }

roundkicked = {
    Method.ACCURACY: new_counter(),
    Method.DOTPRODUCT: new_counter(),
    Method.DOTPRODUCTANDOUTLIER: new_counter(),
    Method.NAIVE: new_counter()
}

def prepare_data_for_graph(rounds: list[Round], participants: dict[int, Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir, freeRiderRound: int):
  if experiment_specs.freerider_start_round != freeRiderRound:
     return
  
  global roundkicked
  disqualified_users: list[tuple[int, Participant]] = []

  bothKickedRound = None
  for round in rounds:
    for dusers in round.disqualifiedUsers:
        disqualified_users.append((round.nr, dusers))
        if(len(disqualified_users) > 1):
           bothKickedRound = round.nr

  for round_nr, participant in disqualified_users:
    roundkicked[Method.from_string(experiment_specs.contribution_score_strategy, experiment_specs.use_outlier_detection)]\
      [participant.futureAttitude].append(round_nr)
  
  if bothKickedRound:
     roundkicked[Method.from_string(experiment_specs.contribution_score_strategy, experiment_specs.use_outlier_detection)]\
      [MetaAttitude.BOTH].append(bothKickedRound)
  print(jsonpickle.dumps(roundkicked, indent=2))

  
def format_for_grouped_bar(data):
    methods = list(data.keys())
    attitudes = list(next(iter(data.values())).keys())

    labels = [m.name for m in methods]
    group_names = [a.name if hasattr(a, "name") else a for a in attitudes]

    means = []
    variances = []

    for att in attitudes:
      att_means = []
      att_vars = []

      for m in methods:
        s = data[m][att]
        avg = s["avg"]

        lower = max(0, avg - s["p25"])
        upper = max(0, s["p75"] - avg)

        att_means.append(avg)
        att_vars.append([lower, upper]) 

      means.append(att_means)
      variances.append(att_vars)

    return labels, means, variances, group_names
      

def get_round_kicked():
    out = {}
    #print(jsonpickle.dumps(roundkicked, indent=2))
    for method, userTypes in roundkicked.items():
        if not userTypes:
            continue

        out.setdefault(method, {})

        for userType, values in userTypes.items():
            if not values:
                continue

            out[method][userType] = {
                "p25": np.percentile(values, 25),
                "avg": np.mean(values),
                "p75": np.percentile(values, 75),
            }

    return out

def kickedGraph(freeriderRound: int, title: str, RESULTDATAFOLDER):
    runProcessor(RESULTDATAFOLDER, lambda rounds, participants, experimentConfig, gasCosts, outdir: \
                 prepare_data_for_graph(rounds, participants, experimentConfig, gasCosts, outdir, freeriderRound))


    labels, means, variances, group_names = format_for_grouped_bar(get_round_kicked())

    grouped_bar_with_variance(labels, means, variances, group_names, ylabel="Round Kicked", title=title)