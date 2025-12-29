import jsonpickle
import numpy as np
from parser import *
from parser.experiment_specs import ExperimentSpec
from parser.helpers.mehods import Method
from parser.types.participant import MetaAttitude
from parser.parseExports import runProcessor
from parser.plotters.groupedBarWithVariance import grouped_bar_with_variance
from parser.helpers.varianceCalculator import getVariances

nrOfRounds = 0

def new_counter():
    return {
        MetaAttitude.GOOD: [],
        MetaAttitude.MALICIOUS: [],
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
  global nrOfRounds
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
    
  nrOfRounds = len(rounds)

  
def format_for_grouped_bar(data):
    global nrOfRounds
    methods = list(data.keys())
    attitudes = list(next(iter(data.values())).keys())

    labels = [m.display_name for m in methods]
    group_names = [a.display_name for a in attitudes]

    means = []
    variances = []
    missing = []

    for att in attitudes:
      att_means = []
      att_vars = []
      attitude_missing = []

      for m in methods:
        s = data[m].get(att)
        is_missing = s is None or s["NoValues"]
        attitude_missing.append(is_missing)
        if s is None:
            att_means.append(nrOfRounds)
            att_vars.append([nrOfRounds, 0])
            continue
        
        avg = s["avg"]
        
        upper = s["high"]
        lower = s["low"]

        att_means.append(avg)
        att_vars.append([lower, upper]) 

      means.append(att_means)
      variances.append(att_vars)
      missing.append(attitude_missing)

    return labels, means, variances, group_names, missing
      

def get_round_kicked():
    out = {}
    #print(jsonpickle.dumps(roundkicked, indent=2))
    for method, userTypes in roundkicked.items():
        if not userTypes:
            continue

        out.setdefault(method, {})

        for userType, values in userTypes.items():
            if not values or all(v == 0 for v in values):
                continue

            out[method][userType] = getVariances(values)

    return out

def kickedGraph(freeriderRound: int, title: str, useSameTests: bool, RESULTDATAFOLDER):
    runProcessor(RESULTDATAFOLDER, useSameTests, lambda rounds, participants, experimentConfig, gasCosts, outdir: \
                 prepare_data_for_graph(rounds, participants, experimentConfig, gasCosts, outdir, freeriderRound))


    labels, means, variances, group_names, missing = format_for_grouped_bar(get_round_kicked())

    grouped_bar_with_variance(labels, means, variances, group_names, missing, ylabel="Round Kicked", title=title)