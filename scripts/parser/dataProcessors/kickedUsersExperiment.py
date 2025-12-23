import numpy as np
from sympy import Range
from parser import *
from parser.experiment_specs import ExperimentSpec
from parser.helpers.mehods import Method

roundkicked = {
  Method.ACCURACY: [],
  Method.DOTPRODUCT: [],
  Method.DOTPRODUCTANDOUTLIER: [],
  Method.NAIVE: []
}

def create_kicked_graph(rounds: list[Round], participants: list[Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
  for i, round in enumerate(rounds):
    if(i == 0): continue
    usersKicked = len(round.disqualifiedUsers)
    #print(round.disqualifiedUsers)
    for j in Range(1, usersKicked): 
      #print(f"round.nr {round.nr}, {experiment_specs.freerider_start_round}, {i}")
      roundkicked[Method.from_string(experiment_specs.contribution_score_strategy, experiment_specs.use_outlier_detection)] \
        .append(round.nr - experiment_specs.freerider_start_round)
      

def get_round_kicked():
  out = {}
  
  for method in roundkicked:
    print(method.value)
    out[method.name] = {
      "p25": np.percentile(method.value, 25),
      "avg": np.mean(method.value),
      "p75": np.percentile(method.value, 75)
    }

  return out

def _get_percentile():
  for method in roundkicked:
    p25 = np.percentile(method, 25)
    avg = np.mean(method)
    p75 = np.percentile(method, 75)
  