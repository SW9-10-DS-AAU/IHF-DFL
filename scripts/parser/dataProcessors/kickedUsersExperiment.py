import numpy as np
from sympy import Range
from parser import *
from parser.experiment_specs import ExperimentSpec
from parser.helpers.mehods import Method

roundkicked = {
  Method.ACCURACY: {},
  Method.DOTPRODUCT: {},
  Method.DOTPRODUCTANDOUTLIER: {},
  Method.NAIVE: {}
}

def create_kicked_graph(rounds: list[Round], participants: list[Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
  
  for i, round in enumerate(rounds):
    if(i == 0): continue
    #print(round.disqualifiedUsers)
    for disqualified in round.disqualifiedUsers: 
      #print(f"round.nr {round.nr}, {experiment_specs.freerider_start_round}, {i}") 
      roundkicked[Method.from_string(experiment_specs.contribution_score_strategy, experiment_specs.use_outlier_detection)] \
        .append()
      

def get_round_kicked():
    out = {}
    for i, (method, values) in enumerate(roundkicked.items()):
        if not values:
           continue
        out[method.name] = {
            "p25": np.percentile(values, 25),
            "avg": np.mean(values),
            "p75": np.percentile(values, 75)
        }
    return out


def _get_percentile():
  for method in roundkicked:
    p25 = np.percentile(method, 25)
    avg = np.mean(method)
    p75 = np.percentile(method, 75)
  