from parser import *

invalidExperiments = []

def save_round_if_invalid(rounds: list[Round], participants: list[Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
  lastround = []
  for i, round in enumerate(rounds):
    if (i == 0):
      lastround = round.GRS
      continue
    for i, userGRS in enumerate(round.GRS):
      if userGRS == lastround[i]:
        invalidExperiments.append(outDir)
        return

def get_invalid_experiments():
  return invalidExperiments