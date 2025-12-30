from parser import *
totals = []

def print_gas(rounds: list[Round], participants: dict[int, Participant], experiment_specs: ExperimentSpec, gasStats: GasStats, outDir):
    if gasStats is None:
        return
    total = sum(gasStats.gas_close) + sum(gasStats.gas_deploy) + sum(gasStats.gas_exit) + sum(gasStats.gas_feedback) + sum(gasStats.gas_register) + sum(gasStats.gas_slot) + sum(gasStats.gas_weights) + sum(gasStats.gas_close)
    print(f"gasclose {sum(gasStats.gas_close)}")
    print(f"gasdeploy {sum(gasStats.gas_deploy)}")
    print(f"gasexit {sum(gasStats.gas_exit)}")
    print(f"gasfeedback {sum(gasStats.gas_feedback)}")
    print(f"gasregister {sum(gasStats.gas_register)}")
    print(f"gasslot {sum(gasStats.gas_slot)}")
    print(f"gasweights {sum(gasStats.gas_weights)}")
    print(f"gasclose {sum(gasStats.gas_close)}")
    print(f"Total {total}")

    print(type(rounds[0].gasTransactions[0].type))
    print(sum([transaction for transaction in rounds[1].gasTransactions if transaction.type == GasType.contrib]))

    totals.append(total)

def get_totals():
  return totals