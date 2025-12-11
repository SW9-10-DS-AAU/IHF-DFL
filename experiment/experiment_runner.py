import os
import time
from pathlib import Path
from openfl.ml import pytorch_model as PM
from openfl.contracts import fl_manager as Manager, fl_challenge as Challenge
from openfl.utils import require_env_var
from types import SimpleNamespace
from web3 import Web3, Account

from openfl.utils.async_writer import AsyncWriter


def run_experiment(dataset_name, experiment_config, writer: AsyncWriter=None):

  experiment_start = time.perf_counter()
  RPC_ENDPOINT = require_env_var("RPC_URL")
    
# Only for the real-net simulation
# In order to use a non-locally forked blockchain, 
# private keys are required to unlock accounts
  if experiment_config.fork == False:
    w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINT))

    raw_keys = require_env_var("PRIVATE_KEYS")
    privKeys = [k.strip() for k in raw_keys.splitlines() if k.strip()]

    # Convert to Web3 Account objects
    loaded_accounts = [Account.from_key(k) for k in privKeys]

    # Wrap for compatibility with older code expecting `.privateKey`
    PRIVKEYS = [
        SimpleNamespace(privateKey=acc._private_key, address=acc.address)
        for acc in loaded_accounts
    ]

    print(f"Loaded {len(PRIVKEYS)} private keys.")
  else:
      PRIVKEYS = None

  pytorch_model = PM.PytorchModel(dataset_name, 
                              experiment_config.number_of_good_contributors, 
                              experiment_config.number_of_contributors, 
                              experiment_config.epochs, 
                              experiment_config.batch_size, 
                              experiment_config.standard_buy_in,
                              experiment_config.max_buy_in,
                              experiment_config.freerider_noise_scale,
                              experiment_config.freerider_start_round)

  for i in range(experiment_config.number_of_bad_contributors):
      pytorch_model.add_participant("bad",3)

  for i in range(experiment_config.number_of_freerider_contributors):
      pytorch_model.add_participant("freerider",1)
      
  for i in range(experiment_config.number_of_inactive_contributors):
      pytorch_model.add_participant("inactive",1)

  manager = Manager.FLManager(pytorch_model, True).init(experiment_config.number_of_good_contributors, 
                                              experiment_config.number_of_bad_contributors,
                                              experiment_config.number_of_freerider_contributors,
                                              experiment_config.number_of_inactive_contributors,
                                              experiment_config.minimum_rounds,
                                              RPC_ENDPOINT,
                                              experiment_config.fork,
                                              PRIVKEYS)
  manager.build_contract()

  configs = manager.deploy_challenge_contract(experiment_config.min_buy_in,
                                          experiment_config.max_buy_in,
                                          experiment_config.reward, 
                                          experiment_config.minimum_rounds,
                                          experiment_config.punish_factor,
                                          experiment_config.first_round_fee)

  extra_configs = {}
  if experiment_config.contribution_score_strategy is not None:
      extra_configs["contribution_score_strategy"] = (
          experiment_config.contribution_score_strategy
      )

  model = Challenge.FLChallenge(manager, 
                      configs,
                      pytorch_model,
                      experiment_config,
                      writer)


  model.simulate(rounds=experiment_config.minimum_rounds)
  experiment_end = time.perf_counter()
  total_experiment_time = experiment_end - experiment_start

  print("\n" + "="*75)
  print(f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  writer.writeComment(f"TOTAL EXPERIMENT TIME: {total_experiment_time:.2f} seconds")
  print("="*75 + "\n")

  return Experiment(model, manager)


def visualizeModel(model):
  model.visualize_simulation("figures")



def print_transactions(experiment):
  model = experiment.model
  print("{:<10} - {:^64} -    Gas Used - {}".format("Function", "Transaction Hash", "Success"))
  print("------------------------------------------------------------------------------------------")
  for f, txhash in model.txHashes:
      r = model.w3.eth.wait_for_transaction_receipt(txhash)
      if r["status"] == 1:
          success = "✅"
      else:
          success = "FAIL"
      
      gas = r["gasUsed"]
      print("{:<10} - {} - {:>9,.0f} -   {}".format(f, txhash, gas, success))


def print_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  print("\\renewcommand{\\arraystretch}{1.3}")
  print("\\begin{center}")
  print("\\begin{tabular}{ c|c }")

  print("Contract & Address (Ropsten Testnet) \\\\")
  print("\\hline")
  print("Ma-1 & {} \\ ".format(manager.manager.address))
  print("Ch-1 & {} \\ ".format(model.model.address))
  for i, p in enumerate(model.pytorch_model.participants[:-1] + \
                            model.pytorch_model.disqualified + \
                            [model.pytorch_model.participants[-1]]):
      print("P-{}  & {} \\ ".format(i+1, p.address))

  print("\\end{tabular}")
  print("\\end{center}")


def table_with_gas_and_transactions_latex(experiment):
  model = experiment.model
  manager = experiment.manager
  reg = model.gas_register, "register"
  fed = model.gas_feedback, "feedback"
  clo = model.gas_close, "settle round"
  slo = model.gas_slot, "reserve slot"
  wei = model.gas_weights, "provide weights**"
  dep = manager.gas_deploy, "deployment"
  dep = manager.gas_deploy, "deployment"
  ext = model.gas_exit, "exit"

  tot  = 0
  tot2 = 0

  print("\\begin{tabular}{ |c|c|c| }\n\\hline\nFunction & Gas Amount & Gas Costs*\\\\ \n\\hline")
  for i, f in [reg,slo,wei,fed,clo]:
      print("{} & {:,.0f} & {:.5f} ETH \\\\".format(f, sum(i)/len(i), sum(i)/len(i) * 20e9 / 1e18 ))
      tot += sum(i)/len(i)
      if i != clo[0]:
              tot2 += sum(i)/len(i)
          
  print("\\hline\n\\hline")
  print("complete round & {:,.0f} & {:.5f} \\ ".format(tot, tot * 20e9 / 1e18))
  print("\\hline\n\\end{tabular}")
    

class Experiment:
  def __init__(self, model, manager):
    self.model = model
    self.manager = manager