import asyncio
import datetime
import os
import time
import warnings

import torch
import numpy as np
from types import SimpleNamespace
from decimal import Decimal
from collections.abc import Mapping
from eth_abi import encode
from web3 import Web3
from termcolor import colored
import matplotlib.pyplot as plt
from web3.exceptions import ContractLogicError
from openfl.contracts import FLManager
from openfl.ml.pytorch_model import gb, rb, b, green, red
from openfl.utils import printer, config
from openfl.api.connection_helper import ConnectionHelper
import openfl.utils.config

# Smart-contract–backed federated learning simulation.
# Handles:
#   - User registration / exit on-chain
#   - Hashed model submission & slot reservation
#   - Feedback exchange (reputation updates)
#   - Contribution score calculation (dot-product & MAD-based)
#   - Round settlement and visualization


class FLChallenge(FLManager):
    def __init__(self, manager, configs, pyTorchModel):
        self.manager = manager
        self.w3 = manager.w3

        # Allow configs to optionally include an extra mapping/namespace with
        # strategy overrides without breaking the legacy tuple signature.
        configs = list(configs)
        extra_config = {}
        if configs and isinstance(configs[-1], (Mapping, SimpleNamespace)):
            # Last element is an optional dict-like config extension
            candidate = configs.pop()
            if isinstance(candidate, Mapping):
                extra_config = dict(candidate)
            else:
                extra_config = dict(vars(candidate))

        self.model, self.modelAddress = configs[:2]
        self.pytorch_model = pyTorchModel
        self.MIN_BUY_IN, self.MAX_BUY_IN , self.REWARD, self.MIN_ROUNDS, = configs[2:-2]
        self.PUNISHMENT_FACTOR = configs[-2]
        self.FREERIDER_FACTOR  = configs[-1]
        self.fork = manager.fork
        
        self.gas_feedback = [] 
        self.gas_register = [] 
        self.gas_slot     = [] 
        self.gas_weights  = [] 
        self.gas_close    = [] 
        self.gas_deploy   = [] 
        self.gas_exit     = []
        self.txHashes     = []

        self._reward_balance = [self.REWARD]
        self._punishments = []
        self.config = config.get_contracts_config()

        self._extra_contract_config = extra_config
        self._contribution_score_strategy = self._determine_contribution_score_strategy()
        self._contribution_score_calculators = {
            "legacy": self._calculate_scores_legacy,
            "mad": self._calculate_scores_mad,
            "naive": self._calculate_scores_naive,
            "accuracy": self._calculate_scores_accuracy
        }

    def _determine_contribution_score_strategy(self):
        # Resolve contribution-score strategy from config, default to mad
        default_strategy = "mad"
        strategy = self._extra_contract_config.get("contribution_score_strategy")

        if strategy is None:
            strategy = default_strategy

        return str(strategy).strip().lower()

    def _get_contribution_score_calculator(self):
        """
        Return the function used for contribution-score calculation,
        based on the configured strategy.
        """

        strategy = self._contribution_score_strategy
        if strategy not in self._contribution_score_calculators:
            available = ", ".join(sorted(self._contribution_score_calculators))
            raise ValueError(
                f"Unknown contribution score strategy '{strategy}'. Available strategies: {available}"
            )
        print("strategy: ", strategy)
        return self._contribution_score_calculators[strategy]
        
    def register_all_users(self):
        """
        Register all participants in the federated learning model
        via the smart contract.
        """
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.isRegistered:
                continue
            if self.fork:
                # Simple tx builder for forked (dev) chain
                tx = super().build_tx(acc.address, self.modelAddress, acc.collateral)
                txHash = self.model.functions.register().transact(tx)
            else:
                # Non-fork: build and sign a raw transaction manually
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                reg = super().build_non_fork_tx(acc.address, nonce, value=acc.collateral)
                reg = self.model.functions.register().build_transaction(reg)
                signed = self.w3.eth.account.sign_transaction(reg, private_key=acc.privateKey)
                txHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            bal = self.w3.eth.get_balance(self.w3.eth.default_account)
            acc.isRegistered = True
            print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                                                           acc.address[0:16] + "...", 
                                                           txHash.hex()[0:6] + "...", 
                                                           acc.collateral
                                                           ))
        
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_register.append(receipt["gasUsed"])
            self.txHashes.append(("register",receipt["transactionHash"].hex()))
        printer._print("-----------------------------------------------------------------------------------", "\n")
        
    
    def get_hashed_weights_of(self, user):
        return self.model.functions.weightsOf(user.address,self.pytorch_model.round-1).call({"to": self.modelAddress})
    
    
    def get_global_reputation_of_user(self, user):
        return self.model.functions.GlobalReputationOf(user).call({"to": self.modelAddress})
        
    
    def get_round_reputation_of_user(self, user):
        return self.model.functions.RoundReputationOf(user).call({"to": self.modelAddress})
    
    
    def get_reward_left(self):
        return self.model.functions.rewardLeft().call({"to": self.modelAddress})

    
    def users_provide_hashed_weights(self):

        txs = []
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Account inactive:", 
                                                                         acc.address[0:16] + "...", 
                                                                         "   ...   ",
                                                                         self.get_global_reputation_of_user(acc.address)
                                                                         ))
                continue
            if self.fork:
                tx = super().build_tx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.provideHashedWeights(acc.hashedModel, acc.secret).transact(tx)

            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                hw = super().build_non_fork_tx(acc.address, nonce)
                hw =  self.model.functions.provideHashedWeights(acc.hashedModel, acc.secret).build_transaction(hw)
                signed = self.w3.eth.account.sign_transaction(hw, private_key=acc.privateKey)
                txHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Weights provided:", 
                                                                         acc.address[0:16] + "...", 
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.get_global_reputation_of_user(acc.address)
                                                                         ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_weights.append(receipt["gasUsed"])
            self.txHashes.append(("weights", receipt["transactionHash"].hex()))
        printer._print("-----------------------------------------------------------------------------------\n")
        

             
    def give_feedback(self, feedbackGiver, target, score):
        """
        Send a feedback transaction from feedbackGiver to target with given score:
          1  -> positive
          0  -> neutral
         -1  -> negative

        If target is in feedbackGiver.cheater list, force score to -1.
        """
        time.sleep(0.1)
        tx = super().build_tx(feedbackGiver.address, self.modelAddress, 0)
        #data = "0x" + encode_abi(['address', 'uint'], [target, score]).hex()
        if target in feedbackGiver.cheater:
            score = -1
        try:
            if self.fork:
                txHash = self.model.functions.feedback(target.address, score).transact(tx)
            else:          
                nonce = self.w3.eth.get_transaction_count(feedbackGiver.address)
                fe = super().build_non_fork_tx(feedbackGiver.address, nonce)
                fe =  self.model.functions.feedback(target.address, score).build_transaction(fe)
                signed = self.w3.eth.account.sign_transaction(fe, private_key=feedbackGiver.privateKey)
                txHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        except ContractLogicError as e:
            if "FRC" in str(e):
                input("Inactive users found - such users do not provide hashed weights.. \nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
                time.sleep(1)
                txHash = self.model.functions.feedback(target.address, score).transact(tx)
            else:
                print(rb("Encountered error at feedback function"))
                raise 
                
        assert(txHash != None)
        
        if score == 1:
            target.roundRep += 1 * self.get_global_reputation_of_user(feedbackGiver.address)
            rep = "Positive"
            pre = "+"
            col = "green"

        elif score == 0:
            rep = "Neutral"
            pre = "+"
            col = None
        else:
            target.roundRep -= 1 * self.get_global_reputation_of_user(feedbackGiver.address)
            rep = "Negative"
            pre = "-"
            col = "red"
        fb = "Feedback:".format(rep)
        
        print(colored("{:<11} {}   |" \
            " {}  | {}{:>25,.0f} WEI".format(fb, 
                                    feedbackGiver.address[0:7]+"... --> "+target.address[0:7]+"...", 
                                    txHash.hex()[0:6] + "...",
                                    pre,
                                    self.get_global_reputation_of_user(feedbackGiver.address)), col))
        return txHash
        
            
    
    def return_stats(self):
        print("\n==================================================================================\n")
        print("\n{:<8}{:^32}  {:^32}".format(f"ROUND {self.pytorch_model.round}","GLOBAL REPUTATION", "ROUND REPUTATION"))
        for acc in self.pytorch_model.participants:
            gs = self.get_global_reputation_of_user(acc.address)
            rs = self.get_round_reputation_of_user(acc.address)
            print("{}..: {:>27,.0f}  {:>27,.0f} WEI".format(acc.address[0:7],gs,rs))
        print("\n==================================================================================\n")
    
            
    def feedback_round(self, fbm):
        txs = []
        for user in self.pytorch_model.participants:
            user_votes = fbm[user.id]
            for ix, vote in enumerate(user_votes):
                if user.id == ix:
                    continue
                if user.attitude == "inactive":
                    continue
                txHash = self.giveFeedback(user, self.pytorch_model.participants[ix], int(vote))
                txs.append(txHash)
           
        l = len(txs)
        for i, txHash in enumerate(txs):
            if txHash == None:
                continue
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_feedback.append(receipt["gasUsed"])
            self.txHashes.append(("feedback", receipt["transactionHash"].hex()))
        for user in self.pytorch_model.participants:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
            
        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
        printer._print("                                                   ")
        print("\n-----------------------------------------------------------------------------------")

    def build_feedback_bytes(self, a, v):
        fbb = ""  # keep as string

        # Addresses: slice last 20 bytes to mimic original behavior
        for addr in a:
            encoded_addr = encode(["address"], [addr])  # 32 bytes
            fbb += encoded_addr.hex()[24:]  # take last 20 bytes in hex

        # Integers: full 32 bytes
        for val in v:
            fbb += encode(["int256"], [val]).hex()

        return fbb

                
    
    def quick_feedback_round(self, fbm, feedback_type, am = None, lm = None, prev_accs = None, prev_losses = None):
        print("Users exchanging feedback...")
        txs = []
        for idx, user in enumerate(self.pytorch_model.participants):
            addrs = []
            votes = []
            user_votes = fbm[user.id]
            filtered_accs = []
            filtered_losses = []
            accs = am[idx]
            losses = lm[idx]
            for ix, vote in enumerate(user_votes):
                if user.id == ix:
                    continue
                if user.attitude == "inactive":
                    continue
                if ix in [i.id for i in self.pytorch_model.disqualified]:
                    continue
                votee = [_u for _u in self.pytorch_model.participants if _u.id == ix][0]
                addrs.append(votee.address)
                votes.append(int(vote))
                votee.roundRep = votee.roundRep + self.get_global_reputation_of_user(user.address) * int(vote)
                filtered_accs.append(accs[ix])
                filtered_losses.append(losses[ix])



            fbb = self.build_feedback_bytes(addrs, votes)
            if feedback_type == "fallback":
                txs.append(self.send_fallback_transaction_onchain(_to=self.modelAddress, _from=user.address, data=fbb,
                                                              private_key=user.privateKey))
            elif feedback_type == "feedbackBytes":
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytes(Web3.to_bytes(hexstr="0x" + fbb)).transact(tx)
                else:  # TODO: Dobbeltjek at logic er rigtig her.
                    nonce = self.w3.eth.get_transaction_count(user.address)
                    cl = super().build_non_fork_tx(user.address, nonce)
                    cl = self.model.functions.submitFeedbackBytes(
                        fbb
                    ).build_transaction(cl)
                    pk = user.privateKey
                    signed = self.w3.eth.account.sign_transaction(cl, private_key=pk)
                    tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                txs.append(tx_hash)
            elif feedback_type == "feedbackBytesAndAccuracy":
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    row_am = am[idx]
                    row_lm = lm[idx]
                    filtered_row_am = row_am[:idx] + row_am[idx + 1:]
                    filtered_row_lm = row_lm[:idx] + row_lm[idx + 1:]
                    prev_acc = prev_accs[idx]
                    prev_loss = prev_losses[idx]

                    tx_hash = self.model.functions.submitFeedbackBytesAndAccuracies(Web3.to_bytes(hexstr="0x" + fbb), filtered_accs, filtered_losses, prev_acc, prev_loss).transact(tx)
                else:  # TODO: Dobbeltjek at logic er rigtig her.
                    nonce = self.w3.eth.get_transaction_count(user.address)
                    cl = super().build_non_fork_tx(user.address, nonce)
                    cl = self.model.functions.submitFeedbackBytesAndAccuracies(
                        Web3.to_bytes(hexstr="0x" + fbb), am[idx]
                    ).build_transaction(cl)
                    pk = user.privateKey
                    signed = self.w3.eth.account.sign_transaction(cl, private_key=pk)
                    tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                txs.append(tx_hash)
            else:
                warnings.warn("INVALID FEEDBACK TYPE")

        for i, txHash in enumerate(txs):
            self.log_receipt(i, txHash, len(txs), "feedback")

        for user in self.pytorch_model.participants:
            if len(user._roundrep) == 0:
                print(f"model participant: {user.address} had no roundrep")
            else:
                print(f"model participant: {user.address} had {user._roundrep[-1]} round reputation")
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
            print(f"model participant: {user.address} now gets {user._roundrep[-1]} round reputation")

        for user in self.pytorch_model.disqualified:
            if len(user._roundrep) == 0:
                print(f"model participant: {user.address} had no roundrep")
            else:
                print(f"model disquilified: {user.address} had {user._roundrep[-1]} round reputation")
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
            print(f"model disqualified: {user.address} now gets {user._roundrep[-1]} round reputation")

        printer._print("                                                   ")
        print("\n-----------------------------------------------------------------------------------")

    def log_receipt(self, i, tx_hash, len_txs, receipt_type: str):
        printer.print_bar(i, len_txs)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash,
                                                           timeout=600,
                                                           poll_latency=1)

        self.gas_feedback.append(receipt["gasUsed"])
        self.txHashes.append((receipt_type, receipt["transactionHash"].hex()))


    def send_fallback_transaction_onchain(self, _to, _from, data, private_key=None):
        try:
            if self.fork:
                tx_hash = self.w3.eth.send_transaction({'to': _to, 'from': _from, 'data': data})
            else:
                nonce = self.w3.eth.get_transaction_count(_from)
                hw = super().build_non_fork_tx(_from, nonce, self.modelAddress, 0, data)
                signed = self.w3.eth.account.sign_transaction(hw, private_key=private_key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        except ContractLogicError as e:
            if "FRC" in str(e):
                input("Inactive users found - such users do not " \
                      + "provide hashed weights.. \nGoing to forward time for 1 day\n")

                self.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
                time.sleep(1)
                tx_hash = self.w3.eth.send_transaction({'to': _to,
                                                       'from': _from,
                                                       'data': data,
                                                       "gas": 500000})
            else:
                print(rb("Encountered error at feedback function"))
                raise
        return tx_hash

    def call_close_feedback_round(self, force):
        if self.fork:
            tx = super().build_tx(self.w3.eth.default_account, self.modelAddress, 0)
            txHash = self.model.functions.closeFeedBackRound(force).transact(tx)

        else:
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address)
            cl = super().build_non_fork_tx(self.pytorch_model.participants[0].address,
                                        nonce,
                                        self.modelAddress,
                                        0)
            cl =  self.model.functions.closeFeedBackRound(force).buildTransaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = self.w3.eth.account.signTransaction(cl, private_key=pk)
            txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)

        return self.w3.eth.wait_for_transaction_receipt(txHash,
                                                        timeout=600,
                                                        poll_latency=1)

    def close_round(self):
        if "inactive" in [acc.attitude for acc in self.pytorch_model.participants]:
                input("Inactive users found - such users do not provide feedback.. " \
                          + "\nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
        
        print(b(f"\Feedback round: {self.pytorch_model.round}"))
        settleStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < settleStart + config.get_contracts_config().FEEDBACK_ROUND_TIMEOUT):
            if (self.model.functions.isFeedBackRoundDone().call()):
                print("Feedback round completed")
                break

            print("Feedback round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            print("Feedback round failed, forcing Contribution...")

        print(b(f"\Contribution round: {self.pytorch_model.round}"))
        contributionStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < contributionStart + config.get_contracts_config().CONTRIBUTION_ROUND_TIMEOUT):
            if (self.model.functions.isContributionRoundDone().call()):
                print("Contribution round completed")
                break
            print("Contribution round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            print("Contribution round failed, forcing settlement...")


        print(b(f"\Settling round: {self.pytorch_model.round}"))
        if self.fork:
            tx = super().build_tx(self.w3.eth.default_account, self.modelAddress, 0)
            txHash = self.model.functions.settle().transact(tx)
            
        else:          
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address, 'pending')
            cl = super().build_non_fork_tx(self.pytorch_model.participants[0].address, nonce)
            cl =  self.model.functions.settle().build_transaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = self.w3.eth.account.sign_transaction(cl, private_key=pk)
            txHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                        timeout=600,
                                                        poll_latency=1)
        print("settling round completed")

        self.txHashes.append(("close", receipt["transactionHash"].hex()))
        self.gas_close.append(receipt["gasUsed"])
        if len(receipt.logs) == 0:
            print("Warning: closeFeedBackRound() emitted no logs")
        self.pytorch_model.round += 1
        self._reward_balance.append(self.get_reward_left())
        printer._print("\n-----------------------------------------------------------------------------------\n")
        return receipt

    def user_register_slot(self):
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.attitude == "inactive":
                print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Account inactive:", 
                                                                         acc.address[0:16] + "...", 
                                                                         "   ...   ",
                                                                         self.get_global_reputation_of_user(acc.address)
                                                                         ))
                continue

            # print("type: ", type(acc.hashedModel)) hexbytes!!
            reservation = Web3.solidity_keccak(['bytes32', 'uint256', 'address'],
                                              [acc.hashedModel,
                                               acc.secret, acc.address])
            if self.fork:
                tx = super().build_tx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.registerSlot(reservation).transact(tx)
            else:
                w3 = ConnectionHelper.get_w3()          
                nonce = w3.eth.get_transaction_count(acc.address) 
                sl = super().build_non_fork_tx(acc.address, nonce)
                sl =  self.model.functions.registerSlot(reservation).build_transaction(sl)
                signed = w3.eth.account.sign_transaction(sl, private_key=acc.privateKey)
                txHash = w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Slot registered: ", 
                                                                         acc.address[0:16] + "...", 
                                                                         txHash.hex()[0:6] + "...",
                                                                         self.get_global_reputation_of_user(acc.address)
                                                                         ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_slot.append(receipt["gasUsed"])
            self.txHashes.append(("slot", receipt["transactionHash"].hex()))
        printer._print("-----------------------------------------------------------------------------------\n")
        return 
    
    
    
    def exit_system(self):
      
        print(b(f"Terminating Model..."))
       
        txs = []
        for acc in self.pytorch_model.participants:
            
            if self.fork:
                tx = super().build_tx(acc.address, self.modelAddress, 0)
                txHash = self.model.functions.exitModel().transact(tx)
            else:
                w3 = ConnectionHelper.get_w3()          
                nonce = w3.eth.get_transaction_count(acc.address) 
                ex = super().build_non_fork_tx(acc.address, nonce)
                ex =  self.model.functions.exitModel().build_transaction(ex)
                signed = w3.eth.account.sign_transaction(ex, private_key=acc.privateKey)
                txHash = w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(txHash)
            print("{:<17}   {} | {} | {:>27,.0f} WEI".format("Account exited:  ", 
                                                             acc.address[0:16] + "...", 
                                                             txHash.hex()[0:6] + "...",
                                                             self.w3.eth.get_balance(acc.address)
                                                             ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_exit.append(receipt["gasUsed"])
            self.txHashes.append(("exit", receipt["transactionHash"].hex()))
        printer._print("-----------------------------------------------------------------------------------\n")

    def get_events(self, w3, contract, receipt, event_names):
        """
        Returns decoded events without ABI mismatch warnings.

        Args:
            w3: Web3 instance
            contract: Contract instance
            receipt: transaction receipt
            event_names: list of event names to extract

        Returns:
            dict: {eventName: [decodedEvents...]}
        """
        results = {name: [] for name in event_names}

        for name in event_names:
            event_abi = getattr(contract.events, name)().abi
            event_signature = w3.keccak(
                text=f"{name}(" + ",".join(i["type"] for i in event_abi["inputs"]) + ")").hex()

            for log in receipt.logs:
                if log["topics"][0].hex() == event_signature:
                    decoded = getattr(contract.events, name)().process_log(log)
                    results[name].append(decoded)

        return results
    def print_round_summary(self, receipt):
        events = self.get_events(
            w3=self.w3,
            contract=self.model,
            receipt=receipt,
            event_names=["EndRound", "Reward", "Punishment", "Disqualification"]
        )

        end_events = events["EndRound"]
        reward_events = events["Reward"]
        punish_events = events["Punishment"]
        disqualify_events = events["Disqualification"]

        # 🟦 End of round summary
        if end_events:
            for ev in end_events:
                args = ev["args"]
                print(b(f"\nEND OF ROUND {args['round'] + 1}"))
                print(b(f"VALID VOTES:      {args['validVotes']}"))
                print(b(f"SUM OF WEIGHTS:  {args['sumOfWeights']:,}"))
                print(b(f"TOTAL PUNISHMENT: {args['totalPunishment']:,}\n"))
            print("-----------------------------------------------------------------------------------\n")

        # 🟩 Rewarded users
        if reward_events:
            print(b("REWARDED USERS"))
            for ev in reward_events:
                args = ev["args"]
                if args["roundScore"] > 0:
                    print(green(f"USER @ {args['user']}"))
                    print(green(f"ROUND SCORE:      {args['roundScore']:,}"))
                    print(green(f"TOTAL REWARD:     {args['win']:,}"))
                    print(green(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
            print("-----------------------------------------------------------------------------------\n")

        # 🟥 Punished users
        if punish_events:
            print(b("PUNISHED USERS"))
            for ev in punish_events:
                args = ev["args"]
                self._punishments.append((self.pytorch_model.round - 1, args["loss"]))
                print(red(f"USER @ {args['victim']}"))
                print(red(f"ROUND SCORE:      {args['roundScore']:,}"))
                print(red(f"TOTAL LOSS:       {args['loss']:,}"))
                print(red(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
            print("-----------------------------------------------------------------------------------\n")

        # 🟧 Disqualified users
        if disqualify_events:
            print(b("DISQUALIFIED USERS"))
            for ev in disqualify_events:
                args = ev["args"]
                self._punishments.append((self.pytorch_model.round - 1, args["loss"]))

                # Mark and remove disqualified users
                for user in list(self.pytorch_model.participants):  # safe remove
                    if args["victim"] == user.address:
                        user.disqualified = True
                        self.pytorch_model.disqualified.append(user)
                        self.pytorch_model.participants.remove(user)

                print(red(f"USER @ {args['victim']}"))
                print(red(f"ROUND SCORE:      {args['roundScore']:,}"))
                print(red(f"TOTAL LOSS:       {args['loss']:,}"))
                print(red(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
            print("-----------------------------------------------------------------------------------\n")

        print()

    # def contribution_score_old(self, _users):
    #     print("START CONTRIBUTION SCORE\n")
    #     merged_model = _users[0].model
    #     num_mergers = len(_users)
    #     txs = []
    #     for u in _users:
    #         u.roundRep = 0
    #         score = calc_contribution_score(u.previousModel, merged_model, num_mergers)
    #         u.is_contrib_score_negative = True if score < 0 else False
    #         u.contribution_score = score
    #
    #         if self.fork:
    #             tx = super().build_tx(u.address, self.modelAddress)
    #             tx_hash = self.model.functions.submitContributionScore(abs(score),
    #                                                                    u.is_contrib_score_negative).transact(tx)
    #         else:
    #             nonce = self.w3.eth.get_transaction_count(u.address)
    #             cl = super().buildNonForkTx(u.address,
    #                                         nonce,
    #                                         self.modelAddress)
    #             cl = self.model.functions.settleContributionScore(abs(score),
    #                                                               u.is_contrib_score_negative).buildTransaction(cl)
    #             pk = u.private_key
    #             signed = self.w3.eth.account.signTransaction(cl, private_key=pk)
    #             tx_hash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
    #         txs.append(tx_hash)
    #
    #         print(green(f"\nUSER @ {u.id}"))
    #         print(green(f"{'CONTRIBUTION SCORE:':25} {u.contribution_score:}"))
    #
    #     for i, txHash in enumerate(txs):
    #         self.log_receipt(i, txHash, len(txs), "con_score")
    #     print("-----------------------------------------------------------------------------------\n")



    # New contribution score
    def contribution_score(self, _users):
        """
        Compute contribution scores for all merging users, submit them to the
        contract, and log them. Strategy is chosen by _get_contribution_score_calculator:
          - legacy: simple dot-product
          - mad: MAD-based outlier filtering of weights
          - naive: equal-share (1 / num_mergers)
        """

        print("START CONTRIBUTION SCORE\n")
        voters, accs, losses = self.model.functions.getAllAccuraciesAbout(_users[0].address).call()
        for v, a, l in zip(voters, accs, losses):
            print(f"{v} gave accuracy {a} and loss {l} for target {_users[0].address}")
        prev_accs, prev_losses = self.model.functions.getAllPreviousAccuraciesAndLosses.call()
        print(f"previous accuracies: {prev_accs}")
        print(f"previous losses: {prev_losses}")

        merged_model = _users[0].model

        # Choose scoring algorithm based on configured strategy
        calculator = self._get_contribution_score_calculator()
        scores = calculator(_users)




        txs = []
        for u, score in zip(_users, scores):
            u.is_contrib_score_negative = True if score < 0 else False
            u.contribution_score = score

            if self.fork:
                tx = super().build_tx(u.address, self.modelAddress)
                tx_hash = self.model.functions.submitContributionScore(
                    abs(score),
                    u.is_contrib_score_negative
                ).transact(tx)
            else:  # TODO: Dobbeltjek at logic er rigtig her.
                nonce = self.w3.eth.get_transaction_count(u.address)
                cl = super().build_non_fork_tx(
                    u.address,
                    nonce,
                )
                cl = self.model.functions.submitContributionScore(
                    abs(score),
                    u.is_contrib_score_negative
                ).build_transaction(cl)
                pk = u.privateKey
                signed = self.w3.eth.account.sign_transaction(cl, private_key=pk)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            txs.append(tx_hash)

            print(green(f"\nUSER @ {u.id}"))
            print(green(f"{'CONTRIBUTION SCORE:':25} {u.contribution_score:}"))

        for i, txHash in enumerate(txs):
            self.log_receipt(i, txHash, len(txs), "contrib")

        print("-----------------------------------------------------------------------------------\n")


    def _calculate_scores_legacy(self, users, merged_model):
        """
        Legacy scoring: for each user, use dot-product–based contribution score.
        """
        num_mergers = len(users)
        return [
            calc_contribution_score(u.previousModel, merged_model, num_mergers)
            for u in users
        ]

    def _calculate_scores_mad(self, users, merged_model):
        """
        MAD-based scoring: robust per-weight outlier filtering before scoring.
        """
        global_update = torch.cat([p.data.view(-1) for p in merged_model.parameters()])
        local_updates = [
            torch.cat([p.data.view(-1) for p in u.previousModel.parameters()])
            for u in users
        ]
        local_updates = torch.stack(local_updates)
        return calc_contribution_scores_mad(local_updates, global_update)

    def _calculate_scores_naive(self, users, merged_model):
        """
        Equal-share scoring: everyone contributing gets 1 / num_mergers.
        """
        _ = merged_model  # unused; included for signature consistency
        num_mergers = len(users)
        return [calc_contribution_score_naive(num_mergers) for _ in users]



    def _calculate_scores_accuracy(self, users):
        """
        Accuracy-based scoring: use accuracy directly as contribution score.
        """

        # accuracies: 1d array
        # losses: 1d array
        # prev_acc: int

        # Array of previous accuracies and losses from all users: A tuple of arrays
        prev_accuracies, prev_losses = self.model.functions.getAllPreviousAccuraciesAndLosses.call()

        # use mad on these and average them
        mad_treshold = 10

        mad_prev_accuracies = remove_outliers_mad(prev_accuracies, mad_treshold)
        mad_prev_losses = remove_outliers_mad(prev_losses, mad_treshold)

        avg_prev_acc = np.mean(mad_prev_accuracies)
        avg_prev_loss = np.mean(mad_prev_losses)
        # Lav en fælles mad

        avg_accuracies = [] # after loop: [30, 20, 30, 40]
        avg_losses = [] # after loop: [60, 70, 50, 80]

        for u in users: # For loop to extract accuracies and loses.
            # Vi kan åbenbart ikke nøjes med kune dette, da vi skal bruge global accuracies, loses.

            # All accuracies and loses per user
            _, accuracies, losses = self.model.functions.getAllAccuraciesAbout(u.address).call()

            try:
                # Multiple accuracies and losses per user
                mad_accuracies = remove_outliers_mad(accuracies,mad_treshold)
                mad_losses = remove_outliers_mad(losses, mad_treshold)

                # One average accuracy and loss per user
                avg_acc = np.mean(mad_accuracies)
                avg_loss = np.mean(mad_losses)

                avg_accuracies.append(avg_acc) # int
                avg_losses.append(avg_loss) # int
            except ValueError:
                print("An error occured")

        scores = []

        norm_accuracies = calc_contribution_scores_accuracy(avg_accuracies, avg_prev_acc)

        norm_losses = calc_contribution_scores_accuracy(avg_losses, avg_prev_loss)

        inverted_losses = [1 - x for x in norm_losses]

        sum_na = sum(norm_accuracies)
        sum_nl = sum(inverted_losses)

        for i in range(len(norm_accuracies)):
            res = (norm_accuracies[i] + inverted_losses[i]) / (sum_na + sum_nl)
            score = int(Decimal(res) * Decimal('1e18'))
            scores.append(score)

        return scores


        # return scores
    # Output: An array of user scores
    # Find out who was merged




    def simulate(self, rounds):
        """
        Run a full FL simulation for a given number of rounds.
        High-level flow per round:
          1) Update user attitudes
          2) Local training
          3) Let malicious/freerider users modify/copy models
          4) Register slots & provide hashed weights
          5) Exchange and verify models
          6) Evaluation & feedback
          7) Merge models
          8) Compute contribution scores
          9) Close round, print summary
        At the end, all users exit the system.
        """
        hashedWeights = []
        print(self.modelAddress)
        self.register_all_users()
        
        for i in range(rounds):
            print(b(f"Round {self.pytorch_model.round} starts..."))
            self.pytorch_model.update_users_attitude()

            self.pytorch_model.federated_training()

            self.pytorch_model.let_malicious_users_do_their_work()
            self.pytorch_model.let_freerider_users_do_their_work()
            
            self.user_register_slot()

            self.users_provide_hashed_weights()

            self.pytorch_model.exchange_models()
            
            self.pytorch_model.verify_models({u.id: self.get_hashed_weights_of(u) for u in self.pytorch_model.participants})

            feedback_matrix, accuracy_matrix, loss_matrix, prev_accs, prev_losses = self.pytorch_model.evaluation()

            self.quick_feedback_round(fbm = feedback_matrix, feedback_type="feedbackBytesAndAccuracy", am=accuracy_matrix, lm=loss_matrix, prev_accs=prev_accs, prev_losses=prev_losses)

            self.pytorch_model.the_merge([user for user in self.pytorch_model.participants if user._roundrep[-1] > 0])
            
            print(b("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n"))

            #contributionScoreTask = asyncio.create_task(self.contribution_score([user for user in self.pytorch_model.participants if user.roundRep > 0]))
            self.contribution_score([user for user in self.pytorch_model.participants if user._roundrep[-1] > 0])
            receipt = self.close_round()
            #contributionScoreTask

            print(b(f"Round {self.pytorch_model.round - 1} actually completed:"))
            for user in self.pytorch_model.participants + self.pytorch_model.disqualified:
                user._globalrep.append(self.get_global_reputation_of_user(user.address))
                i, j = user._globalrep[-2:]
                print(b("{}  {:>25,.0f} -> {:>25,.0f}".format(user.address[0:16] + "...", i, j)))

            self.print_round_summary(receipt)

        self.exit_system()
            
            
    
    def visualize_simulation(self, output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)  # ensure folder exists
        accuracy = [0] + self.pytorch_model.accuracy
        loss = [self.pytorch_model.loss[0]] + self.pytorch_model.loss

        f, axs = plt.subplots(1, 4,figsize=(16, 3),  gridspec_kw={'width_ratios': [0.8,2,2,2],
                                                                      'height_ratios': [1]})
        colors = ["#00629b", "#629b00", "#000000", "#d93e6a"]

        participants =self.pytorch_model.participants + self.pytorch_model.disqualified

        #  True to get old setep graph, False to get point graph
        use_step_grs = False

        rounds = list(range(len(accuracy)))
        #x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]

        y = accuracy
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        acc_line = axs[1].plot(rounds, y, color=colors[0], linewidth=2.5, label="Avg. Accuracy")[0]
        twin = axs[1].twinx()
        y = loss
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        loss_line = twin.plot(rounds, y, color=colors[1], linewidth=2.5, linestyle="--", label="Avg. Loss")[0]



        grs_rounds = list(range(len(participants[0]._globalrep)))
        if use_step_grs:
            grs_x = [item for sublist in zip(grs_rounds, (np.array(grs_rounds) + 1).tolist()) for item in sublist]
            grs_ticks = grs_rounds
            for i, user in enumerate(participants):
                grs_y = [item for sublist in zip(user._globalrep, user._globalrep) for item in sublist]
                axs[2].plot(grs_x, grs_y, linewidth=2.5, color=user.color)
        else:
            grs_x = grs_rounds
            grs_ticks = grs_rounds
            # plotting the points
            for i, user in enumerate(participants):
                axs[2].plot(
                    grs_x,
                    user._globalrep,
                    linewidth=2.5,
                    color=user.color,
                    alpha=0.9,
                    marker="o",
                    markersize=4,
                    markevery=1,
                )



        pun = {}
        for i, j in self._punishments:
            if i in pun.keys():
                pun[i] += j
            else:
                pun[i] = j

        rew = list()
        for i, j in enumerate(self._reward_balance):
            if i in pun.keys():
                rew.append(j+pun[i])
            else:
                rew.append(j)    

        y_reward = [item for sublist in zip(self._reward_balance,self._reward_balance) for item in sublist]
        y2_reward = [item for sublist in zip(rew,rew) for item in sublist]
        x_reward = list(range(len(self._reward_balance)))
        x_reward = [item for sublist in zip(x_reward,(np.array(x_reward)+1).tolist()) for item in sublist]


        axs[3].plot(x_reward,y_reward, label="reward", color=colors[0], linewidth=2.5)
        axs[3].plot(x_reward,y2_reward, label="reward +\npunishments", color=colors[1], linewidth=2.5)
        axs[3].fill_between(x_reward,y_reward, y2_reward, alpha=0.2, hatch=r"//", color = colors[1])


        axs[0].text(0, 0.1, f'dataset={self.pytorch_model.DATASET}\n'\
                                 + f'epochs={self.pytorch_model.EPOCHS}\n' \
                                 + f'rounds={self.pytorch_model.round-1}\n' \
                                 + f'users={self.pytorch_model.NUMBER_OF_CONTRIBUTERS}\n' \
                                 + f'malicious={self.pytorch_model.NUMBER_OF_BAD_CONTRIBUTORS}\n'\
                                 + f'copycat={self.pytorch_model.NUMBER_OF_FREERIDER_CONTRIBUTORS}', fontsize=15)
        axs[0].set_axis_off()
        
        axs[1].set_xlabel('rounds\n(a)', fontsize=14)
        axs[2].set_xlabel('rounds\n(b)', fontsize=14)
        axs[3].set_xlabel('rounds\n(c)', fontsize=14)
        #axs[0].set_ylabel(f'users={participants};\n malicious={malicious_users};\n copycat={sneaky_freerider}', fontsize=14)
        axs[1].set_ylabel('Avg. Accuracy', fontsize=14)
        twin.set_ylabel('Avg. Loss', fontsize=14)
        axs[1].tick_params(axis='both', which='major', labelsize=14)

        axs[2].set_ylabel('GRS', fontsize=14)
        axs[3].set_ylabel('Contract Balance', fontsize=14)

        axs[2].tick_params(axis='both', which='major', labelsize=14)
        axs[3].tick_params(axis='both', which='major', labelsize=14)
        
        if len(rounds) > 20:
            axs[1].set_xticks([i for i in rounds if i%5==0 or i == 0])
        else:
            axs[1].set_xticks([i for i in rounds])

        if len(grs_ticks) > 20:
            axs[2].set_xticks([i for i in grs_ticks if i%5==0 or i == 0])
        else:
            axs[2].set_xticks([i for i in grs_ticks])

        if len(x_reward) > 20:
            axs[3].set_xticks([i for i in x_reward if i%5==0 or i == 0])
        else:
            axs[3].set_xticks([i for i in x_reward])
    
        axs[1].set_xlim(0, max(rounds) if rounds else 0)
        
        axs[2].yaxis.get_offset_text().set_fontsize(14)
        axs[3].yaxis.get_offset_text().set_fontsize(14)
        
        axs[1].grid()
        axs[2].grid()
        axs[3].grid()

        # Legend for the dual-axis accuracy/loss plot
        twin_lines = [acc_line, loss_line]
        axs[1].legend(twin_lines, [l.get_label() for l in twin_lines], loc="lower right", fontsize=10)

        lgnd = axs[3].legend(fontsize=10)

        # giving a title to my graph 
        #axs[1].set_title(f'users={participants}; malicious={malicious_users}; copycat={sneaky_freerider}', fontsize=10) 

        # function to show the plot
        print(output_folder_path)
        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(output_folder_path, f"{self.pytorch_model.DATASET}_simulation.pdf"), bbox_inches='tight')
        #plt.show()
        return plt


def calc_contribution_score(local_model, global_model, num_mergers, eps=1e-12) -> int:
    """
    FedAvg-normalized dot product score so that sum = 1.

    Args:
        u: local model
        U: global model found by FedAvg
        num_clients: int, number of clients that merged this round
        eps: float, small tolerance to avoid division by zero

    Returns:
        contribution score in WEI.
        1 * 1e18 is 100% contribution score
    """

    # Flatten parameters
    local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
    global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])

    norm_U_sq = torch.dot(global_update, global_update)

    if norm_U_sq.abs() < eps:
        score = Decimal(1) / Decimal(num_mergers)
        return int(score * Decimal('1e18'))
    score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)

    return int(Decimal(score.item()) * Decimal('1e18'))


def calc_contribution_score_naive(num_mergers) -> int:
    score = Decimal(1) / Decimal(num_mergers)
    return int(score * Decimal('1e18'))

# New function
def calc_contribution_scores_mad(local_updates: torch.Tensor,
                                 global_update: torch.Tensor,
                                 eps: float = 1e-12,
                                 mad_thresh: float = 3.5):
    """
    Compute contribution scores using MAD-based outlier filtering on weights.

    Args:
        local_updates: Tensor of shape (num_mergers, D)
                       flattened parameters for each user's local model.
        global_update: Tensor of shape (D,)
                       flattened parameters for the global (merged) model.
        eps:           Small tolerance to avoid division by zero.
        mad_thresh:    Threshold on robust z-score to mark outliers.

    Returns:
        List[int]: contribution scores scaled by 1e18, like before.
    """

    num_mergers, D = local_updates.shape

    # --- MAD-based filtering (per-weight, across participants) ---
    # Median across users per weight
    median = local_updates.median(dim=0).values  # (D,)

    # Absolute deviations from median
    abs_dev = (local_updates - median).abs()     # (num_mergers, D)

    # Median absolute deviation per weight
    mad = abs_dev.median(dim=0).values           # (D,)

    # Avoid division by zero in MAD
    safe_mad = mad.clone()
    safe_mad[safe_mad < eps] = eps

    # Robust z-score for each weight/user
    # 0.6745 factor makes MAD comparable to std for normal data
    robust_z = 0.6745 * abs_dev / safe_mad       # (num_mergers, D)

    # Mask of "non-outlier" weights: True = keep, False = outlier
    mask = robust_z <= mad_thresh                # (num_mergers, D)

    # Zero-out outlier weights for each user individually
    filtered_local_updates = local_updates * mask

    # --- Dot-product scoring with filtered updates ---
    norm_U_sq = torch.dot(global_update, global_update)

    if norm_U_sq.abs() < eps:
        # Global update basically zero → give everyone equal share 1 / num_mergers
        score = Decimal(1) / Decimal(num_mergers)
        equal_int_score = int(score * Decimal('1e18'))
        return [equal_int_score for _ in range(num_mergers)]

    # For each user i: score_i = (u_i_filtered · U) / (num_mergers * ||U||^2)
    dots = torch.mv(filtered_local_updates, global_update)  # (num_mergers,)
    scores = dots / (num_mergers * norm_U_sq)

    # Convert to your integer fixed-point format (×1e18)
    int_scores = [
        int(Decimal(score.item()) * Decimal('1e18'))
        for score in scores
    ]
    return int_scores

    # norm_U_sq = torch.dot(global_update, global_update)
    #
    # if norm_U_sq.abs() < eps:
    #     # Global update is basically zero => everyone gets 0
    #     return [0 for _ in range(num_mergers)]
    #
    # # For each user i: score_i = (u_i_filtered · U) / (num_mergers * ||U||^2)
    # dots = torch.mv(filtered_local_updates, global_update)  # (num_mergers,)
    # scores = dots / (num_mergers * norm_U_sq)
    #
    # # Convert to your integer fixed-point format (×1e18)
    # int_scores = [
    #     int(Decimal(score.item()) * Decimal('1e18'))
    #     for score in scores
    # ]
    # return int_scores

# def flatten_model_params(model: torch.nn.Module) -> torch.Tensor:
#     return torch.cat([p.data.view(-1) for p in model.parameters()])

def calc_contribution_scores_accuracy(arr, prev_val):
    # This method takes a 1d array of an array (accuracy or loss) a scalar of previous accuracy or loss
    # Output is an array of normalized input array values
    norm_arr = []
    sum_val = 0.0

    for i in range(len(arr)):
        norm_arr.append(arr[i] - prev_val)
        sum_val += norm_arr[i]

    if len(norm_arr) == 0:
        raise Exception("No values to normalize")

    for i in range(len(norm_arr)):
        if sum_val == 0.0:
            return [1.0 / len(norm_arr)] * len(norm_arr)
        norm_arr[i] /= sum_val
    return norm_arr


def remove_outliers_mad(arr, z_threshold):
    arr = np.asarray(arr)
    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return arr

    # Compute z-scores
    zscores = (arr - mean) / std
    # Keep values with |z| <= threshold
    mask = np.abs(zscores) <= z_threshold

    return arr[mask]
