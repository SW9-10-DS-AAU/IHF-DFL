import os
import time
import warnings

import torch
import numpy as np
from eth_abi import encode
from web3 import Web3
from termcolor import colored
import matplotlib.pyplot as plt 
from web3.exceptions import ContractLogicError
from openfl.contracts import FLManager
from openfl.ml.pytorch_model import gb, rb, b, green, red
from openfl.utils import printer, config
from openfl.api.connection_helper import ConnectionHelper
from decimal import Decimal

class FLChallenge(FLManager):
    def __init__(self, manager, configs, pyTorchModel):
        self.manager = manager
        self.w3 = manager.w3
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

              
        
    def register_all_users(self):
        txs = []
        for acc in self.pytorch_model.participants:
            if acc.isRegistered:
                continue
            if self.fork:
                tx = super().build_tx(acc.address, self.modelAddress, acc.collateral)
                txHash = self.model.functions.register().transact(tx)
            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                reg = super().build_non_fork_tx(acc.address, nonce, self.modelAddress, acc.collateral)   
                reg = self.model.functions.register().buildTransaction(reg)
                signed = self.w3.eth.account.signTransaction(reg, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
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
                hw = super().build_non_fork_tx(acc.address, nonce, self.modelAddress, 0)   
                hw =  self.model.functions.provideHashedWeights(acc.hashedModel, acc.secret).buildTransaction(hw)
                signed = self.w3.eth.account.signTransaction(hw, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
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
        time.sleep(0.1)
        tx = super().build_tx(feedbackGiver.address, self.modelAddress, 0)
        #data = "0x" + encode_abi(['address', 'uint'], [target, score]).hex()
        if target in feedbackGiver.cheater:
            score = -1
        try:
            if self.fork:
                txHash = self.model.functions.feedback(target.address, score).transact(tx)
            else:          
                nonce = self.w3.eth.get_transaction_count(acc.address) 
                fe = super().build_non_fork_tx(acc.address, nonce, self.modelAddress, 0)   
                fe =  self.model.functions.feedback(target.address, score).buildTransaction(fe)
                signed = self.w3.eth.account.signTransaction(fe, private_key=acc.privateKey)
                txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
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

                
    
    def quick_feedback_round(self, fbm, feedback_type, am = None):
        print("Users exchanging feedback...")
        txs = []
        for user in self.pytorch_model.participants:
            addrs = []
            votes = []
            user_votes = fbm[user.id]
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

            fbb = self.build_feedback_bytes(addrs, votes)
            if feedback_type == "fallback":
                txs.append(self.send_fallback_transaction_onchain(_to=self.modelAddress, _from=user.address, data=fbb,
                                                              private_key=user.privateKey))
            elif feedback_type == "feedbackBytes":
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytes(fbb).transact(tx)
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
                    tx_hash = self.model.functions.submitFeedbackBytesAndAccuracies(fbb, am).transact(tx)
                else:  # TODO: Dobbeltjek at logic er rigtig her.
                    nonce = self.w3.eth.get_transaction_count(user.address)
                    cl = super().build_non_fork_tx(user.address, nonce)
                    cl = self.model.functions.submitFeedbackBytesAndAccuracies(
                        fbb, am
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
            user._roundrep.append(self.get_round_reputation_of_user(user.address))

        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))

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
                hw = super().buildNonForkTx(_from, nonce, self.modelAddress, 0, data)
                signed = self.w3.eth.account.signTransaction(hw, private_key=private_key)
                tx_hash = self.w3.eth.sendRawTransaction(signed.rawTransaction)

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


    def close_round(self):
        if "inactive" in [acc.attitude for acc in self.pytorch_model.participants]:
                input("Inactive users found - such users do not provide feedback.. " \
                          + "\nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])
        
        print(b(f"\nSettle round: {self.pytorch_model.round}"))
                
        if self.fork:
            tx = super().build_tx(self.w3.eth.default_account, self.modelAddress, 0)
            txHash = self.model.functions.closeRound().transact(tx)
            
        else:          
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address) 
            cl = super().build_non_fork_tx(self.pytorch_model.participants[0].address, 
                                        nonce, 
                                        self.modelAddress, 
                                        0)   
            cl =  self.model.functions.closeRound().buildTransaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = self.w3.eth.account.signTransaction(cl, private_key=pk)
            txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
        receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)          

        self.txHashes.append(("close", receipt["transactionHash"].hex()))
        self.gas_close.append(receipt["gasUsed"])
        if len(receipt.logs) == 0:
            print("⚠️ Warning: closeRound() emitted no logs")
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
                sl = super().build_non_fork_tx(acc.address, nonce, self.modelAddress, 0)   
                sl =  self.model.functions.registerSlot(reservation).buildTransaction(sl)
                signed = w3.eth.account.signTransaction(sl, private_key=acc.privateKey)
                txHash = w3.eth.sendRawTransaction(signed.rawTransaction)
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
                ex = super().build_non_fork_tx(acc.address, nonce, self.modelAddress, 0)   
                ex =  self.model.functions.exitModel().buildTransaction(ex)
                signed = w3.eth.account.signTransaction(ex, private_key=acc.privateKey)
                txHash = w3.eth.sendRawTransaction(signed.rawTransaction)
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

    def contribution_score(self, _users):
        print("START CONTRIBUTION SCORE\n")
        merged_model = _users[0].model
        num_mergers = len(_users)
        txs = []
        for u in _users:
            u.roundRep = 0
            score = calc_contribution_score(u.previousModel, merged_model, num_mergers)
            u.is_contrib_score_negative = True if score < 0 else False
            u.contribution_score = score

            if self.fork:
                tx = super().build_tx(u.address, self.modelAddress)
                tx_hash = self.model.functions.submitContributionScore(abs(score),
                                                                       u.is_contrib_score_negative).transact(tx)
            else:
                nonce = self.w3.eth.get_transaction_count(u.address)
                cl = super().buildNonForkTx(u.address,
                                            nonce,
                                            self.modelAddress)
                cl = self.model.functions.settleContributionScore(abs(score),
                                                                  u.is_contrib_score_negative).buildTransaction(cl)
                pk = u.private_key
                signed = self.w3.eth.account.signTransaction(cl, private_key=pk)
                tx_hash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            txs.append(tx_hash)

            print(green(f"\nUSER @ {u.id}"))
            print(green(f"{'CONTRIBUTION SCORE:':25} {u.contribution_score:}"))

        for i, txHash in enumerate(txs):
            self.log_receipt(i, txHash, len(txs), "con_score")
        print("-----------------------------------------------------------------------------------\n")


    def simulate(self, rounds):
        hashedWeights = []
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

            feedback_matrix, accuracy_matrix = self.pytorch_model.evaluation()
            # accuracy matrix
            # XX 60 64 69 69 70 75
            # 59 XX 66 67 78 75 40
            # 70 68 69 69 69 72 85
            # 85 70 42 56 65 70 71
            # 75 75 75 75 75 XX 75 # free-rider giver alle samme accuracy som han fik med global model fra sidste runde på eget data
            # 00 00 00 00 00 00 XX # malicious giver alle andre 0 accuracy
            
            self.quick_feedback_round(fbm = feedback_matrix, feedback_type="fallback", am=accuracy_matrix)
            # self.quick_feedback_round(feedback_matrix, accuracy_matrix, fallback_type="feedbackBytes")
            # self.quick_feedback_round(feedback_matrix, accuracy_matrix, fallback_type="feedbackBytesAndAccuracy")

            self.pytorch_model.the_merge([user for user in self.pytorch_model.participants if user.roundRep > 0])
            
            print(b("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n"))

            self.contribution_score([user for user in self.pytorch_model.participants if user.roundRep > 0])

            receipt = self.close_round()
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

        x = list(range(0,len(accuracy)))
        #x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]

        y = accuracy
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        axs[1].plot(x, y, color=colors[0], linewidth=2.5) 
        twin = axs[1].twinx()
        y = loss
        #y = [item for sublist in zip(yy,yy) for item in sublist]
        twin.plot(x, y, color=colors[1], linewidth=2.5) 



        x = list(range(len(participants[0]._globalrep)))
        x = [item for sublist in zip(x,(np.array(x)+1).tolist()) for item in sublist]


        # plotting the points  
        yy=[]
        for i, user in enumerate(participants):
            y = [item for sublist in zip(user._globalrep, user._globalrep) for item in sublist]
            axs[2].plot(x, y, linewidth=2.5, color=user.color) 



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
        
        if len(x) > 20:
            axs[1].set_xticks([i for i in x if i%5==0 or i == 0])
            axs[2].set_xticks([i for i in x if i%5==0 or i == 0])
            axs[3].set_xticks([i for i in x if i%5==0 or i == 0])
        else:
            axs[1].set_xticks([i for i in x])
            axs[2].set_xticks([i for i in x])
            axs[3].set_xticks([i for i in x])
    
        axs[1].set_xlim(0,max(x))
        
        axs[2].yaxis.get_offset_text().set_fontsize(14)
        axs[3].yaxis.get_offset_text().set_fontsize(14)
        
        axs[1].grid()
        axs[2].grid()
        axs[3].grid()

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
    local_update = torch.cat([p.data.view(-1) for p in local_model.parameters()])
    global_update = torch.cat([p.data.view(-1) for p in global_model.parameters()])

    norm_U_sq = torch.dot(global_update, global_update)

    if norm_U_sq.abs() < eps:
        score = Decimal(1) / Decimal(num_mergers)
        return int(score * Decimal('1e18'))
    score = torch.dot(local_update, global_update) / (num_mergers * norm_U_sq)

    return int(Decimal(score.item()) * Decimal('1e18'))
