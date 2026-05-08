import datetime
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ml.attacks as attacks
import ml.evaluation as evaluation
import ml.aggregation as aggregation
from eth_abi import encode
from web3 import Web3
from termcolor import colored
from web3.exceptions import ContractLogicError
from utils.colors import rb, b, green, red
from utils import printer, config
from api.connection_helper import ConnectionHelper
from utils.async_writer import AsyncWriter, NullWriter
from contracts import contribution
from contracts import logging
from contracts.contribution import contribution_score

UINT256_MAX = 2**256 - 1

class FLChallenge(ConnectionHelper):
    """
    Smart-contract-backed federated learning simulation.

    Handles:
      - User registration / exit on-chain
      - Hashed model submission & slot reservation
      - Feedback exchange (reputation updates)
      - Contribution scoring (delegated to `contribution`)
      - Round settlement and visualization
    """
    def __init__(self, manager, configs, pyTorchModel, experiment_config, writer: AsyncWriter=None, logger=None): # pragma: no cover
        self.w3 = manager.w3
        self.model, self.modelAddress = configs[:2]
        self.pytorch_model = pyTorchModel
        self.MIN_BUY_IN, self.MAX_BUY_IN, self.REWARD, self.MIN_ROUNDS = configs[2:6]
        self.PUNISHMENT_FACTOR = configs[6]
        self.PUNISHMENT_FACTOR_CONTRIB = configs[7]
        self.FREERIDER_FACTOR = configs[8]
        self.fork = manager.fork
        self.gas_feedback = []
        self.gas_register = []
        self.gas_slot     = []
        self.gas_weights  = []
        self.gas_close    = []
        self.gas_deploy   = []
        self.gas_exit     = []
        self.txHashes     = []
        self.scores       = []
        self._reward_balance = [self.REWARD]
        self._punishments = []
        self.config = config.get_contracts_config()
        self.writer = writer or NullWriter()
        self._logger = logger
        self.writeTxProgress = 0
        self.experiment_config = experiment_config
        self.disqualifiedUserEvents = []

        print("Contract address:", self.model.address)
        print("Contract ABI functions:", [f["name"] for f in self.model.abi if f["type"] == "function"])

        
    def register_all_users(self): # pragma: no cover
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
            self.txHashes.append(("register",receipt["transactionHash"].hex(), receipt["gasUsed"]))
            logging.log_receipt(self, receipt, "register", round=0)
        printer._print("-----------------------------------------------------------------------------------", "\n")
        
    
    def get_hashed_weights_of(self, user): # pragma: no cover
        return self.model.functions.weightsOf(user.address,self.pytorch_model.round-1).call({"to": self.modelAddress})


    def get_global_reputation_of_user(self, user_addr):
        user = self.model.functions.getUser(user_addr).call()
        return user[2]


    def get_round_reputation_of_user(self, user): # pragma: no cover
        user_struct = self.model.functions.users(user).call()
        return user_struct[2]


    def get_all_accuracies_and_losses_about(self, user_addr):
        voters, accuracies, losses = self.model.functions.getAllAccuraciesLossesAbout(user_addr).call()
        return voters, accuracies, losses


    def get_all_accuracies_about(self, user_addr):
        voters, accuracies = self.model.functions.getAllAccuraciesAbout(user_addr).call()
        return voters, accuracies


    def get_all_losses_about(self, user_addr):
        voters, losses = self.model.functions.getAllLossesAbout(user_addr).call()
        return voters, losses


    # 'all' as in users
    def get_all_previous_accuracies_and_losses(self):
        prev_accuracies, prev_losses = self.model.functions.getAllPreviousAccuraciesAndLosses().call()
        return prev_accuracies, prev_losses


    # def get_all_n_prior_losses(self, n_rounds: int):
    #     # returns whatever rounds are available, up to n_rounds. So:
    #     #   - Round 0 or 1: returns empty list []
    #     #   - Round 2: returns [round-1] — only one round back
    #     #   - Round 3: returns [round-1, round-2]
    #     #   - Round 5+: returns [round-1, round-2, round-3, round-4] (if n_rounds=4)
    #     #
    #     #   The caller checks the length and decides what to do — if fewer than 2 entries, not enough data to compute a trend, fall back to default
    #     #   behavior.
    #     assert n_rounds >= 2, "n_rounds must be at least 2 to compute a trend"
    #     contract_round = self.model.functions.round().call()
    #     losses_per_round = []
    #
    #     for steps_back in range(1, n_rounds + 1):
    #         if contract_round >= steps_back:
    #             losses = self.model.functions.getAllNPriorLosses(steps_back).call()
    #             mad_losses = contribution.remove_outliers_mad(losses)
    #             losses_per_round.append(np.mean(mad_losses))
    #     return losses_per_round  # [round-1, round-2, ..., round-n]


    def get_reward_left(self): # pragma: no cover
        return self.model.functions.rewardLeft().call({"to": self.modelAddress})


    def users_provide_hashed_weights(self): # pragma: no cover

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
            # print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Weights provided:",
            #                                                              acc.address[0:16] + "...",
            #                                                              txHash.hex()[0:6] + "...",
            #                                                              self.get_global_reputation_of_user(acc.address)
            #                                                              ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_weights.append(receipt["gasUsed"])
            self.txHashes.append(("weights", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            logging.log_receipt(self, receipt, "weights")
        # printer._print("-----------------------------------------------------------------------------------\n")

             
    def give_feedback(self, feedbackGiver, target, score): # pragma: no cover
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
                raise e
                
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

    
    def return_stats(self): # pragma: no cover
        print("\n==================================================================================\n")
        print("\n{:<8}{:^32}  {:^32}".format(f"ROUND {self.pytorch_model.round}","GLOBAL REPUTATION", "ROUND REPUTATION"))
        for acc in self.pytorch_model.participants:
            gs = self.get_global_reputation_of_user(acc.address)
            rs = self.get_round_reputation_of_user(acc.address)
            print("{}..: {:>27,.0f}  {:>27,.0f} WEI".format(acc.address[0:7],gs,rs))
        print("\n==================================================================================\n")
    
            
    def feedback_round(self, fbm): # pragma: no cover
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
            self.txHashes.append(("feedback", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            logging.log_receipt(self, receipt, "feedback")
        for user in self.pytorch_model.participants:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))

        for user in self.pytorch_model.disqualified:
            user._roundrep.append(self.get_round_reputation_of_user(user.address))
        # printer._print("                                                   ")
        # print("\n-----------------------------------------------------------------------------------")


    def build_feedback_bytes(self, a, v): # pragma: no cover
        fbb = ""  # keep as string

        # Addresses: slice last 20 bytes to mimic original behavior
        for addr in a:
            encoded_addr = encode(["address"], [addr])  # 32 bytes
            fbb += encoded_addr.hex()[24:]  # take last 20 bytes in hex

        # Integers: full 32 bytes
        for val in v:
            fbb += encode(["int256"], [val]).hex()

        return fbb


    def quick_feedback_round(self, fbm, am = None, lm = None, prev_accs = None, prev_losses = None): # pragma: no cover
        print("Users exchanging feedback...")
        txs = []
        for idx, user in enumerate(self.pytorch_model.participants):
            if user.disqualified:
                break
            addrs = []
            votes = []
            user_votes = fbm[user.id]
            filtered_accs = []
            filtered_losses = []

            # Add null.check
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
                votee._roundrep.append(self.get_global_reputation_of_user(user.address) * int(vote))
                filtered_accs.append(accs[ix])
                filtered_losses.append(min(UINT256_MAX, losses[ix]))

            fbb = self.build_feedback_bytes(addrs, votes) # hex-encoded
            rb_fbb = Web3.to_bytes(hexstr="0x" + fbb)

            if self.experiment_config.contribution_score_strategy in [ "naive", "dotproduct" ]:
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytes(
                        rb_fbb
                    ).transact(tx)
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.model.functions.submitFeedbackBytes(rb_fbb)
                    )
                txs.append(tx_hash)

            elif self.experiment_config.contribution_score_strategy == "accuracy_loss":
                prev_acc = prev_accs[idx]
                prev_loss = prev_losses[idx]
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytesAndAccuraciesLosses(rb_fbb, filtered_accs, filtered_losses, prev_acc, prev_loss).transact(tx)
                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.model.functions.submitFeedbackBytesAndAccuraciesLosses(rb_fbb, filtered_accs, filtered_losses, prev_acc, prev_loss)
                    )
                txs.append(tx_hash)

            elif self.experiment_config.contribution_score_strategy == "accuracy_only":
                prev_acc = prev_accs[idx]
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytesAndAccuracies(rb_fbb, filtered_accs, prev_acc).transact(tx)

                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.model.functions.submitFeedbackBytesAndAccuracies(rb_fbb, filtered_accs, prev_acc)
                    )
                txs.append(tx_hash)

            elif self.experiment_config.contribution_score_strategy == "loss_only":
                prev_loss = prev_losses[idx]
                if self.fork:
                    tx = super().build_tx(user.address, self.modelAddress)
                    tx_hash = self.model.functions.submitFeedbackBytesAndLosses(rb_fbb, filtered_losses, prev_loss).transact(tx)

                else:
                    tx_hash = self.sign_and_send_tx(
                        user,
                        self.model.functions.submitFeedbackBytesAndLosses(rb_fbb, filtered_losses, prev_loss)
                    )
                txs.append(tx_hash)

            else:
                warnings.warn("INVALID FEEDBACK TYPE")

        for i, txHash in enumerate(txs):
            self.track_transaction(i, txHash, len(txs), "feedback")

        printer._print("                                                   ")
        print("\n-----------------------------------------------------------------------------------")


    def sign_and_send_tx(self, user, contract_fn_call): # pragma: no cover
        nonce = self.w3.eth.get_transaction_count(user.address)
        tx = super().build_non_fork_tx(user.address, nonce)
        tx = contract_fn_call.build_transaction(tx)

        signed = self.w3.eth.account.sign_transaction(tx, private_key=user.privateKey)
        return self.w3.eth.send_raw_transaction(signed.raw_transaction)


    # formerly named log_receipt
    def track_transaction(self, i, tx_hash, len_txs, receipt_type: str): # pragma: no cover
        #   1. Prints a progress bar — i out of len_txs transactions done
        #   2. Waits for the transaction to be mined — blocks until the receipt comes back (up to 600s timeout)
        #   3. Stores gas used — appends to self.gas_feedback
        #   4. Stores the tx hash + gas — appends to self.txHashes along with the receipt_type label (e.g. "feedback", "contrib")

        printer.print_bar(i, len_txs)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash,
                                                           timeout=600,
                                                           poll_latency=1)

        self.gas_feedback.append(receipt["gasUsed"])
        self.txHashes.append((receipt_type, receipt["transactionHash"].hex(), receipt["gasUsed"]))
        # Writer (old logger) uses this to log

        logging.log_receipt(self, receipt, receipt_type)
        # New logger log this way


    def send_fallback_transaction_onchain(self, _to, _from, data, private_key=None): # pragma: no cover
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


    def close_round(self): # pragma: no cover
        if "inactive" in [acc.attitude for acc in self.pytorch_model.participants]:
                input("Inactive users found - such users do not provide feedback.. " \
                          + "\nGoing to forward time for 1 day\n")
                self.w3.provider.make_request("evm_increaseTime", [self.config.WAIT_DELAY])

        settleStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < settleStart + config.get_contracts_config().FEEDBACK_ROUND_TIMEOUT):
            if (self.model.functions.isFeedBackRoundDone().call()):
                print("Feedback round completed")
                break
            print("Feedback round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            print("Feedback round failed, forcing Contribution...")

        contributionStart = datetime.datetime.now(datetime.timezone.utc).timestamp()
        while (datetime.datetime.now(datetime.timezone.utc).timestamp() < contributionStart + config.get_contracts_config().CONTRIBUTION_ROUND_TIMEOUT):
            if (self.model.functions.isContributionRoundDone().call()):
                print("Contribution round completed")
                break
            print("Contribution round not done, sleeping for 10 seconds...")
            time.sleep(10)
        else:
            print("Contribution round failed, forcing settlement...")


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

        self.txHashes.append(("close", receipt["transactionHash"].hex(), receipt["gasUsed"]))
        self.gas_close.append(receipt["gasUsed"])
        logging.log_receipt(self, receipt, "close")
        if len(receipt.logs) == 0:
            print("Warning: closeFeedBackRound() emitted no logs")
        self.pytorch_model.round += 1
        self._reward_balance.append(self.get_reward_left())
        printer._print("\n-----------------------------------------------------------------------------------\n")
        return receipt


    def user_register_slot(self): # pragma: no cover
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
            # print("{:<17}   {} | {} | {:>25,.0f} WEI".format("Slot registered: ",
            #                                                              acc.address[0:16] + "...",
            #                                                              txHash.hex()[0:6] + "...",
            #                                                              self.get_global_reputation_of_user(acc.address)
            #                                                              ))
        l = len(txs)
        for i, txHash in enumerate(txs):
            printer.print_bar(i, l)
            receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                            timeout=600, 
                                                            poll_latency=1)
            
            self.gas_slot.append(receipt["gasUsed"])
            self.txHashes.append(("slot", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            logging.log_receipt(self, receipt, "slot")
        # printer._print("-----------------------------------------------------------------------------------\n")
        return 
    

    def exit_system(self): # pragma: no cover
        self.pytorch_model.close_pool()
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
            self.txHashes.append(("exit", receipt["transactionHash"].hex(), receipt["gasUsed"]))
            logging.log_receipt(self, receipt, "exit")
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


    def print_round_summary(self, receipt, _current_round_no, contributors):
        # for user in self.pytorch_model.participants + self.pytorch_model.disqualified:
        #     user.temporary_grs_evaluation = None

        events = self.get_events(
            w3=self.w3,
            contract=self.model,
            receipt=receipt,
            event_names=["EndRound", "Reward", "Punishment", "ContributionPunishment", "PassivePunishment", "Disqualification"]
        )

        end_events = events["EndRound"]
        reward_events = events["Reward"]
        punish_events = events["Punishment"]
        contrib_punish_events = events["ContributionPunishment"]
        passive_punish_events = events["PassivePunishment"]
        disqualify_events = events["Disqualification"]

        # End of round summary
        # if end_events:
        #     for ev in end_events:
        #         args = ev["args"]
        #         print(b(f"\nEND OF ROUND      {args['round'] + 1}"))
        #         print(b(f"VALID VOTES:      {args['validVotes']}"))
        #         print(b(f"SUM OF WEIGHTS:   {args['sumOfWeightedContribScore']:,}"))
        #         print(b(f"TOTAL PUNISHMENT: {args['totalPunishment']:,}\n"))
        #     print("-----------------------------------------------------------------------------------\n")

        if passive_punish_events:
            print(b("PASSIVE PUNISHMENTS"))
            for ev in passive_punish_events:
                args = ev["args"]
                print(green(f"USER/VICTIM @    {args['victim']}"))
                print(green(f"ROUND SCORE:     {args['roundScore']:,}"))
                print(green(f"PUNISHED TARGET: {args['punishedTarget']}\n"))

        # if eval_reward_events:
        #     # print(b("EVALUATION VOTING REWARDS DISTRIBUTION"))
        #
        #     user_map = {u.address: u for u in contributors}
        #
        #     for ev in eval_reward_events:
        #         args = ev["args"]
        #         # print(green(f"USER @          {args['user']}"))
        #         # print(green(f"STAKED:         {args['staked']:,}"))
        #         # print(green(f"REWARDED:       {args['rewarded']:,}"))
        #         # print(green(f"NEW REPUTATION: {args['newReputation']:,}\n"))
        #
        #         user_map[args['user']].temporary_grs_evaluation = args['newReputation']
        #
        #     print("-----------------------------------------------------------------------------------\n")

        # Rewarded users
        if reward_events:
            # print(b("REWARDED USERS"))
            for ev in reward_events:
                args = ev["args"]
                if args["roundScore"] >= 0:
                    continue
                        # print(green(f"USER @            {args['user']}"))
                        # print(green(f"ROUND SCORE:      {args['roundScore']:,}"))
                        # print(green(f"TOTAL REWARD:     {args['win']:,}"))
                        # print(green(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
                else: warnings.warn(f"User {args['user']} had negative round score but was rewarded? Score: {args['roundScore']}, Reward: {args['win']}")
            print("-----------------------------------------------------------------------------------\n")

        # Punished users
        if punish_events:
            # print(b("RRS < 0 PUNISHED USERS FOR NOT GETTING MERGED"))
            for ev in punish_events:
                args = ev["args"]
                self._punishments.append((
                    _current_round_no,
                    args["loss"],
                    next((i + 1 for i, x in enumerate(self.pytorch_model.participants) if x.address == args["victim"]), 0),
                    ))
            #     print(red(f"USER @            {args['victim']}"))
            #     print(red(f"ROUND SCORE:      {args['roundScore']:,}"))
            #     print(red(f"TOTAL LOSS:       {args['loss']:,}"))
            #     print(red(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
            # print("-----------------------------------------------------------------------------------\n")

        if contrib_punish_events:
            # print(b("CONTRIBUTION-PUNISHED USERS"))
            for ev in contrib_punish_events:
                # print("Punishing a user for bad contribution")
                args = ev["args"]
                if args["roundScore"] >= 0:
                    continue
                    # print(green(f"USER @ {args['user']}"))
                    # print(green(f"ROUND SCORE:      {args['roundScore']:,}"))
                    # print(green(f"LOSS:             {args['loss']:,}"))
                    # print(green(f"NEW REPUTATION:   {args['newReputation']:,}\n"))
                else:
                    warnings.warn(
                        f"User {args['user']} had negative round score but was rewarded? Score: {args['roundScore']}, Reward: {args['win']}")
            # print("-----------------------------------------------------------------------------------\n")

        # Disqualified users
        if disqualify_events:
            print(b("DISQUALIFIED USERS"))
            for ev in disqualify_events:
                print("Disqualifying a user")
                args = ev["args"]
                self._punishments.append((
                    _current_round_no,
                    args["loss"],
                    next((i + 1 for i, x in enumerate(self.pytorch_model.participants) if x.address == args["victim"]), 0)),
                    )

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

        logging.log_punishments(self, events, _current_round_no)
        # logging.log_evaluation_voting_rewards(self, events, _current_round_no)

        # # round grs summary print
        # print(b(f"Round {_current_round_no} completed:"))
        # print(b("Round Rewards (per user):"))
        # print(b("{:>20}  {:>25} -> {:>25} -> {:>25}".format("address" + "...", "previous grs",
        #                                                     "evaluation votes grs",
        #                                                     "final grs")))
        # for user in self.pytorch_model.participants + self.pytorch_model.disqualified:
        #     user._globalrep.append(self.get_global_reputation_of_user(user.address))
            # eval_grs = user.temporary_grs_evaluation
            # if eval_grs is None:
            #     j = "NO EVAL GRS (NOT MERGED)"
            # else:
            #     j = f"{eval_grs:,.0f}"
            # i, k = user._globalrep[-2:]
            # print(b("{:>20}  {:>25,.0f} -> {:>25} -> {:>25,.0f}".format(user.address[0:16] + "...", i, j, k)))


    def get_round_rewards(self, receipt): # pragma: no cover
        events = self.get_events(
            w3=self.w3,
            contract=self.model,
            receipt=receipt,
            event_names=["Reward"]
        )
        reward_events = events["Reward"]
        
        result = []
        for ev in reward_events:
            args = ev["args"]
            if args["roundScore"] >= 0:
                result.append(
                    (
                        args["user"],
                        args["roundScore"],
                        args["win"], # Reward/Punishment
                        args["newReputation"], # New global reputation after reward/punishmens
                    )
                )
        return result


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

        print(self.modelAddress)
        self.register_all_users()
        
        grs = [(user.address, user._globalrep[-1]) for user in self.pytorch_model.participants + self.pytorch_model.disqualified]

        _current_round = self.pytorch_model.round

        roundTx = self.txHashes[self.writeTxProgress:]
        self.writeTxProgress = len(self.txHashes)

        self.writer.writeResult({
                "round": 0,
                "GRS": grs,
                "globalAcc": self.pytorch_model.accuracy[-1] or 0,
                "globalLoss": self.pytorch_model.loss[-1] or 0,
                "conctractBalanceRewards": self._reward_balance[-1],
                "punishments": [],
                "rewards": [],
                "accAvgPerUser": [],
                "lossAvgPerUser": [],
                "feedbackMatrix": None,
                "disqualifiedUsers": [],
                "contributionScores": [],
                "userStatuses": [user.getStatus() for user in self.pytorch_model.participants],
                "GasTransactions": roundTx
            })

        logging.log_round_zero(self)
        try:
            for i in range(rounds):
                print(b(f"\n\nRound {_current_round} starts..."))
                _round_start = time.perf_counter()

                attacks.update_users_attitude(self.pytorch_model)

                self.pytorch_model.federated_training()

                attacks.let_malicious_users_do_their_work(self.pytorch_model)

                attacks.let_freerider_users_do_their_work(self.pytorch_model)

                self.user_register_slot()

                self.users_provide_hashed_weights()

                evaluation.exchange_models(self.pytorch_model)

                evaluation.verify_models(self.pytorch_model, {u.id: self.get_hashed_weights_of(u) for u in self.pytorch_model.participants})

                self.feedback_matrix, accuracy_matrix, loss_matrix, prev_accs, prev_losses = evaluation.evaluate_peers(self.pytorch_model)

                self.quick_feedback_round(fbm = self.feedback_matrix, am=accuracy_matrix, lm=loss_matrix, prev_accs=prev_accs, prev_losses=prev_losses)

                for user in self.pytorch_model.participants:
                    user._roundrep.append(self.get_round_reputation_of_user(user.address))
                    print(f"model participant: {user.address} gets {user._roundrep[-1]} round reputation")
                for user in self.pytorch_model.disqualified:
                    print(f"disqualified model participant: {user.address} has no round reputation, as he is disqualified")

                # A roundRep of 0, does not nec. mean mal.
                contributors = [user for user in self.pytorch_model.participants if user._roundrep[-1] >= 0] # Keeps track of who will be merged in the_merge()
                if len(contributors) == 0: # If all are negative, we merge everyone and let the contribution score calculation sort them out.
                    contributors = self.make_everyone_contributors()

                users_weight_collector = {}
                agg_switch_collector = {}
                warning_collector = []


                # # Ordering of the merge. If dotproduct we merge before contribution score
                # if self.experiment_config.contribution_score_strategy == "dotproduct":
                #     aggregation.the_merge(self.pytorch_model, _current_round, contributors, aggregation_rule=self.experiment_config.aggregation_rule, merge_weight_collector=users_weight_collector, agg_switch_collector=agg_switch_collector, warning_collector=warning_collector)
                #     for msg in warning_collector:
                #         logging.log_warning(self, msg, round=_current_round)

                aggregation.the_merge(self.pytorch_model, _current_round, contributors, warning_collector=warning_collector)

                print(b("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n"))
                contribution_score(self, contributors, _current_round)

                receipt = self.close_round() # Increments round number by 1
                _current_round = self.pytorch_model.round - 1 # Minus 1 since close_round increments. Reassign _current_round

                # # If not dotproduct, we calculate contribution score before the merge
                # if not self.experiment_config.contribution_score_strategy == "dotproduct":
                #     avg_losses = self.get_all_n_prior_losses(3)
                #     aggregation.the_merge(self.pytorch_model, _current_round, contributors, aggregation_rule=self.experiment_config.aggregation_rule, merge_weight_collector=users_weight_collector, agg_switch_collector=agg_switch_collector, avg_prior_losses=avg_losses, warning_collector=warning_collector)
                #     for msg in warning_collector:
                #         logging.log_warning(self, msg, round=_current_round)

                if receipt is not None:
                    self.print_round_summary(receipt, _current_round, contributors)

                _round_time = time.perf_counter() - _round_start

                logging.log_round(self,
                    _current_round, _round_time,
                    accuracy_matrix, loss_matrix, prev_accs, prev_losses,
                    contributors, receipt, users_weight_collector, agg_switch_collector,
                )

                grs = [(user.address, user._globalrep[-1]) for user in self.pytorch_model.participants + self.pytorch_model.disqualified]
                round_punishment = [(punishment[0], punishment[1]) for punishment in self._punishments if punishment[0] == _current_round]
                round_kicked = [punishment[2] for punishment in self._punishments if punishment[0] == _current_round]
                roundTx = self.txHashes[self.writeTxProgress:]
                self.writeTxProgress = len(self.txHashes) - 1
                self.writer.writeResult({
                    "round": _current_round,
                    "GRS": grs,
                    "globalAcc": self.pytorch_model.accuracy[-1] or 0, # Checks out
                    "globalLoss": self.pytorch_model.loss[-1] or 0, # Checks out
                    "conctractBalanceRewards": self._reward_balance[-1],
                    "punishments": round_punishment,
                    "rewards": self.get_round_rewards(receipt),
                    "accAvgPerUser": prev_accs, # Check - Should come from am
                    "lossAvgPerUser": prev_losses, # Check - Should come from lm
                    "feedbackMatrix": self.feedback_matrix.tolist(),
                    "disqualifiedUsers": round_kicked,
                    "contributionScores": self.scores,
                    "userStatuses": [user.getStatus() for user in self.pytorch_model.participants],
                    "GasTransactions": roundTx
                    })

                _current_round = self.pytorch_model.round # Update current round to match with incremented round in close_round()


            # print(f"Number of Shapley Axioms violated: {len(contribution.runtime_warnings)}\n")
            # if contribution.runtime_warnings:
            #     print("\n" + red("!" * 30 + " SHAPLEY WARNINGS " + "!" * 30))
            #     for warn in contribution.runtime_warnings:
            #         print(colored(warn, 'yellow'))
            #     print(red("!" * 78))
            contribution.print_shapley_warnings()

            self.writer.writeComment(f"$gasCosts${self.gas_feedback},{self.gas_register},{self.gas_slot},{self.gas_weights},{self.gas_close},{self.gas_deploy},{self.gas_exit}")
            self.exit_system()
        finally:
            self.pytorch_model.shutdown()


    
    def visualize_simulation(self, output_folder_path): # pragma: no cover
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
        for i, j, y in self._punishments:
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
                                 + f'users={self.pytorch_model.NUMBER_OF_CONTRIBUTORS}\n' \
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


    def make_everyone_contributors(self):
        msg = "All users had negative round reputation - merging all users and letting contribution score calculation sort them out."
        print(rb(msg))
        logging.log_warning(challenge=self, msg=msg, round=self.pytorch_model.round)
        contributors = [user for user in self.pytorch_model.participants]

        if self.fork:
            tx = super().build_tx(self.w3.eth.default_account, self.modelAddress, 0)
            tx_hash = self.model.functions.makeRoundReputationsPositive().transact(tx)
        else:
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address, 'pending')
            cl = super().build_non_fork_tx(self.pytorch_model.participants[0].address, nonce)
            cl = self.model.functions.makeRoundReputationsPositive().build_transaction(cl)
            pk = self.pytorch_model.participants[0].privateKey
            signed = self.w3.eth.account.sign_transaction(cl, private_key=pk)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash,
                                                           timeout=600,
                                                           poll_latency=1)
        print("All round reputations set to positive")

        self.txHashes.append(("makeRoundRepsPositive", receipt["transactionHash"].hex(), receipt["gasUsed"]))
        self.gas_close.append(receipt["gasUsed"])
        logging.log_receipt(self, receipt, "makeRoundRepsPositive")

        return contributors