from web3 import Web3
from openfl.ml.pytorch_model import gb, rb, b, green, red
from openfl.api import ConnectionHelper
from artifacts.bytecode.abi_model import OPEN_FL_MODEL_ABI  # import the ABI directly

class FLManager(ConnectionHelper):
    
    def __init__(self, pytorch_model, manual_ganache_setup=False):
        self.w3 = None
        self.latestBlock = None
        self.manager = None
        self.challenge_contract = None
        self.pytorch_model = pytorch_model
        self.modelOf = {}
        self.manual_setup = manual_ganache_setup
        
        self.gas_deploy = []
        self.txHashes   = []
        
    
    def init(self, 
             NUMBER_OF_GOOD_CONTRIBUTORS, 
             NUMBER_OF_BAD_CONTRIBUTORS, 
             NUMBER_OF_FREERIDER_CONTRIBUTORS, NUMBER_OF_INACTIVE_CONTRIBUTORS, 
             MINIMUM_ROUNDS, 
             infuraurl=None, 
             fork=True,
             accounts=None): 
        
        self.fork = fork
        self.w3, self.latestBlock = super().initiate_rpc(NUMBER_OF_GOOD_CONTRIBUTORS=NUMBER_OF_GOOD_CONTRIBUTORS,
                                                         NUMBER_OF_BAD_CONTRIBUTORS=NUMBER_OF_BAD_CONTRIBUTORS,
                                                         NUMBER_OF_FREERIDER_CONTRIBUTORS=NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                         NUMBER_OF_INACTIVE_CONTRIBUTORS=NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                         MINIMUM_ROUNDS=MINIMUM_ROUNDS, pytorch_model=self.pytorch_model,
                                                         infura_url=infuraurl, manual_setup=self.manual_setup, fork=fork,
                                                         accounts=accounts)
        self.manager = super().initialize()
        return self
    
    
    # Deploy contract and initiate proxy
    def build_contract(self):
        manager_abi = self.manager.abi

        if self.fork:
            genesisHash = self.manager.constructor().transact()  # Build Contract
        else:
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account) 
            depl = super().build_non_fork_tx(self.w3.eth.default_account, nonce)   
            depl = self.manager.constructor().build_transaction(depl)
            signed = self.w3.eth.account.sign_transaction(depl, private_key=self.pytorch_model.participants[0].privateKey)

            genesisHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            
        receipt = self.w3.eth.wait_for_transaction_receipt(genesisHash,
                                                           timeout=600,
                                                           poll_latency=1)
        if receipt.get("status", 0) != 1:
            raise RuntimeError(
                f"Manager deployment failed (tx={genesisHash.hex()}, status={receipt.get('status')}). "
                "Check Sepolia gas settings and account balance."
            )

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex()))

        deployed_address = self.w3.to_checksum_address(receipt.contractAddress)
        self.manager = self.w3.eth.contract(address=deployed_address, abi=manager_abi)

        print("\n{:<17} {} | {}\n".format("Manager deployed", 
                                          "@ Address " + self.manager.address, 
                                          genesisHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        return 


    
    
    def get_model_of(self, p, c):
        return self.manager.functions.ModelOf(p.address, c).call({"to": self.manager.address,
                                                                  "from": p.address})
    
    
    def get_model_count_of(self, p):
        print(f"[DEBUG] get_model_count_of(): manager.address={self.manager.address}, "
              f"participant.address={p.address}, "
              f"challenge.address={getattr(self, 'challenge_contract', None) and getattr(self.challenge_contract, 'address', None)}")

        # ModelCountOf is exposed on the manager contract ABI
        return self.manager.functions.ModelCountOf(p.address).call({
            "from": p.address
        })


    
    
    def deploy_challenge_contract(self, *args):
        print(b("Starting simulation..."))
        print(b("-----------------------------------------------------------------------------------"))
        min_buyin, max_buyin, reward, min_rounds, punishment, freerider_fee = args
        p1_collateral = self.pytorch_model.participants[0].collateral
        value = reward + p1_collateral
        deployer = self.pytorch_model.participants[0].address
        modelHash = self.pytorch_model.participants[0].modelHash
        model_hash_bytes = Web3.to_bytes(hexstr=modelHash)

        # 💡 Helpful debug info
        balance_eth = self.w3.from_wei(self.w3.eth.get_balance(deployer), 'ether')
        est_cost_eth = self.w3.from_wei(value, 'ether')
        print(f"Balance: {balance_eth:.4f} ETH | Estimated tx+value cost: {est_cost_eth:.4f} ETH")

        if self.fork:
            tx = super().build_tx(deployer, self.manager.address, value)
            txHash = self.manager.functions.deployModel(
                model_hash_bytes,
                min_buyin,
                max_buyin,
                reward,
                min_rounds,
                punishment,
                freerider_fee
            ).transact(tx)
        else:
            nonce = self.w3.eth.get_transaction_count(deployer)
            # When building the transaction via contract ABI we must not pre-set the `to` field.
            depl = super().build_non_fork_tx(deployer, nonce, value=value)
            depl = self.manager.functions.deployModel(
                model_hash_bytes,
                min_buyin,
                max_buyin,
                reward,
                min_rounds,
                punishment,
                freerider_fee
            ).build_transaction(depl)
            signed = self.w3.eth.account.sign_transaction(depl, private_key=self.pytorch_model.participants[0].privateKey)
            txHash = self.w3.eth.send_raw_transaction(signed.raw_transaction)

        receipt = self.w3.eth.wait_for_transaction_receipt(txHash, timeout=600, poll_latency=1)
        if receipt.get("status", 0) != 1:
            raise RuntimeError(
                f"Challenge deployment failed (tx={txHash.hex()}, status={receipt.get('status')}). "
                "Contract creation likely ran out of gas or reverted. "
                "Recheck reward/buy-in sizing and Sepolia balances."
            )

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildChallenge", receipt["transactionHash"].hex()))
        c = self.get_model_count_of(self.pytorch_model.participants[0])
        deployed_address = self.get_model_of(self.pytorch_model.participants[0], c)
        deployed_address = Web3.to_checksum_address(deployed_address)

        self.challenge_contract = self.w3.eth.contract(
            address=deployed_address,
            abi=OPEN_FL_MODEL_ABI
        )
        print("\n{:<17} {} | {}\n".format("Model deployed", 
                                          "@ Address " + self.challenge_contract.address, 
                                          txHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        print("{:<17} {} | {} | {:>25,.0f} WEI".format(
            "Account registered:",
            self.pytorch_model.participants[0].address[0:16] + "...",
            txHash.hex()[0:6] + "...",
            p1_collateral
        ))

        self.pytorch_model.participants[0].isRegistered = True
        return (self.challenge_contract, self.challenge_contract.address) + args

