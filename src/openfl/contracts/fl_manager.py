from web3 import Web3
from openfl.ml.pytorch_model import gb, rb, b, green, red
from openfl.api import ConnectionHelper

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
        if self.fork:
            genesisHash = self.manager.constructor().transact()  # Build Contract
        else:
            nonce = self.w3.eth.get_transaction_count(self.w3.eth.default_account) 
            depl = super().build_non_fork_tx(self.w3.eth.default_account, nonce)   
            depl = self.manager.constructor().buildTransaction(depl)
            signed = self.w3.eth.account.signTransaction(depl, private_key=self.pytorch_model.participants[0].privateKey)

            genesisHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
        receipt = self.w3.eth.wait_for_transaction_receipt(genesisHash,
                                                        timeout=600, 
                                                        poll_latency=1)
        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildManager", receipt["transactionHash"].hex()))
        
        self.manager.address = receipt.contractAddress
        print("\n{:<17} {} | {}\n".format("Manager deployed", 
                                          "@ Address " + self.manager.address, 
                                          genesisHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        return 
    
    
    
    def get_model_of(self, p, c):
        return self.manager.functions.ModelOf(p.address, c).call({"to": self.manager.address,
                                                                  "from": p.address})
    
    
    def get_model_count_of(self, p):
        return self.manager.functions.ModelCountOf(p.address).call({"to": self.manager.address,
                                                                  "from": p.address})
    
    
    def deploy_challenge_contract(self, *args):
        print(b("Starting simulation..."))
        print(b("-----------------------------------------------------------------------------------"))
        min_buyin, max_buyin, reward, min_rounds, punishment, freerider_fee = args
        p1_collateral = self.pytorch_model.participants[0].collateral
        value = reward + p1_collateral
        deployer =  self.pytorch_model.participants[0].address
        modelHash = self.pytorch_model.participants[0].modelHash
        model_hash_bytes = Web3.to_bytes(hexstr=modelHash)
        if self.fork:
            tx = super().build_tx(deployer, self.manager.address, value)
            txHash = self.manager.functions.deployModel(model_hash_bytes, #change!
                                                        min_buyin, 
                                                        max_buyin, 
                                                        reward,
                                                        min_rounds,
                                                        punishment,
                                                        freerider_fee).transact(tx)
        else:          
            nonce = self.w3.eth.get_transaction_count(self.pytorch_model.participants[0].address) 
            depl = super().build_non_fork_tx(deployer, nonce, self.manager.address, value)   
            depl = self.manager.functions.deployModel(modelHash,
                                                      min_buyin, 
                                                      max_buyin, 
                                                      reward,
                                                      min_rounds,
                                                      punishment,
                                                      freerider_fee).buildTransaction(depl)
            signed = self.w3.eth.account.signTransaction(depl, private_key=self.pytorch_model.participants[0].privateKey)
            txHash = self.w3.eth.sendRawTransaction(signed.rawTransaction)
            
            
        receipt = self.w3.eth.wait_for_transaction_receipt(txHash,
                                                        timeout=600, 
                                                        poll_latency=1)

        self.gas_deploy.append(receipt["gasUsed"])
        self.txHashes.append(("buildChallenge", receipt["transactionHash"].hex()))
        c = self.get_model_count_of(self.pytorch_model.participants[0])
        address = self.get_model_of(self.pytorch_model.participants[0], c)
        
        self.challenge_contract = super().initialize_model(address)
        print("\n{:<17} {} | {}\n".format("Model deployed", 
                                          "@ Address " + self.challenge_contract.address, 
                                          txHash.hex()[0:6]+"..."))
        print("-----------------------------------------------------------------------------------")
        print("{:<17} {} | {} | {:>25,.0f} WEI".format("Account registered:", 
                                                           self.pytorch_model.participants[0].address[0:16] + "...", 
                                                           txHash.hex()[0:6] + "...", 
                                                           p1_collateral
                                                           ))

        self.pytorch_model.participants[0].isRegistered = True
        self.model_address = self.challenge_contract.address
        return (self.challenge_contract, self.challenge_contract.address) + args