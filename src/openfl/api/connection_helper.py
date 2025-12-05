import os
from pathlib import Path
import re
import os
import time
import signal
import numpy as np
import pandas as pd
from web3 import Web3
from termcolor import colored
from subprocess import Popen, PIPE
from openfl.ml.pytorch_model import gb, rb, b, green, red
from openfl.utils import require_env_var

class ConnectionHelper:
    # Start Ganache client with connection to infura
    # Create web3 instance
    # Recursive function used to first get the latest block and then
    # ...fork the chain latest possible
    def initiate_rpc(self, 
                         NUMBER_OF_GOOD_CONTRIBUTORS, 
                         NUMBER_OF_BAD_CONTRIBUTORS, 
                         NUMBER_OF_FREERIDER_CONTRIBUTORS, 
                         NUMBER_OF_INACTIVE_CONTRIBUTORS,
                         MINIMUM_ROUNDS,
                         pytorch_model,
                         latestBlock=1000000, 
                         infura_url=None, 
                         manual_setup=False,
                         fork=True,
                         accounts=None):
        global w3
        NUMBER_OF_CONTRIBUTORS = NUMBER_OF_GOOD_CONTRIBUTORS \
                                    + NUMBER_OF_BAD_CONTRIBUTORS \
                                    + NUMBER_OF_FREERIDER_CONTRIBUTORS \
                                    + NUMBER_OF_INACTIVE_CONTRIBUTORS
        infura_url = require_env_var("RPC_URL")





        if fork:
            if not manual_setup:
                port = require_env_var("RPC_URL").split(':')[1]
                process = Popen(["lsof", "-i", ":{0}".format(port)], stdout=PIPE, stderr=PIPE)
                stdout, stderr = process.communicate()
                for process in str(stdout.decode("utf-8")).split("\n")[1:]:       
                    data = [x for x in process.split(" ") if x != '']
                    if (len(data) <= 1):
                        continue

                    os.kill(int(data[1]), signal.SIGKILL)
                command = "ganache --fork.url='{}' -a {} -b 10".format(infura_url, NUMBER_OF_CONTRIBUTORS)
                os.system("gnome-terminal -e 'bash -c \"{}; bash\" '".format(command))
        while latestBlock == 1000000:
            time.sleep(1)
            try:
                if fork:
                    w3 = Web3(Web3.HTTPProvider(infura_url))
                    print("Connected:", w3.is_connected())
                    print("Client:", w3.client_version)
                    print("Chain ID:", w3.eth.chain_id)
                    print("Latest block:", w3.eth.block_number)
                    print("Accounts:", w3.eth.accounts[:3])
                    print("Default account:", w3.eth.default_account)
                    w3.eth.default_account = w3.eth.accounts[0]
                    print("New Default account:", w3.eth.default_account)

                else:
                    w3 = Web3(Web3.HTTPProvider(infura_url))
                latestBlock = w3.eth.block_number
            except:
                latestBlock = 1000000
        
        
        #print("\n==================================================================================\n")
        print("Connected to Ethereum: {}".format(colored(w3.is_connected(), "green", attrs=['bold'])))
        print("initiated Ganache-Client @ Block Nr. {:,.0f}\n".format(latestBlock))        
        print("Total Contributers:       {}".format(NUMBER_OF_CONTRIBUTORS))
        print("Good Contributers:        {} ({:.0f}%)".format(NUMBER_OF_GOOD_CONTRIBUTORS,
                                                        NUMBER_OF_GOOD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100)) 
        print("Malicious Contributers:   {} ({:.0f}%)".format(NUMBER_OF_BAD_CONTRIBUTORS,
                                                        NUMBER_OF_BAD_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Freeriding Contributers:  {} ({:.0f}%)".format(NUMBER_OF_FREERIDER_CONTRIBUTORS,
                                                        NUMBER_OF_FREERIDER_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Inactive Contributers:    {} ({:.0f}%)".format(NUMBER_OF_INACTIVE_CONTRIBUTORS,
                                                        NUMBER_OF_INACTIVE_CONTRIBUTORS/NUMBER_OF_CONTRIBUTORS*100 )) 
        print("Learning Rounds:          {}".format(MINIMUM_ROUNDS)) 
        
        print("-----------------------------------------------------------------------------------")
        
        if fork:
            while not w3.eth.default_account:
                time.sleep(0.2)
                try:
                    w3.eth.default_account = w3.eth.accounts[0]
                except:
                    w3.eth.default_account = None
            
            if len(w3.eth.accounts) < len(self.pytorch_model.participants):
                print(rb("Nr. of Ganache Addresses <> Nr. of Model Participants"))
                print(rb(str(len(w3.eth.accounts))  + "<>" +  str(len(self.pytorch_model.participants))))
                print(rb("Increase number of unlocked accounts"))
                raise NotEnoughUnlockedAccounts()
                
        # Every user receives an address
        for ix in range(len(self.pytorch_model.participants)):
            if fork:
                self.pytorch_model.participants[ix].address = w3.to_checksum_address(w3.eth.accounts[ix])
            else:
                if ix == 0:
                    w3.eth.default_account = accounts[ix].address 
                self.pytorch_model.participants[ix].address = w3.to_checksum_address(accounts[ix].address)
                self.pytorch_model.participants[ix].privateKey = accounts[ix].privateKey           
                
            
        for i, acc in enumerate(self.pytorch_model.participants):
            if acc.futureAttitude == "good":
                prefix = "FAIR"
            elif acc.futureAttitude == "freerider":
                prefix = "FREE"
            elif acc.futureAttitude == "inactive":
                prefix = "AFK "
            else:
                prefix = "MAL."
            bal = w3.eth.get_balance(acc.address)
            print("{:<17} {} with {:<4,.1f} ETH | {} USER".format("Account initiated", 
                                                           "@ Address "+acc.address[0:25]+"...",
                                                           bal/1e18,
                                                           prefix))
        print("-----------------------------------------------------------------------------------")
        self.w3 = w3
        return w3, latestBlock

    def get_w3():
        return w3
    
    def initialize(self):
        bytecode_path = Path(__file__).resolve().parents[3] / "artifacts" / "bytecode"
        with open(bytecode_path / "abi.txt") as abiFile:
            abi = re.sub("\n|\t|\ ", "", abiFile.read())
        with open(bytecode_path /  "bytecode.txt") as abiFile:
            bytecode = abiFile.read().strip()
        return self.w3.eth.contract(bytecode=bytecode, abi=abi)
    
    
    
    def initialize_model(self, address):
        bytecode_path = Path(__file__).resolve().parents[3] / "artifacts" / "bytecode"
        with open(bytecode_path / "abi_model.txt") as abiFile:
            abi = re.sub("\n|\t|\ ", "", abiFile.read())
        with open(bytecode_path / "bytecode_model.txt") as abiFile:
            bytecode = abiFile.read().strip()
        return self.w3.eth.contract(address=address, bytecode=bytecode, abi=abi)
    
    
    
    def build_tx(self, _from, _to, _value=0):
        assert(_to != "0x0000000000000000000000000000000000000000")
        _from = w3.to_checksum_address(_from)
        _to = w3.to_checksum_address(_to)
        return {
            'from': _from,
            'to': _to,
            'value': _value,
            #'gas': 300000,
            #'maxFeePerGas': self.w3.to_wei(250, 'gwei'),
            #'maxPriorityFeePerGas': self.w3.to_wei(5, 'gwei'),
        }
    
    
    
    def build_non_fork_tx(self, addr, nonce, to=None, value=0, data=None, gas_limit=None):
        # Dynamically detect correct chain ID
        chain_id = w3.eth.chain_id

        # Give on-chain deployments breathing room unless caller overrides
        if gas_limit is None:
            gas_limit = 5_000_000

        # Adaptive low gas fee settings
        max_fee_per_gas = w3.to_wei(10, 'gwei')
        max_priority_fee_per_gas = w3.to_wei(1, 'gwei')

        # Check balance before building TX
        balance = w3.eth.get_balance(addr)
        est_cost = gas_limit * max_fee_per_gas
        if balance < est_cost:
            print(f"\n Warning: Account {addr} has only {w3.from_wei(balance, 'ether'):.4f} ETH, "
                f"but may need {w3.from_wei(est_cost, 'ether'):.4f} ETH for gas.\n")

        # Build TX (same structure as before)
        if data:
            return {
                'chainId': chain_id,
                'from': addr,
                'to': to,
                'gas': gas_limit,
                'maxFeePerGas': max_fee_per_gas,
                'maxPriorityFeePerGas': max_priority_fee_per_gas,
                'nonce': nonce,
                'value': value,
                'data': data
            }

        if to:
            return {
                'chainId': chain_id,
                'from': addr,
                'to': to,
                'gas': gas_limit,
                'maxFeePerGas': max_fee_per_gas,
                'maxPriorityFeePerGas': max_priority_fee_per_gas,
                'nonce': nonce,
                'value': value
            }

        # Default case (no 'to' address)
        return {
            'chainId': chain_id,
            'from': addr,
            'gas': gas_limit,
            'maxFeePerGas': max_fee_per_gas,
            'maxPriorityFeePerGas': max_priority_fee_per_gas,
            'nonce': nonce,
            'value': value
        }
        
class NotEnoughUnlockedAccounts(Exception):
    pass