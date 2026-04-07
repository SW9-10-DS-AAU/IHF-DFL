import copy
import sys
import warnings
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
import time
import math
import logging
from web3 import Web3
from termcolor import colored
from typing import Tuple, Dict
from collections import OrderedDict
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader, random_split, Subset
from collections import Counter
torch._dynamo.config.cache_size_limit = 512
debugging = sys.gettrace() is not None
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
RNG = np.random.default_rng()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = (DEVICE.type == "cuda")
PIN_MEMORY = USE_CUDA
NON_BLOCKING = USE_CUDA
NUM_WORKERS = min(4, os.cpu_count() // 2) if torch.cuda.is_available() else 0
PERSISTENT_WORKERS = USE_CUDA and NUM_WORKERS > 0
AMP = USE_CUDA # Optional: mixed precision on CUDA

# cuDNN autotune for fixed-size inputs (both MNIST 28x28 and CIFAR-10 32x32)
torch.backends.cudnn.benchmark = USE_CUDA
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

def model_to_device(net: nn.Module) -> nn.Module:
    # Move model once; keep it on the chosen device
    return net.to(DEVICE, non_blocking=NON_BLOCKING)

def cuda_safe_dataloader(ds, batch_size, shuffle=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
    )


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bad_c  = "#d62728"
free_c = "#9467bd"
colors.remove(bad_c)
colors.remove(free_c)

class Participant:
    def __init__(self, _id, _train, _val, _model, _optimizer, _criterion,
                 _attitude, _default_collateral, _max_collateral,
                 _attitudeSwitch=1, number_of_participants=None):
        self.id = _id
        self.train = _train
        self.val  = _val
        self.model = _model
        self.previousModel = copy.deepcopy(_model)
        self.modelHash = Web3.solidity_keccak(['string'],[str(_model)]).hex()
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.userToEvaluate = []
        self.currentAcc = 0
        # User's locally-trained model accuracy on their own validation set (after they trained on top of the global model).
        # Is set in: apply_training_results().
        self.currentLoss = 0
        # New variable introduced. Needs to be implemented in code. Alongside currentAcc.
        self.attitude = "good"
        self.futureAttitude = _attitude
        self.attitudeSwitch = _attitudeSwitch
        self.hashedModel = None
        self.address = None
        self.privateKey = None
        self.isRegistered = False
        # Old:  self.collateral = _default_collateral + np.random.randint(0,int(_max_collateral-_default_collateral))
        # ---- collateral (handles huge ranges; avoids int32 cap) ----
        lo = int(_default_collateral)
        hi = int(_max_collateral)
        if hi < lo:
            raise ValueError(f"max_collateral ({hi}) must be >= default_collateral ({lo})")

        diff = hi - lo
        jitter = int(RNG.integers(0, np.int64(diff), dtype=np.int64)) if diff > 0 else 0
        self.collateral = lo + jitter

        # ---- secret (big nonce) ----
        self.secret = int(RNG.integers(0, np.int64(10 ** 18), dtype=np.int64))
        # self.secret = np.random.randint(0,int(1e18))

        self.color = get_color(number_of_participants, self.attitude)
        self.roundRep = 0

        self.disqualified = False

        # INTERFACE VARIABLES - Not used for training. Only for reporting.
        self._accuracy = [] # User's accuracy on the global model. The actual accuracy evaluated on test set - is set in: finalize_user_evaluation().
        self._loss = [] # User's loss on the global model. The actual loss evaluated on test set - is set in: finalize_user_evaluation().
        self._globalrep = [self.collateral]
        self._roundrep = []
    
    def getStatus(self):
        user = f"$user${self.id}, {self.currentAcc}, {self.attitude}, {self.futureAttitude}, {self.attitudeSwitch}, {self.address}"
        return user
          
class Net_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Net_MNIST(nn.Module):
    def __init__(self):
        super(Net_MNIST, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # feature map size is 14*14 by pooling
        # padding=2 for same padding
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(64*7*7, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 64*7*7)   # reshape Variable
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return F.log_softmax(x)
        return x


        
class PytorchModel:
    def __init__(self, DATASET, _good_participants, _bad_participants, _freerider_participants, epochs, batchsize, default_collateral, max_collateral, freerider_noise_scale: float = 1.0, freerider_start_round: int = 3, malicious_start_round: int = 3, malicious_noise_scale: float = 1.0,force_merge_all: bool = False, use_nobody_is_kicked: bool = False, data_distribution : str = None, dirichlet_alpha: float = None):
        self.DATASET = DATASET
        if self.DATASET == "mnist":
            self.global_model = Net_MNIST().to(DEVICE)
        else:
            self.global_model = Net_CIFAR().to(DEVICE)

        self.NUMBER_OF_GOOD_CONTRIBUTORS = _good_participants
        self.NUMBER_OF_BAD_CONTRIBUTORS = _bad_participants
        self.NUMBER_OF_FREERIDER_CONTRIBUTORS = _freerider_participants
        self.NUMBER_OF_INACTIVE_CONTRIBUTORS = 0
        self.NUMBER_OF_CONTRIBUTORS = _good_participants + _bad_participants + _freerider_participants + self.NUMBER_OF_INACTIVE_CONTRIBUTORS
        self.DATA = None
        self.participants = []
        self.disqualified = []
        self.EPOCHS = epochs
        self.BATCHSIZE = batchsize

        if data_distribution is None:
            self.data_distribution = "random_split_42"
        else:
            self.data_distribution = data_distribution

        if dirichlet_alpha is None:
            self.dirichlet_alpha = 0.5
        else:
            self.dirichlet_alpha = dirichlet_alpha

        self.train, self.val, self.test = self.load_data(self.NUMBER_OF_CONTRIBUTORS, _print=True)
        self.default_collateral = default_collateral
        self.max_collateral = max_collateral
        self.force_merge_all = force_merge_all
        self.use_nobody_is_kicked = use_nobody_is_kicked
        self.has_switched = False
        self.agg_func_1 = None
        self.agg_weight_1 = None
        self.agg_func_2 = None
        self.agg_weight_2 = None


        if freerider_noise_scale < 0:
            raise ValueError("freerider_noise_scale must be non-negative")
        self.freerider_noise_scale = freerider_noise_scale

        if freerider_start_round < 1:
            raise ValueError("freerider_start_round must be at least 1")
        self.freerider_start_round = freerider_start_round

        if malicious_start_round < 1:
            raise ValueError("malicious_start_round must be at least 1")
        self.malicious_start_round = malicious_start_round

        if malicious_noise_scale < 0:
            raise ValueError("malicious_noise_scale must be non-negative")
        self.malicious_noise_scale = malicious_noise_scale

        loss, accuracy = test(self.global_model,self.test,DEVICE)
        
        # INTERFACE VARIABLES
        self.accuracy = [accuracy]
        self.loss = [loss]

        self.round = 1
        self.previous_global_model = None
        self.two_previous_global_model = None
        print("===================================================================================")
        print("Pytorch Model created:\n")
        print(str(self.global_model))
        print("\n===================================================================================")

        for i in range(self.NUMBER_OF_GOOD_CONTRIBUTORS):
            self.add_participant("good")

        for i in range(self.NUMBER_OF_BAD_CONTRIBUTORS):
            self.add_participant("bad")

        for i in range(self.NUMBER_OF_FREERIDER_CONTRIBUTORS):
            self.add_participant("freerider")

        for i in range(self.NUMBER_OF_INACTIVE_CONTRIBUTORS):
            self.add_participant("inactive")

            
    def add_participant(self, _attitude):
        _train, _val, _test = self.load_data(self.NUMBER_OF_CONTRIBUTORS)
        
        if self.DATASET == "mnist":
            _model = Net_MNIST().to(DEVICE)
        else:
            _model = Net_CIFAR().to(DEVICE)
            
        optimizer = optim.SGD(_model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()


        if _attitude == "good":
            self.NUMBER_OF_GOOD_CONTRIBUTORS +=1
            _attitudeSwitch = None
        elif _attitude == "bad":
            self.NUMBER_OF_BAD_CONTRIBUTORS +=1
            _attitudeSwitch = self.malicious_start_round
        elif _attitude == "freerider":
            self.NUMBER_OF_FREERIDER_CONTRIBUTORS +=1
            _attitudeSwitch = self.freerider_start_round
        elif _attitude == "inactive":
            self.NUMBER_OF_INACTIVE_CONTRIBUTORS +=1
            _attitudeSwitch = None
        else:
            _attitudeSwitch = None
            raise Exception("Unknown attitude {}".format(_attitude))

        length = len(self.participants)
        self.participants.append(Participant(len(self.participants), 
                                             _train[length],
                                             _val[length],
                                             _model, 
                                             optimizer, 
                                            criterion,
                                             _attitude,
                                             self.default_collateral,
                                             self.max_collateral,
                                             _attitudeSwitch,
                                             length
                                            ))
        
        print("Participant added: {:<9} {}".format(rb(_attitude.upper()[0]+_attitude[1:]), rb("User")))

    def print_client_data_distributions(self):
        """Prints the label distribution for every client's train/val sets and the global test set."""
        if self.DATA is None:
            self.load_data(self.NUMBER_OF_CONTRIBUTORS)

        trainloaders, valloaders, testloader = self.DATA

        print("\n--- CLIENT DATA DISTRIBUTIONS ---")
        print("\nUSING DATA DISTRIBUTION:", self.data_distribution)
        if "dirichlet" in self.data_distribution:
            print(f"\nDirichlet alpha: {self.dirichlet_alpha}")

        for i, (train_loader, val_loader) in enumerate(zip(trainloaders, valloaders)):
            print(f"\nClient {i} Training Set:")
            print_label_distribution(train_loader, name=f"Client {i} Train")

            print(f"Client {i} Validation Set:")
            print_label_distribution(val_loader, name=f"Client {i} Val")

        print("\nGlobal Test Set:")
        print_label_distribution(testloader, name="Test Set")


    def get_client_data_distribution(self):
        if self.DATA is None:
            self.load_data(self.NUMBER_OF_CONTRIBUTORS)

        trainloaders, valloaders, testloader = self.DATA

        result = {
            "clients": [],
            "test": {}
        }

        for train_loader, val_loader in zip(trainloaders, valloaders):
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset

            client_info = {
                "train": {
                    "size": len(train_dataset),
                    "label_counts": get_label_distribution(train_loader),
                    "indices": train_dataset.indices if hasattr(train_dataset, "indices") else None
                },
                "val": {
                    "size": len(val_dataset),
                    "label_counts": get_label_distribution(val_loader),
                    "indices": val_dataset.indices if hasattr(val_dataset, "indices") else None
                }
            }
            result["clients"].append(client_info)

        test_dataset = testloader.dataset
        result["test"] = {
            "size": len(test_dataset),
            "label_counts": get_label_distribution(testloader),
            "indices": test_dataset.indices if hasattr(test_dataset, "indices") else None
        }

        return result

    def load_data(self, NUM_CLIENTS, _print=False):
        if self.DATA:
            return self.DATA
        
        if self.DATASET == "cifar-10":
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trainset = CIFAR10("./data", train=True, download=True, transform=transform)
            testset = CIFAR10("./data", train=False, download=True, transform=transform_test)
        else:
            trainset = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
            testset = MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
            
        
        if _print:
            print("Data Loaded:")
            print("Nr. of images for training: {:,.0f}".format(len(trainset)))
            print("Nr. of images for testing:  {:,.0f}\n".format(len(testset)))

        # Split training set into partitions to simulate the individual dataset
        partition_size = len(trainset) // NUM_CLIENTS
        lengths = [partition_size] * NUM_CLIENTS
        
        images_needed = partition_size * NUM_CLIENTS
        if images_needed < len(trainset):
            trainset,_ = random_split(trainset, [images_needed, len(trainset)-images_needed])

        gen = torch.Generator().manual_seed(42) if "42" in str(self.data_distribution) else None

        dist = self.data_distribution or "random_split"

        if dist.startswith("random_split"):
            datasets = random_split(trainset, lengths, generator=gen)

        elif dist.startswith("stratified_split"):
            datasets = stratified_split(trainset, NUM_CLIENTS, generator=gen)

        elif dist.startswith("dirichlet_split"):
            datasets = dirichlet_split(trainset, NUM_CLIENTS, alpha=self.dirichlet_alpha, generator=gen)

        else:
            raise ValueError(f"Data distribution {self.data_distribution} not recognized")

        # Split each partition into train/val and create DataLoader
        trainloaders = []
        valloaders = []

        for ds in datasets:
            len_val = len(ds) // 10
            len_train = len(ds) - len_val
            tv_lengths = [len_train, len_val]

            if "stratified" in str(self.data_distribution):
                ds_train, ds_val = stratified_split(ds, tv_lengths, generator=gen)
            else:
                ds_train, ds_val = random_split(ds, tv_lengths, generator=gen)

            trainloaders.append(DataLoader(
                ds_train,
                batch_size=self.BATCHSIZE,
                shuffle=True,
                pin_memory=PIN_MEMORY,
                num_workers=NUM_WORKERS,
                persistent_workers=PERSISTENT_WORKERS,
            ))
            valloaders.append(DataLoader(
                ds_val,
                batch_size=self.BATCHSIZE,
                shuffle=False,
                pin_memory=PIN_MEMORY,
                num_workers=NUM_WORKERS,
                persistent_workers=PERSISTENT_WORKERS,
            ))
        testloader = DataLoader(
            testset,
            batch_size=self.BATCHSIZE,
            shuffle=False,
            pin_memory=PIN_MEMORY,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
        )
        self.DATA = (trainloaders, valloaders, testloader)
        return trainloaders, valloaders, testloader


    def federated_training(self):
        print(b("\n================ PARALLEL FEDERATED TRAINING START ================"))

        start_total = time.perf_counter()

        if debugging:
            print(yellow("Debugging mode detected → running sequential training"))
            results = self.run_sequential()
        else:
            results = self.run_multi_processing()

        self.apply_training_results(results)
        total_time = time.perf_counter() - start_total

        print(b("=================== PARALLEL TRAINING END ===================\n"))
        print(green(f"Total federated training time: {total_time:.2f} seconds\n"))

    def finalize_user_evaluation(self, user): # Same as lines 294-296,306 in orgiginal code.
        loss, acc = test(user.model, self.test, DEVICE) # TODO: Investigate if this should be user.val instead.
        user._accuracy.append(acc) # Line 295 in original code # TODO: Investigate if this should be test and not validation accuracy.
        user._loss.append(loss) # Line 296 in original code # TODO: Investigate if this should be test and not validation loss.
        user.hashedModel = self.get_hash(user.model.state_dict())


    def apply_training_results(self, results):
        # Apply results back to participants
        user_map = {u.id: u for u in self.participants}
        for user_id, state_dict, val_loss, val_acc in results:
            user = user_map[user_id]
            user.model.load_state_dict(state_dict)
            user.currentAcc = val_acc # Line 287 in original code
            user.currentLoss = val_loss
            self.finalize_user_evaluation(user)


    def run_sequential(self):
        num_gpus = torch.cuda.device_count()
        print_training_mode(num_gpus, 1)

        results = []

        for idx, user in enumerate(self.participants):
            device_id = idx % max(1, num_gpus)
            sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}

            if user.attitude == "good":
                result = train_user_proc(
                    user.id,
                    sd_cpu,
                    user.train.dataset,
                    user.val.dataset,
                    self.EPOCHS,
                    device_id,
                    self.DATASET,
                    self.BATCHSIZE,
                    PIN_MEMORY,
                    False
                )
                results.append(result)
            else:
                self.finalize_user_evaluation(user)
        return results

    def run_multi_processing(self):
        num_gpus = torch.cuda.device_count()
        ctx = mp.get_context("spawn")

        num_processes = min(
            len(self.participants),
            num_gpus if num_gpus > 0 else os.cpu_count()
        )

        print_training_mode(num_gpus, num_processes)

        with ctx.Pool(processes=num_processes) as pool:
            start_pool = time.perf_counter()

            async_results = []
            for idx, user in enumerate(self.participants):
                device_id = idx % max(1, num_gpus)
                sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}  # safe copy

                if user.attitude=="good": # train
                    async_results.append(pool.apply_async(
                        train_user_proc,
                        (user.id,
                        sd_cpu,
                        user.train.dataset,
                        user.val.dataset,
                        self.EPOCHS,
                        device_id,
                        self.DATASET,
                        self.BATCHSIZE,
                        PIN_MEMORY,
                        False)
                    ))
                else: # If user's behaviour !good, skip Training.
                    # Skips apply_training_results() - goes directly to evaluation. Corresponds to lines 261-277 in original code.
                    self.finalize_user_evaluation(user)

            results = [r.get() for r in async_results] # Gather results from Multi-Processing
        print(green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def let_malicious_users_do_their_work(self):
        for i in range(len(self.participants)):
            if self.participants[i].attitude == "bad":                
                print(red("Address {} going to provide random weights".format(self.participants[i].address[0:16]+"...")))
                manipulated_state_dict = manipulate(self.participants[i].model,scale=self.malicious_noise_scale,)
                self.participants[i].model.load_state_dict(manipulated_state_dict)
                self.participants[i].hashedModel = self.get_hash(self.participants[i].model.state_dict())
                loss, accuracy = test(self.participants[i].model, self.test, DEVICE)

                self.participants[i].currentAcc = accuracy
                self.participants[i].currentLoss = loss

                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                self.participants[i].address[0:16]+"...",
                                                                                accuracy*100, loss))


    def let_freerider_users_do_their_work(self):
        for i in range(len(self.participants)):
            if self.participants[i].attitude == "freerider":
                print(red("Address {} going to provide random weights".format(self.participants[i].address[0:16]+"...")))
                manipulated_state_dict = manipulate(self.participants[i].model,scale=self.freerider_noise_scale,)
                self.participants[i].model.load_state_dict(manipulated_state_dict)
                self.participants[i].hashedModel = self.get_hash(self.participants[i].model.state_dict())
                loss, accuracy = test(self.participants[i].model, self.test, DEVICE)

                self.participants[i].currentAcc = accuracy
                self.participants[i].currentLoss = loss

                print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                self.participants[i].address[0:16]+"...",
                                                                                accuracy*100, loss))

    def update_users_attitude(self):
        for user in self.participants:
            if user.attitudeSwitch == self.round and user.attitude != user.futureAttitude:
                print(rb("Address {} going to switch attitude to {}".format(user.address[0:16]+"...",
                                                                            user.futureAttitude)))
                user.attitude = user.futureAttitude
                user.color = get_color(None, user.attitude)


    def _freerider_submit_with_noise(self, user):
        """Freerider reuses the global model with configurable noise."""

        if self.freerider_noise_scale < 0:
            raise ValueError("freerider_noise_scale must be non-negative")

        if self.freerider_noise_scale == 0: # Copy global model if noise is zero
            print(yellow("Address {} resubmitting original model".format(user.address[0:16]+"...")))
            return copy.deepcopy(user.model).state_dict()

        print(red(
            "Address {} adding noise (scale={}) to global weights".format(
                user.address[0:16]+"...",
                self.freerider_noise_scale,
            )
        ))
        return manipulate(copy.deepcopy(user.model), scale=self.freerider_noise_scale)

    def the_merge(self, _users, aggregation_rule: str, merge_weight_collector=None, agg_switch_collector=None, avg_prior_prior_accs=None, avg_prior_prior_losses=None, avg_prior_accs=None, avg_prior_losses=None):
        # No qualified users → skip merge this round
        if not _users:
            print("-----------------------------------------------------------------------------------")
            print(red("No participants qualified for merge this round – skipping aggregation"))
            print("-----------------------------------------------------------------------------------\n")
            return

        pre_merge_snapshot = copy.deepcopy(self.global_model)

        self.two_previous_global_model = self.previous_global_model
        self.previous_global_model = pre_merge_snapshot

        client_models, users_contribution_scores, users_merge_weights = [], {}, {}

        for u in _users:
            client_models.append(u.model)
            print("Account {} participating in merge".format(u.address[0:16] + "..."))
            users_contribution_scores[u.address] = u.contribution_score

        print("Using aggregation rule: {}".format(aggregation_rule))

        # -------------------------
        # Compute weights (UNIFIED)
        # -------------------------
        n_clients = len(client_models)

        if aggregation_rule == "FedAVG":
            users_merge_weights = {u.address: 1.0 / n_clients for u in _users}
            self.agg_func_1 = "FedAVG"
            self.agg_weight_1 = 1.0
            self.agg_func_2 = None
            self.agg_weight_2 = None

        elif aggregation_rule == "positives_only":
            users_merge_weights = positives_only(users_contribution_scores)
            self.agg_func_1 = "positives_only"
            self.agg_weight_1 = 1.0
            self.agg_func_2 = None
            self.agg_weight_2 = None

        elif aggregation_rule == "plus_one_normalize":
            users_merge_weights = plus_one_normalize(users_contribution_scores)
            self.agg_func_1 = "plus_one_normalize"
            self.agg_weight_1 = 1.0
            self.agg_func_2 = None
            self.agg_weight_2 = None

        elif aggregation_rule == "plus_more_than_one_normalize":
            users_merge_weights = plus_more_than_one_normalize(users_contribution_scores)
            self.agg_func_1 = "plus_more_than_one_normalize"
            self.agg_weight_1 = 1.0
            self.agg_func_2 = None
            self.agg_weight_2 = None

        elif aggregation_rule == "binary_switch":
            users_merge_weights = self.binary_switch(users_contribution_scores, positives_only, plus_one_normalize)

        elif aggregation_rule == "partial_switch_accuracy":
            users_merge_weights = self.partial_switch_accuracy(users_contribution_scores, positives_only, plus_one_normalize)

        elif aggregation_rule == "partial_switch_fixed_loss":
            func1 = positives_only
            func2 = plus_one_normalize
            if avg_prior_losses is not None:
                users_merge_weights = self.partial_switch_fixed_loss(users_contribution_scores, avg_prior_losses, func1, func2, agg_switch_collector)
            else:
                print(yellow("Warning: Missing prior losses for partial_switch_fixed_loss. Defaulting to plus_one_normalize."))
                users_merge_weights = plus_one_normalize(users_contribution_scores)

        elif aggregation_rule == "partial_switch_retrospective":
            func1 = positives_only
            func2 = plus_one_normalize

            if avg_prior_prior_losses is not None and avg_prior_losses is not None:
                users_merge_weights = self.partial_switch(users_contribution_scores, avg_prior_prior_losses, avg_prior_losses, func1, func2, agg_switch_collector)
            else:
                print(yellow("Warning: Missing prior losses for partial_switch_retrospective. Defaulting to plus_one_normalize."))
                users_merge_weights = plus_one_normalize(users_contribution_scores)

        elif aggregation_rule == "GRS_aggregation":
            users_merge_weights = self.GRS_aggregation(_users)

        else:
            raise ValueError(f"Unknown merge strategy: {aggregation_rule}")

        for u in _users:
            u.merge_weight = users_merge_weights[u.address]

        if merge_weight_collector is not None:
            merge_weight_collector.update(users_merge_weights)

        assert abs(sum(users_merge_weights.values()) - 1.0) < 1e-6, "Aggregation weights must sum to 1"


        # -------------------------
        # Cache client state_dicts (IMPORTANT OPTIMIZATION)
        # -------------------------
        client_state_dicts = [m.state_dict() for m in client_models]
        ordered_weights = [users_merge_weights[u.address] for u in _users]

        with torch.no_grad():
            global_dict = self.global_model.state_dict()

            for k in global_dict.keys():
                # Stack all client parameters
                stacked = torch.stack([
                    client_state_dicts[i][k].to(
                        device=global_dict[k].device,
                        dtype=global_dict[k].dtype
                    )
                    for i in range(n_clients)
                ], dim=0)

                # Prepare weights tensor (once per param for correct device/dtype)
                w = torch.tensor(ordered_weights, device=stacked.device, dtype=stacked.dtype)
                w = w.view(-1, *([1] * (stacked.dim() - 1)))

                # Weighted aggregation (covers ALL rules including FedAvg)
                global_dict[k] = (stacked * w).sum(0)

            self.global_model.load_state_dict(global_dict)

        # -------------------------
        # Evaluation
        # -------------------------
        loss, accuracy = test(self.global_model, self.test, DEVICE)
        self.accuracy.append(accuracy)
        self.loss.append(loss)

        for u in _users:
            print(f"User {u.address[0:16]}... merge_weight: {users_merge_weights[u.address]:.4f}")

        print("-----------------------------------------------------------------------------------")
        print(b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy * 100, loss)))

        # -------------------------
        # Distribute global model
        # -------------------------
        for u in self.participants:
            u.previousModel = copy.deepcopy(u.model)
            u.model.load_state_dict(self.global_model.state_dict())

        print("-----------------------------------------------------------------------------------\n")

    def exchange_models(self):
        print("Users exchanging models...")
        for user in self.participants:
            user.userToEvaluate = []
            for j in self.participants:
                if user.model == j.model:
                    continue
                if j.model in user.userToEvaluate:
                    continue
                user.userToEvaluate.append(j)
        print("-----------------------------------------------------------------------------------")

    def verify_models(self, on_chain_hashes):
        print("Users verifying models...")
        for _user in self.participants:
            _user.cheater = []
            for user in _user.userToEvaluate:  
                if not self.get_hash(user.model.state_dict()) == on_chain_hashes[user.id]:
                    print(red(f"Account {_user.id}: Account {user.address[0:16]}... could not provide the registered model"))
                    _user.cheater.append(user)
                    
        print("-----------------------------------------------------------------------------------")


    def get_hash(self, _state_dict):
        if not isinstance(_state_dict, dict):
            _state_dict = dict(_state_dict)

        parts = []
        for k, v in sorted(_state_dict.items(), key=lambda x: x[0]):
            t = v.detach()
            if t.is_cuda:
                t = t.cpu()
            t = t.contiguous()
            parts.append(k.encode("utf-8"))
            parts.append(b"|")
            # include shape to avoid accidental collisions
            parts.append(np.asarray(t.shape, dtype=np.int64).tobytes())
            parts.append(b"|")
            parts.append(t.numpy().tobytes())
            parts.append(b"\n")
        blob = b"".join(parts)
        return Web3.keccak(blob)  #remove hex to match old, with improved algo.

    def evaluation(self):
        print("Users evaluating models...")

        scalar = 100 # Adds more decimals for precision (Adding 0 gives another decimal, vice versa)
        MAX_UINT16_SIZE = 65535
        count_dq = len(self.disqualified)

        feedback_matrix = np.zeros((1, len(self.participants) + count_dq, len(self.participants) + count_dq))[0]
        n = len(self.participants) + count_dq
        accuracy_matrix = [[0 for _ in range(n)] for _ in range(n)]
        loss_matrix = [[0 for _ in range(n)] for _ in range(n)]
        prev_accs = [0 for _ in range(n)]
        prev_losses = [0 for _ in range(n)]

        for feedbackGiver in self.participants:
            valloader = feedbackGiver.val
            bad_att = feedbackGiver.attitude == "bad"
            free_att = feedbackGiver.attitude == "freerider"
            accuracy_last_round = -1

            for ix, user in enumerate(feedbackGiver.userToEvaluate):
                if not bad_att and not free_att:
                    loss, accuracy = test(user.model, valloader, DEVICE)
                    prev_loss, prev_acc = test(self.global_model, valloader, DEVICE)
                    prev_acc = round(prev_acc * 100 * scalar)
                    prev_loss = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

                if bad_att:
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = 0
                    loss_matrix[feedbackGiver.id][user.id] = 65535
                    prev_loss, prev_acc = test(self.global_model, valloader, DEVICE)
                    prev_accs[feedbackGiver.id] = round(prev_acc * 100 * scalar)
                    prev_losses[feedbackGiver.id] = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

                elif free_att:
                    feedback_matrix[feedbackGiver.id][user.id] = 0
                    if accuracy_last_round == -1:
                        loss_last_round, accuracy_last_round = test(self.global_model, valloader, DEVICE)
                        accuracy_last_round = round(accuracy_last_round * 100 * scalar)
                        loss_last_round = safe_scale(loss_last_round, scalar, MAX_UINT16_SIZE)
                    accuracy_matrix[feedbackGiver.id][user.id] = accuracy_last_round
                    loss_matrix[feedbackGiver.id][user.id] = min(loss_last_round, MAX_UINT16_SIZE)
                    prev_accs[feedbackGiver.id] = accuracy_last_round
                    prev_losses[feedbackGiver.id] = loss_last_round

                elif user in feedbackGiver.cheater:
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                    loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    prev_accs[feedbackGiver.id] = prev_acc
                    prev_losses[feedbackGiver.id] = prev_loss

                elif accuracy > feedbackGiver.currentAcc - 0.07 : # 7% Worse
                    feedback_matrix[feedbackGiver.id][user.id] = 1
                    accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                    loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    prev_accs[feedbackGiver.id] = prev_acc
                    prev_losses[feedbackGiver.id] = prev_loss

                elif accuracy > feedbackGiver.currentAcc - 0.14: # 14% Worse
                    feedback_matrix[feedbackGiver.id][user.id] = 0
                    accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                    loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    prev_accs[feedbackGiver.id] = prev_acc
                    prev_losses[feedbackGiver.id] = prev_loss

                else:
                    feedback_matrix[feedbackGiver.id][user.id] = -1
                    accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                    loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                    prev_accs[feedbackGiver.id] = prev_acc
                    prev_losses[feedbackGiver.id] = prev_loss

                if self.force_merge_all:
                    feedback_matrix[feedbackGiver.id][user.id] = 0

            # Reset
            feedbackGiver.userToEvaluate = []
        # acc_mat = [[x / 10 for x in sublist] for sublist in accuracy_matrix]
        # loss_mat = [[x / 10 for x in sublist] for sublist in loss_matrix]
        # prev_accs_divided = [x / 10 for x in prev_accs]
        # prev_losses_divided = [x / 10 for x in prev_losses]

        print("FEEDBACK MATRIX:")
        print(feedback_matrix)
        print("-----------------------------------------------------------------------------------")
        print("ACCURACY MATRIX:")
        print(accuracy_matrix)
        print("-----------------------------------------------------------------------------------")
        print("LOSS MATRIX:")
        print(loss_matrix)
        print("-----------------------------------------------------------------------------------")
        print("PREVIOUS ACCURACIES:")
        print(prev_accs)
        print("-----------------------------------------------------------------------------------")
        print("PREVIOUS LOSSES:")
        print(prev_losses)
        print("-----------------------------------------------------------------------------------")

        return feedback_matrix, accuracy_matrix, loss_matrix, prev_accs, prev_losses

    @staticmethod
    def models_are_equal(model_a, model_b):
        previous_model_params = model_a.state_dict()
        pre_previous_model_params = model_b.state_dict()

        if previous_model_params.keys() != pre_previous_model_params.keys():
            return False

        return all(
            torch.equal(previous_model_params[x], pre_previous_model_params[x])
            for x in previous_model_params
        )

    def binary_switch(self, users_contrib_scores, func_1, func_2):
        if self.has_switched:
            print(f"  [binary_switch] At round {self.round}: Using {func_2.__name__}")
            self.agg_func_1 = func_1.__name__
            self.agg_weight_1 = 0.0
            self.agg_func_2 = func_2.__name__
            self.agg_weight_2 = 1.0
            return func_2(users_contrib_scores)

        if self.round <= 1 or self.two_previous_global_model is None:
            print(f"  [binary_switch] At round {self.round}: Using {func_1.__name__}")
            self.agg_func_1 = func_1.__name__
            self.agg_weight_1 = 1.0
            self.agg_func_2 = func_2.__name__
            self.agg_weight_2 = 0.0
            return func_1(users_contrib_scores)

        converged = self.models_are_equal(self.previous_global_model, self.two_previous_global_model)

        if converged:
            self.has_switched = True
            print(f"  [binary_switch] Convergens detected at round {self.round}: Switching from {func_1.__name__} to {func_2.__name__}")
            self.agg_func_1 = func_1.__name__
            self.agg_weight_1 = 0.0
            self.agg_func_2 = func_2.__name__
            self.agg_weight_2 = 1.0
            return func_2(users_contrib_scores)


        print(f"  [binary_switch] At round {self.round}: Using {func_1.__name__}")
        self.agg_func_1 = func_1.__name__
        self.agg_weight_1 = 1.0
        self.agg_func_2 = func_2.__name__
        self.agg_weight_2 = 0.0
        return func_1(users_contrib_scores)

    def partial_switch_accuracy(self, users_contrib_scores, func_1, func_2):

        current_accuracy = self.accuracy[-1] # latest accuracy TODO Fix med rigtig accuracy

        accuracy_measure = current_accuracy  # How much % to use func_2 (0,47 -> 47% func_2)

        weights_1 = func_1(users_contrib_scores) # Generate weights using the primary/early-stage aggregator
        weights_2 = func_2(users_contrib_scores) # Generate weights using the secondary/late-stage aggregator

        mixed_weights = {
            key: (1 - accuracy_measure) * weights_1[key] + accuracy_measure * weights_2[key]
            for key in users_contrib_scores
        }

        # Normalizing
        total = sum(mixed_weights.values())
        mixed_weights = {key: value / total for key, value in mixed_weights.items()}

        self.agg_func_1 = func_1.__name__
        self.agg_weight_1 = 1 - accuracy_measure
        self.agg_func_2 = func_2.__name__
        self.agg_weight_2 = accuracy_measure

        return mixed_weights


    def partial_switch_fixed_loss(self, users_contrib_scores, avg_prior_losses, func_1, func_2, threshold=100, agg_switch_collector=None):
        if avg_prior_losses is None:
            alpha = 0.0
        elif avg_prior_losses >= threshold:
            alpha = 0.0
        else:
            ratio = 1.0 - (avg_prior_losses / threshold)
            alpha = math.sin(math.radians(ratio * 90.0))

        beta = 1.0 - alpha

        print(f"partial_switch_fixed_loss: prior_loss={avg_prior_losses}, threshold={threshold}, improvement_ratio={ratio}, "
              f"alpha(func2)={alpha:.3f}, beta(func1)={beta:.3f}")

        if agg_switch_collector is not None:
            agg_switch_collector.update({
                "func_1": func_1.__name__,
                "weight_1": beta,
                "func_2": func_2.__name__,
                "weight_2": alpha,
            })

        res1 = func_1(users_contrib_scores)
        res2 = func_2(users_contrib_scores)

        combined = {
            addr: beta * res1[addr] + alpha * res2[addr]
            for addr in users_contrib_scores
        }

        total = sum(combined.values())
        if total <= 0:
            n = len(combined)
            return {addr: 1.0 / n for addr in combined}

        return {addr: w / total for addr, w in combined.items()}


    def GRS_aggregation(self, users):
        total_grs = sum(user._globalrep[-1] for user in users)

        aggregation_scores = {user.address: user._globalrep[-1] / total_grs for user in users}
        return aggregation_scores



    def partial_switch(self, users_contrib_scores, avg_prior_prior_losses, avg_prior_losses, func_1, func_2, agg_switch_collector=None):
        # Look 1 round back - if no improvement, switch to func_2 with increasing probability as rounds go by
        # Check if not improving (Convergence)
        # calculate sinus/cosine.
        # Look at loss. First loss is max (100%)
        # Then consider a sinusoidal function that converts the diff loss'-loss into an angle between 0 and 90 degrees, where 0 degrees (func_1) corresponds to no improvement and 90 degrees (func_2) corresponds to max improvement.
        # 100% positive top/bottom - 100% plus_one left/right

        # I need access to a user's - loss. We need to only access what we can - not cheat by taking the global one.
        # Return to aggregation score. That's partially made up of percentage-wise of the two provided functions
        # Question is: how do we combine the output of two? Luckily, its a dict, so we can combine a users (address) agg. scores across two functions.
        # Next question is: how do we calculate the percentage
        # Find out if we are in convergence.


        # ---
        #   - Compare avg_prev_losses (last round) against avg_prior_losses (round before that)
        #   - Map the improvement into an angle 0–90°, where 0° = no improvement → use func_1 fully, 90° = max improvement → use func_2
        #   fully
        #   - Use sine of that angle as the blend weight so the transition is smooth (not linear)
        #
        #   The implementation:
        #
        #   1. Improvement ratio = (prior - prev) / prior, clamped to [0, 1]. This normalises the loss drop relative to the starting
        #   point so the scale doesn't matter.
        #   2. Sinusoidal blend: alpha = sin(ratio × 90°). At ratio=0 → sin(0°)=0 (fully func_1). At ratio=1 → sin(90°)=1 (fully func_2).
        #    The sine curve gives a slow start and fast finish — conservative early, more aggressive as improvement grows.
        #   3. Combine: w[addr] = beta * res1[addr] + alpha * res2[addr], then renormalise to sum to 1 since both func outputs are
        #   already normalised separately but their linear combination may not be.
        #
        #   Why func_1 = no improvement, func_2 = improvement?
        #   positives_only (func_1) is strict — it zeroes out negative contributors, which is useful when loss isn't moving and you need
        #   to filter noise. plus_one_normalize (func_2) is softer — it rewards improvement more broadly, which makes sense when the
        #   model is actually converging.
        # ---


        # ---
        # - alpha — weight for func_2 (plus_one_normalize). High when loss improved a lot. Soft/rewarding.
        #   - beta — weight for func_1 (positives_only). High when loss barely improved. Strict/conservative.
        #
        #   They always sum to 1. The more the model improves round-over-round, the more alpha dominates and the softer the
        #   aggregation becomes. When improvement stagnates, beta dominates and bad contributors get filtered out more
        #   aggressively.
        # ---

        res1 = func_1(users_contrib_scores)  # e.g. positives_only    — strict, filters noise
        res2 = func_2(users_contrib_scores)  # e.g. plus_one_normalize — soft, rewards broadly

        # How much did loss improve from the prior round to the previous round?
        # Normalise by prior loss so the ratio is scale-independent, clamp to [0, 1].
        if avg_prior_losses <= 0:
            improvement_ratio = 0.0
        else:
            improvement_ratio = max(0.0, min(1.0, (avg_prior_prior_losses - avg_prior_losses) / avg_prior_losses)) # 40.000-30.000/30.000 = 0.33

        # Map ratio → angle [0°, 90°] → sine blend weight.
        # sin(0°) = 0  → no improvement  → fully func_1 (strict/conservative)
        # sin(90°) = 1 → max improvement → fully func_2 (soft/rewarding)
        # Sine gives a slow start and fast finish: cautious early, more aggressive as convergence grows.
        alpha = math.sin(math.radians(improvement_ratio * 90.0))  # weight for func_2   0.33 → 30° → sin(30°)=0.5
        beta  = 1.0 - alpha                                        # weight for func_1  0.67 → 60° → sin(60°)=0.87

        print(f"partial_switch: prior_loss={avg_prior_prior_losses:.4f}, prev_loss={avg_prior_losses:.4f}, "
              f"improvement={improvement_ratio:.3f}, alpha(func2)={alpha:.3f}, beta(func1)={beta:.3f}")

        if agg_switch_collector is not None:
            agg_switch_collector.update({
                "func_1": func_1.__name__,
                "weight_1": beta,
                "func_2": func_2.__name__,
                "weight_2": alpha,
            })

        # Blend the two score dicts linearly by (beta, alpha)
        combined = {
            addr: beta * res1[addr] + alpha * res2[addr]
            for addr in users_contrib_scores
        }

        # Re-normalise: each func's output sums to 1 individually, but their
        # linear combination may not — especially if func_1 zeroed some users out.
        total = sum(combined.values())
        if total <= 0:
            n = len(combined)
            return {addr: 1.0 / n for addr in combined}

        return {addr: w / total for addr, w in combined.items()}


# PYTORCH FUNCTIONS
def train(
    net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,
) -> None:

    # Compile ONCE per process (not per batch)
    if device.type == "cuda":
        try:
            net = torch.compile(net, mode="reduce-overhead")
        except Exception:
            pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


def test(net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model on test set: forward pass only (no gradients), with optional AMP on CUDA
    Accumulate total cross-entropy loss and count correct predictions for accuracy
    Returns (total_loss, accuracy) over the entire test dataset
    """
    criterion = nn.CrossEntropyLoss()
    net.eval()

    correct = 0
    total = 0
    loss = 0.0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=NON_BLOCKING)
            labels = labels.to(device, non_blocking=NON_BLOCKING)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    loss = min(sys.float_info.max, loss)

    return loss, accuracy
    
def green(text):
    return colored(text, "green")

def gb(string):
    return colored(string, color="green", attrs=["bold"])

def rb(string):
    return colored(string, color="red", attrs=["bold"])

def b(string):
    return colored(string, color=None, attrs=["bold"])

def red(text):
    return colored(text, "red")

def yellow(text):
    return colored(text, "yellow", attrs=["bold"])


def manipulate(model, scale: float = 1.0) -> OrderedDict:
    sd = OrderedDict()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            t = v.clone()
            if t.is_floating_point():
                # uniform noise in [-scale, scale]
                noise = torch.empty_like(t).uniform_(-scale, scale)
                t.add_(noise)
            sd[k] = t
    return sd


def add_noise(model, offset_from_end: int = 5) -> OrderedDict:
    """
    GPU-friendly: keep tensors on their original device/dtype and add a tiny scalar
    to the tensor at index len(state_dict)-offset_from_end.
    """
    items = list(model.state_dict().items())
    target_idx = max(0, len(items) - offset_from_end)

    new_sd = OrderedDict()
    with torch.no_grad():
        for idx, (k, v) in enumerate(items):
            t = v.clone()
            if t.is_floating_point() and idx == target_idx:
                # Match original magnitude: 9e-6 or 1e-5
                eps = 1e-5 if random.randint(9, 10) == 10 else 9e-6
                t.add_(eps)  # in-place scalar add on the same device (CPU/GPU)
            new_sd[k] = t
    return new_sd


def get_color(i, a):
    if a == "bad":
        return bad_c
    if a == "freerider":
        return free_c
    try:
        return colors[i]
    except:
        return None


def train_user_proc(user_id, model_state, train_ds, val_ds, epochs, device_id, dataset, batchsize, pin_memory, shuffle):
        # Multi-GPU Support
        # Select device
        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")

        # Recreate model based on dataset
        if dataset == "mnist":
            model = Net_MNIST()
        else:
            model = Net_CIFAR()

        model.load_state_dict(model_state)
        model.to(device)

        # Rebuild dataloaders inside the process
        train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=shuffle, pin_memory=pin_memory) # TODO: Investigate if this breaks something
        val_loader = DataLoader(val_ds, batch_size=batchsize, shuffle=False, pin_memory=pin_memory) # TODO: Investigate if this breaks something

        train(model, train_loader, epochs, device) # Line 285 in original code
        val_loss, val_acc = test(model, val_loader, device) # Line 286 in original code

        # del: Mark for GC
        del train_loader
        del val_loader

        print(f"[{device_label(device, device_id)}] User {user_id} done | Acc: {val_acc:.3f}, Loss: {val_loss:.3f}")
        
        # Ensure all GPU work is complete before worker exits
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return user_id, model.state_dict(), val_loss, val_acc


def print_training_mode(num_gpus: int, num_processes: int):
    """Prints a clean status message describing how training will run."""
    if num_gpus >= 2:
        print(green(f"Detected {num_gpus} GPU(s) → Parallel multi-GPU training"))

    elif num_gpus == 1:
        if num_processes > 1:
            print(yellow(
                f"Detected 1 GPU → Parallel training on one GPU (shared across {num_processes} workers)"
            ))
        else:
            print(green("Detected 1 GPU → Sequential GPU training"))

    else:  # CPU-only
        if num_processes > 1:
            print(yellow(
                f"Detected 0 GPU(s) → Parallel CPU training ({num_processes} workers)"
            ))
        else:
            print(red("Detected 0 GPU(s) → Sequential CPU mode"))


def device_label(device: torch.device, device_id: int = 0) -> str:
    if device.type == "cuda":
        return f"GPU {device_id}"
    else:
        return "CPU"

def safe_scale(value, scalar, max_val):
    if not math.isfinite(value):
        return max_val

    scaled = value * scalar

    if not math.isfinite(scaled):
        return max_val

    return min(round(scaled), max_val)


def dirichlet_split(dataset, num_clients, alpha=0.5, generator=None):
    """
    Dirichlet split that ensures:
    - All clients have exactly len(dataset)//num_clients samples
    - Non-IID distribution controlled by alpha
    - Optional reproducibility with generator
    """
    # Handle Subset
    if isinstance(dataset, Subset):
        actual_dataset = dataset.dataset
        indices = dataset.indices
    else:
        actual_dataset = dataset
        indices = list(range(len(dataset)))

    # Get labels
    if hasattr(actual_dataset, "targets"):
        targets = torch.tensor(actual_dataset.targets)[indices]
    elif hasattr(actual_dataset, "labels"):
        targets = torch.tensor(actual_dataset.labels)[indices]
    else:
        raise AttributeError("Dataset must have targets or labels attribute")

    num_classes = len(torch.unique(targets))
    class_indices = {c: (targets == c).nonzero(as_tuple=True)[0].tolist() for c in range(num_classes)}

    # Shuffle class indices
    seed = 42 if generator is not None else None
    rng = np.random.default_rng(seed)
    for c in class_indices:
        rng.shuffle(class_indices[c])

    splits = [[] for _ in range(num_clients)]
    total_per_client = len(dataset) // num_clients
    client_counts = [0] * num_clients  # track total assigned per client

    # Allocate samples per class
    for c in range(num_classes):
        cls_idx = class_indices[c]
        n_cls = len(cls_idx)

        # Sample Dirichlet proportions
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = (proportions * n_cls).astype(int)

        # Fix rounding to match class total
        diff = n_cls - counts.sum()
        if diff > 0:
            topk = np.argsort(proportions)[-diff:]
            counts[topk] += 1

        # Assign to clients, but do not exceed total_per_client
        start = 0
        for i in range(num_clients):
            # adjust count if client is almost full
            remaining_space = total_per_client - client_counts[i]
            assign_count = min(counts[i], remaining_space)
            end = start + assign_count
            splits[i].extend(cls_idx[start:end])
            client_counts[i] += assign_count
            start += assign_count

    # At this point, all clients should have exactly total_per_client
    # If any client still has less (due to rounding), fill from leftover pool
    assigned = set(idx for client in splits for idx in client)
    all_indices = set(idx for cls in class_indices.values() for idx in cls)
    remaining = list(all_indices - assigned)
    rng.shuffle(remaining)

    for i in range(num_clients):
        current_len = len(splits[i])
        if current_len < total_per_client:
            needed = total_per_client - current_len
            splits[i].extend(remaining[:needed])
            remaining = remaining[needed:]

    return [Subset(actual_dataset, split) for split in splits]




def stratified_split(dataset, lengths, generator=None):
    from torch.utils.data import Subset
    import torch

    # Normalize input
    if isinstance(lengths, int):
        num_splits = lengths
        lengths = [1] * num_splits  # equal split
    else:
        num_splits = len(lengths)

    total_length = sum(lengths)

    # Handle subset
    if isinstance(dataset, Subset):
        actual_dataset = dataset.dataset
        base_indices = list(dataset.indices)
    else:
        actual_dataset = dataset
        base_indices = list(range(len(dataset)))

    # Get targets
    if hasattr(actual_dataset, "targets"):
        targets = actual_dataset.targets
    elif hasattr(actual_dataset, "labels"):
        targets = actual_dataset.labels
    else:
        raise AttributeError("Dataset must have targets or labels")

    targets = torch.as_tensor(targets, dtype=torch.long)[base_indices]

    # Group by class
    classes = torch.unique(targets)
    class_indices = {
        int(c): (targets == c).nonzero(as_tuple=True)[0].tolist()
        for c in classes
    }

    # Shuffle per class
    for c in class_indices:
        perm = torch.randperm(len(class_indices[c]), generator=generator).tolist()
        class_indices[c] = [class_indices[c][i] for i in perm]

    splits = [[] for _ in range(num_splits)]

    # Split per class proportionally
    for c in class_indices:
        cls_idx = class_indices[c]
        cls_len = len(cls_idx)

        proportions = [l / total_length for l in lengths]
        raw = [p * cls_len for p in proportions]
        cls_lengths = [int(x) for x in raw]

        # distribute remainder
        remainder = cls_len - sum(cls_lengths)
        for i in range(remainder):
            cls_lengths[i % num_splits] += 1

        start = 0
        for i in range(num_splits):
            selected_local = cls_idx[start:start + cls_lengths[i]]
            selected_global = [base_indices[idx] for idx in selected_local]
            splits[i].extend(selected_global)
            start += cls_lengths[i]

    # Final shuffle
    for i in range(num_splits):
        perm = torch.randperm(len(splits[i]), generator=generator).tolist()
        splits[i] = [splits[i][j] for j in perm]

    return [Subset(actual_dataset, split) for split in splits]


def get_label_distribution(loader):
    counter = Counter()
    for _, labels in loader:
        counter.update(labels.tolist())
    return dict(counter)



# OLD — BUG: division by zero when all scores <= 0
def positives_only(users_contrib_scores: dict):
     positive_sum = sum(score for score in users_contrib_scores.values() if score > 0)
     if positive_sum <= 0:
         raise Exception("No positive contribution scores; cannot normalize")
     aggregation_scores = {user_id: score / positive_sum if score > 0 else 0 for user_id, score in users_contrib_scores.items()}
     return aggregation_scores


# OLD — BUG: denominator n+1 is only correct when scores already sum to 1
def plus_one_normalize(users_contrib_scores: dict):
     # normalized_scores = [score + 1 for score in users_contrib_scores]
     normalized_scores = {user_id: score + 1 for user_id, score in users_contrib_scores.items()}
     sum_ = sum(normalized_scores.values())
     aggregation_scores = {user_id: score / sum_ for user_id, score in normalized_scores.items()}
     return aggregation_scores


# OLD — BUG: denominator n*more_than_one+1 is only correct when scores already sum to 1
def plus_more_than_one_normalize(users_contrib_scores: dict, more_than_one=1.1):
     normalized_scores = {user_id: score + more_than_one for user_id, score in users_contrib_scores.items()}
     sum_ = sum(normalized_scores.values())
     aggregation_scores = {user_id: score / sum_ for user_id, score in normalized_scores.items()}
     return aggregation_scores




def print_label_distribution(loader, name="Dataset"):
    """
    Prints the distribution of labels in a DataLoader.
    """
    label_counts = Counter()
    for _, labels in loader:
        label_counts.update(labels.numpy())  # Convert tensor to numpy for Counter

    total = sum(label_counts.values())
    print(f"{name} | Total samples: {total}")
    for label in range(10):
        count = label_counts[label]
        print(f"  Label {label}: {count} ({count/total*100:.2f}%)")
    print("-" * 50)
