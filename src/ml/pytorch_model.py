import sys
import atexit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
import time
import ml.training as training
import ml.data as data
import ml.evaluation as evaluation
from utils.colors import green, red, yellow, b
from ml.cnn_models import Net_CIFAR, Net_MNIST
from ml.runtime import DEVICE, PIN_MEMORY, print_training_mode
from ml.participant import Participant

debugging = sys.gettrace() is not None

class PytorchModel:
    def __init__(self, DATASET, _good_participants, _bad_participants, _freerider_participants, epochs, batchsize, default_collateral, max_collateral, freerider_noise_scale: float = 1.0, freerider_start_round: int = 3, malicious_start_round: int = 3, malicious_noise_scale: float = 1.0,force_merge_all: bool = False, use_nobody_is_kicked: bool = False, data_distribution : str = None, dirichlet_alpha: float = None, malicious_attack_type: str = "noise", freerider_attack_type: str = "noise"):
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
        self._pool = None
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

        self.train, self.val, self.test = data.load_data(self, self.NUMBER_OF_CONTRIBUTORS, _print=True)
        self.default_collateral = default_collateral
        self.max_collateral = max_collateral
        self.force_merge_all = force_merge_all
        self.use_nobody_is_kicked = use_nobody_is_kicked
        self.has_switched = False


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

        valid_malicious_attack_types = {"noise", "byzantine"}
        if malicious_attack_type not in valid_malicious_attack_types:
            raise ValueError(f"malicious_attack_type must be one of {valid_malicious_attack_types}, got '{malicious_attack_type}'")
        self.malicious_attack_type = malicious_attack_type

        valid_freerider_attack_types = {"noise", "delta_weight"}
        if freerider_attack_type not in valid_freerider_attack_types:
            raise ValueError(f"freerider_attack_type must be one of {valid_freerider_attack_types}, got '{freerider_attack_type}'")
        self.freerider_attack_type = freerider_attack_type

        loss, accuracy = training.test(self.global_model,self.test,DEVICE)

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

        self._pool_size = 0


    def add_participant(self, _attitude):
        _train, _val, _test = data.load_data(self, self.NUMBER_OF_CONTRIBUTORS)

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

        color_fn = green if _attitude == "good" else (yellow if _attitude == "inactive" else red)
        print("Participant added: {:<9} {}".format(color_fn(_attitude.upper()[0]+_attitude[1:]), color_fn("User")))


    def create_pool(self):
        if self._pool is not None:
            return
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            ctx = mp.get_context("spawn")
            self._pool_size = num_gpus
            self._pool = ctx.Pool(processes=self._pool_size)
        elif num_gpus == 0:
            ctx = mp.get_context("spawn")
            self._pool_size = min(len(self.participants), os.cpu_count() or 1)
            self._pool = ctx.Pool(processes=self._pool_size)
        else:
            self._pool_size = 1
        # Single GPU: _pool stays None, run_sequential() used instead
        if self._pool is not None:
            atexit.register(self.close_pool)

    def federated_training(self):
        self.create_pool()
        _sequential = debugging or self._pool is None
        mode = "SEQUENTIAL" if _sequential else "PARALLEL"
        print(b(f"\n================ {mode} FEDERATED TRAINING START ================"))

        start_total = time.perf_counter()

        if _sequential:
            if self._pool is None and not debugging:
                print(yellow("Single GPU detected → running sequential training"))
            elif debugging:
                print(yellow("Debugging mode detected → running sequential training"))
            results = self.run_sequential()
        else:
            results = self.run_multi_processing()

        self.apply_training_results(results)
        total_time = time.perf_counter() - start_total

        print(b(f"=================== {mode} TRAINING END ===================\n"))
        print(green(f"Total federated training time: {total_time:.2f} seconds\n"))


    def apply_training_results(self, results):
        # Apply results back to participants
        user_map = {u.id: u for u in self.participants}
        for user_id, state_dict, val_loss, val_acc in results:
            user = user_map[user_id]
            user.model.load_state_dict(state_dict)
            user.currentAcc = val_acc # Line 287 in original code
            user.currentLoss = val_loss
            evaluation.finalize_user_evaluation(self, user)


    def run_sequential(self):
        num_gpus = torch.cuda.device_count()
        print_training_mode(num_gpus, 1)

        results = []

        for idx, user in enumerate(self.participants):
            device_id = idx % max(1, num_gpus)
            sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}

            if user.attitude == "good":
                result = training.train_user_proc(
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
                evaluation.finalize_user_evaluation(self, user)
        return results

    def run_multi_processing(self):
        if len(self.participants) == 0:
            raise RuntimeError("All participants have been disqualified - simulation cannot continue.")

        num_gpus = torch.cuda.device_count()
        print_training_mode(num_gpus, self._pool_size)

        start_pool = time.perf_counter()
        async_results = []
        for idx, user in enumerate(self.participants):
            device_id = idx % max(1, num_gpus)
            sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}

            if user.attitude == "good":  # train
                async_results.append(self._pool.apply_async(
                    training.train_user_proc,
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
            else:  # If user's behaviour !good, skip Training.
                # Skips apply_training_results() - goes directly to evaluation. Corresponds to lines 261-277 in original code.
                evaluation.finalize_user_evaluation(self, user)


        try:
            results = [r.get() for r in async_results]
        except KeyboardInterrupt:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            raise

        print(green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results

    def close_pool(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
