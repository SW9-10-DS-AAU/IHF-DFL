import gc
import sys
import signal
import atexit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import os
import time
import psutil
import ml.training as training
import ml.data as data
import ml.evaluation as evaluation
from utils.colors import green, red, yellow, b
from ml.cnn_models import Net_CIFAR, Net_MNIST
from ml.runtime import DEVICE, PIN_MEMORY, print_training_mode
from ml.participant import Participant

debugging = sys.gettrace() is not None

class PytorchModel:
    def __init__(self, DATASET, _good_participants, _bad_participants, _freerider_participants, epochs, batchsize, default_collateral, max_collateral, freerider_noise_scale: float = 1.0, freerider_start_round: int = 3, malicious_start_round: int = 3, malicious_noise_scale: float = 1.0,force_merge_all: bool = False, use_nobody_is_kicked: bool = False, run_id: int = None):
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
        self.run_id = 0 if run_id is None else run_id
        self.train, self.val, self.test = data.load_data(self, _print=True)
        self.default_collateral = default_collateral
        self.max_collateral = max_collateral
        self.force_merge_all = force_merge_all
        self.use_nobody_is_kicked = use_nobody_is_kicked


        if freerider_noise_scale is not None and freerider_noise_scale < 0:
            raise ValueError("freerider_noise_scale must be non-negative")
        self.freerider_noise_scale = freerider_noise_scale

        if freerider_start_round is not None and freerider_start_round < 1:
            raise ValueError("freerider_start_round must be at least 1")
        self.freerider_start_round = freerider_start_round

        if malicious_start_round is not None and malicious_start_round < 1:
            raise ValueError("malicious_start_round must be at least 1")
        self.malicious_start_round = malicious_start_round

        if malicious_noise_scale is not None and malicious_noise_scale < 0:
            raise ValueError("malicious_noise_scale must be non-negative")
        self.malicious_noise_scale = malicious_noise_scale

        loss, accuracy = training.test(self.global_model,self.test,DEVICE)

        # INTERFACE VARIABLES
        self.accuracy = [accuracy]
        self.loss = [loss]

        self.round = 1
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
        _train, _val, _test = data.load_data(self)

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
            self._pool_size = self.resolve_cpu_pool_size()
            if self._pool_size > 1:
                self._pool = ctx.Pool(processes=self._pool_size)
        else:
            self._pool_size = 1
        # Single-worker CPU and single-GPU runs stay sequential.
        if self._pool is not None:
            atexit.register(self.close_pool)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


    def resolve_cpu_pool_size(self):
        override = os.getenv("IHP-DFL_CPU_WORKERS")
        if override is not None:
            try:
                workers = int(override)
            except ValueError as exc:
                raise ValueError("IHP-DFL_CPU_WORKERS must be an integer") from exc
            if workers < 1:
                raise ValueError("IHP-DFL_CPU_WORKERS must be at least 1")
            return max(1, min(workers, len(self.participants)))

        logical_cores = os.cpu_count() or 1
        physical_cores = psutil.cpu_count(logical=False) or logical_cores
        available_gb = psutil.virtual_memory().available / (1024 ** 3)

        core_cap = max(1, physical_cores - 1)
        ram_cap = max(1, int((available_gb - 2) // 2))
        return max(1, min(len(self.participants), core_cap, ram_cap))



    def federated_training(self):
        if not debugging:
            self.create_pool()
        _sequential = debugging or self._pool is None
        mode = "SEQUENTIAL" if _sequential else "PARALLEL"
        print(b(f"\n================ {mode} FEDERATED TRAINING START ================"))

        start_total = time.perf_counter()

        if _sequential:
            if self._pool is None and not debugging:
                num_gpus = torch.cuda.device_count()
                if num_gpus == 0:
                    print(yellow("CPU detected → running sequential training"))
                else:
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

    @staticmethod
    def _shutdown_loader(loader):
        if loader is None:
            return
        it = getattr(loader, "_iterator", None)
        if it is not None:
            shutdown = getattr(it, "_shutdown_workers", None)
            if callable(shutdown):
                shutdown()
            loader._iterator = None

    def shutdown(self):
        self.close_pool()

        self._shutdown_loader(self.test)
        for loader in (self.train if isinstance(self.train, list) else [self.train]):
            self._shutdown_loader(loader)
        for loader in (self.val if isinstance(self.val, list) else [self.val]):
            self._shutdown_loader(loader)

        for p in self.participants + self.disqualified:
            self._shutdown_loader(getattr(p, "train", None))
            self._shutdown_loader(getattr(p, "val", None))
            p.train = None
            p.val = None

        self.train = None
        self.val = None
        self.test = None
        self.DATA = None
        gc.collect()
