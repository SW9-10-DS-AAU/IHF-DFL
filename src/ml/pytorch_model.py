import gc
import sys
import signal
import atexit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import time
import ml.training as training
import ml.data as data
import ml.evaluation as evaluation
from utils.colors import green, red, yellow, b
from utils.printer import print_divider
from ml.cnn_models import Net_CIFAR, Net_MNIST
from ml.execution import (
    TrainingPlan,
    close_pools,
    create_cpu_pool,
    create_gpu_pools,
    print_cpu_pool_decision,
    print_system_capabilities,
    resolve_training_plan,
)
from ml.runtime import DEVICE, PIN_MEMORY
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
        self._gpu_pools = []
        self._pool_cleanup_registered = False
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
        self._capabilities_printed = False
        self._cpu_pool_decision = None
        print_divider("=")
        print("Pytorch Model created:\n")
        print(str(self.global_model))
        print() # New line before
        print_divider("=")

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


    def create_pool(self, plan: TrainingPlan | None = None):
        if self._pool is not None or self._gpu_pools:
            return

        if plan is None:
            plan, self._cpu_pool_decision = resolve_training_plan(len(self.participants), debugging)
        self._pool_size = plan.workers

        if not plan.parallel:
            return

        ctx = mp.get_context("spawn")
        if plan.reason == "multi_gpu":
            self._gpu_pools = create_gpu_pools(ctx, plan.num_gpus)
            print(green(f"Training worker plan: multi-GPU parallel, workers={self._pool_size} (one per GPU)"))
        elif plan.reason == "cpu_parallel":
            self._pool = create_cpu_pool(ctx, self._pool_size)
            print(green(f"Training worker plan: CPU parallel, workers={self._pool_size}"))
        else:
            return

        if not self._pool_cleanup_registered:
            atexit.register(self.close_pool)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            self._pool_cleanup_registered = True


    def print_training_plan(self, plan: TrainingPlan):
        if self._capabilities_printed:
            return

        print_system_capabilities(plan.num_gpus)
        if plan.num_gpus == 0:
            print_cpu_pool_decision(self._cpu_pool_decision)
        self._capabilities_printed = True

        if plan.reason == "debug":
            print(yellow("Debugging mode detected → running sequential training"))
        elif plan.reason == "single_gpu":
            print(yellow("Single GPU detected → running sequential training"))
        elif plan.reason == "cpu_single_worker":
            print(yellow("CPU detected → running sequential training"))



    def federated_training(self):
        plan, self._cpu_pool_decision = resolve_training_plan(len(self.participants), debugging)
        self.print_training_plan(plan)
        self.create_pool(plan)

        print(b(f"\n================ FEDERATED TRAINING START ================"))

        start_total = time.perf_counter()

        if plan.reason == "multi_gpu":
            results = self.run_multi_gpu(plan.num_gpus)
        elif plan.parallel:
            results = self.run_cpu_multiprocessing(plan.num_gpus)
        else:
            results = self.run_sequential(plan.num_gpus)

        self.apply_training_results(results)
        total_time = time.perf_counter() - start_total

        print(b(f"=================== TRAINING END ===================\n"))
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


    def run_sequential(self, num_gpus: int | None = None):
        if len(self.participants) == 0:
            raise RuntimeError("All participants have been disqualified - simulation cannot continue.")

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

        start_pool = time.perf_counter()
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
            else:  # If user's behaviour !good, skip Training.
                # Skips apply_training_results() - goes directly to evaluation. Corresponds to lines 261-277 in original code.
                evaluation.finalize_user_evaluation(self, user)

        print(green(f"Sequential execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def run_multi_gpu(self, num_gpus: int):
        if len(self.participants) == 0:
            raise RuntimeError("All participants have been disqualified - simulation cannot continue.")
        if not self._gpu_pools:
            raise RuntimeError("Multi-GPU training requested before GPU pools were created.")

        start_pool = time.perf_counter()
        async_results = []
        for idx, user in enumerate(self.participants):
            device_id = idx % num_gpus
            sd_cpu = {k: v.cpu() for k, v in user.model.state_dict().items()}

            if user.attitude == "good":  # train
                async_results.append(self._gpu_pools[device_id].apply_async(
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
            results = [r.get() for r in async_results]  # Collect results, waiting for all to finish.
        except KeyboardInterrupt:
            for pool in self._gpu_pools:
                pool.terminate()
                pool.join()
            self._gpu_pools = []
            raise

        print(green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def run_cpu_multiprocessing(self, num_gpus: int | None = None):
        if len(self.participants) == 0:
            raise RuntimeError("All participants have been disqualified - simulation cannot continue.")
        if self._pool is None:
            raise RuntimeError("CPU multiprocessing requested before the pool was created.")

        if num_gpus is None:
            num_gpus = torch.cuda.device_count()

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
            results = [r.get() for r in async_results]  # Collect results, waiting for all to finish.
        except KeyboardInterrupt:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
            raise

        print(green(f"Parallel execution time: {time.perf_counter() - start_pool:.2f} seconds"))
        return results


    def close_pool(self):
        close_pools(self._pool, self._gpu_pools)
        self._pool = None
        self._gpu_pools = []

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
