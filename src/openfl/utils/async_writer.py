import csv
from datetime import datetime
from pathlib import Path
from io import TextIOWrapper
import os
import threading
from queue import Queue, Full
from typing import List
import platform
import psutil

from openfl.ml.pytorch_model import Participant

def _time_handler(item):
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

SPECIAL = {
    "time": _time_handler,
    "special": lambda item: "special",   # example of another override
}

class AsyncWriter:
    # Remember to call finish on writer object
    def __init__(self, path: Path, header: List[str], queue_size, config, author):
        # path.parent.mkdir()
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {path.absolute()}")
        self.csv_path = path
        self.header = header
        self.queue = Queue(maxsize=queue_size)
        self.author = author
        self.config = config

        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _writer(self):
        self._writeMetaAndHeaderIfEmpty()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)

            while True:
                item, isComment = self.queue.get()

                if item is None:  # stop signal
                    f.flush()
                    os.fsync(f.fileno())
                    self.queue.task_done()
                    print("stopping write queue")
                    break

                if isComment:
                    f.write(f"# {item}\n")
                else:
                    row = [SPECIAL[key](item) if key in SPECIAL else item.get(key, "") for key in self.header]
                    w.writerow(row)
                
                self.queue.task_done()

                if self.queue.empty():
                    f.flush()
                    os.fsync(f.fileno())
        print("Stopping Writer")

    def _writeMetaAndHeaderIfEmpty(self):
        empty = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
        if (empty):
            with open(self.csv_path, "w", newline="") as f:
                f.write(f"# Author {self.author}\n")
                self._write_hardware(f)
                self._write_config(f)
                w = csv.writer(f)
                w.writerow(self.header)

    def writeResult(self, item):
        try:
            self.queue.put((item, False), block=False)
        except Full:
            raise RuntimeError("writer queue overflow")

    def writeComment(self, comment: str):
        self.queue.put((comment, True), block=False)

    def finish(self):
        print("finish1")
        # Tell the writer thread to stop
        self.queue.put((None, False)) 
        print("finish2")
        # Wait for all queued tasks (including stop) to be marked done
        self.queue.join()
        print("finish3")
        # Now wait for the thread to exit
        self.thread.join()
        print("finish4")

    def _write_config(self, file: TextIOWrapper):
      cfg = self.config
      fields = [
          ("good_contributors", cfg.number_of_good_contributors, "honest participants"),
          ("bad_contributors", cfg.number_of_bad_contributors, "malicious participants"),
          ("freeriders", cfg.number_of_freerider_contributors, "contribute 0"),
          ("inactive", cfg.number_of_inactive_contributors, "never join"),
          ("reward", cfg.reward, "total reward pool"),
          ("minimum_rounds", cfg.minimum_rounds, "rounds to simulate"),
          ("min_buy_in", cfg.min_buy_in, "lower buy-in bound"),
          ("max_buy_in", cfg.max_buy_in, "upper buy-in bound"),
          ("standard_buy_in", cfg.standard_buy_in, "default buy-in"),
          ("epochs", cfg.epochs, "local epochs per round"),
          ("batch_size", cfg.batch_size, "training batch size"),
          ("punish_factor", cfg.punish_factor, "penalty multiplier"),
          ("punish_factor_contrib", cfg.punish_factor_contrib, "penalty multiplier contribution score"),
          ("force_merge_all", cfg.force_merge_all, "Whether to force merging all participants"),
          ("first_round_fee", cfg.first_round_fee, "fee for first round"),
          ("fork", cfg.fork, "True=local fork, False=real net"),
          ("contribution_score_strategy", cfg.contribution_score_strategy, "scoring method"),
          ("use_outlier_detection", cfg.use_outlier_detection, "Whether to use outlier detection"),
          ("freerider_start_round", cfg.freerider_start_round, "When freeriders should start freeriding"),
          ("freerider_noise_scale", cfg.freerider_noise_scale, "How much noise a freeder should add"),
      ]
      file.write("# ------Config--------")
      file.writelines([
        f"# {name}: {value} ({desc})\n"
        for (name, value, desc) in fields
      ])

    def _write_hardware(self, file):
        cpu_name = platform.processor()
        cpu_cores = psutil.cpu_count(logical=False)
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

        # GPU detection depends on your environment
        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        except ImportError:
            gpu_name = "Unknown"

        os_name = platform.system()

        fields = [
            ("cpu_name", cpu_name, "processor model"),
            ("cpu_cores", cpu_cores, "physical cores"),
            ("ram_gb", ram_gb, "system memory in GB"),
            ("gpu_name", gpu_name, "graphics processor"),
            ("os_name", os_name, "operating system"),
        ]

        file.write("# ------Specs--------")
        file.writelines([
            f"# {name}: {value} ({desc})\n"
            for (name, value, desc) in fields
        ])

class NullWriter:
    def submitResult(self, *args, **kwargs):
        pass