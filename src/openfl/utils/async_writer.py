import csv
from datetime import datetime
from io import TextIOWrapper
import os
import threading
from queue import Queue, Full
from typing import List

def _time_handler():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

SPECIAL = {
    "time": _time_handler,
    "special": lambda: "special",   # example of another override
}

class AsyncWriter:
  # Remember to call finish on writer object
  def __init__(self, path, header: List[str], queue_size, config, author):
    print(f"Writing to {path}")
    self.csv_path = path
    self.header = header
    self.queue = Queue(maxsize=queue_size)
    self.author = author
    self.config = config

    self.thread = threading.Thread(target=self._writer, daemon=True)
    self.thread.start()

  # def writer(self):
  #     self.writeMetaAndHeaderIfEmpty()
  #     with open(self.csv_path, "a", newline="") as f:
  #         w = csv.writer(f)
  #         while True:
  #             item = self.queue.get()
  #             if item is None:  # poison pill
  #                 print("Empty Queue, breaking")
  #                 break

  #             row = [ SPECIAL[key]() if key in SPECIAL else item.get(key, "") for key in self.header ]

  #             print(f"Row {row}")

  #             w.writerow(row)
  #             self.queue.task_done()

  def _writer(self):
    self._writeMetaAndHeaderIfEmpty()
    with open(self.csv_path, "a", newline="") as f:
        w = csv.writer(f)

        while True:
            item = self.queue.get()
           
            if item is None: # stop signal
                f.flush()
                os.fsync(f.fileno())
                break

            row = [SPECIAL[key]() if key in SPECIAL else item.get(key, "") for key in self.header]
            print(f"Row {row}")

            w.writerow(row)
            self.queue.task_done()

            if self.queue.empty():
                f.flush()
                os.fsync(f.fileno())

  def _writeMetaAndHeaderIfEmpty(self):
      empty = (not os.path.exists(self.csv_path)) or os.path.getsize(self.csv_path) == 0
      if (empty):
        with open(self.csv_path, "w", newline="") as f:
            f.write(f"# Author {self.author}\n")
            self._write_config(f)
            w = csv.writer(f)
            w.writerow(self.header)

  def submitResult(self, item):
      try:
          self.queue.put(item, block=False)
      except Full:
          raise RuntimeError("writer queue overflow")

  def finish(self):
      # Wait until writer has processed every queued item
      self.queue.join()
      # Tell the writer thread to stop
      self.queue.put(None)
      # Wait until thread exits
      self.thread.join()

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
          ("first_round_fee", cfg.first_round_fee, "fee for first round"),
          ("fork", cfg.fork, "True=local fork, False=real net"),
          ("contribution_score_strategy", cfg.contribution_score_strategy, "scoring method"),
      ]
      file.writelines([
        f"# {name}: {value} ({desc})\n"
        for (name, value, desc) in fields
      ])

class NullWriter:
    def submitResult(self, *args, **kwargs):
        pass