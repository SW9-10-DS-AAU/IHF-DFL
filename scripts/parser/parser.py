import ast
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterator, Tuple

from .experiment_specs import parse_experiment_spec

from .gasCosts import GasStats

from .types.participant import Participant, ParticipantState, parse_attitude
from .types.round import Round

def _filtered_lines(path: Path) -> Tuple[list[str], list[str]]:
    kept = []
    filtered = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                filtered.append(line)
            else:
                kept.append(line)

    return kept, filtered

def load_data(path: str):
  rounds: list[Round] = []
  participants: dict[int, Participant] = {}

  last_time = None

  filtered, leftOver = list(_filtered_lines(path))
  reader = csv.DictReader(filtered)

  for row in reader:
      nr = int(row["round"])
      time = datetime.strptime(row["time"], "%H:%M:%S.%f")
      def parse(x):
          try:
              return ast.literal_eval(x)
          except Exception:
              return x

      round_obj = Round(
          _nr=nr,
          _globalAcc=float(row["globalAcc"]),
          _globalLoss=float(row["globalLoss"]),
          _attitude=None,
          _time=time,
          _GRS=parse(row["GRS"]),
          _accAvgPerUser=parse(row["accAvgPerUser"]),
          _lossAvgPerUser=parse(row["lossAvgPerUser"]),
          _rewards=parse(row["rewards"]),
          _conctractBalanceRewards=parse(row["conctractBalanceRewards"]),
          _punishments=parse(row["punishments"]),
          _contributionScores=parse(row["contributionScores"]),
          _feedbackMatrix=parse(row["feedbackMatrix"]),
          _disqualifiedUsers=parse(row["disqualifiedUsers"]), # This is wrong, but a workaround has been found, and this gets overwritten later
          _gasTransactions=parse(row["GasTransactions"]),
          _lastTime=last_time
      )

      rounds.append(round_obj)
      last_time = time

      # ---- USER STATUSES ----
      user_status_list = ast.literal_eval(row["userStatuses"])

      for entry in user_status_list:
          uid, curAcc, att, futAtt, attSwitch, addr = parse_user_status(entry)

          if uid not in participants:
              participants[uid] = Participant(uid, curAcc, att, futAtt, attSwitch, addr)

          state = ParticipantState(uid, curAcc, att, futAtt, attSwitch, addr)
          participants[uid].states.append(state)
    
    
  gasStats = parse_gas_stats(next((s.replace("# $gasCosts$", "") for s in leftOver if s.startswith("# $gasCosts$")), None))
  experimentConfig = parse_experiment_spec(leftOver)

  return rounds, participants, gasStats, experimentConfig

def parse_user_status(entry: str):
    parts = [p.strip() for p in entry.split(",")]

    user_id = int(parts[0].replace("$user$", ""))

    currentAcc = float(parts[1])

    attitude = parse_attitude(parts[2])
    futureAttitude = parse_attitude(parts[3])

    attitudeSwitch_raw = parts[4]
    attitudeSwitch = None if attitudeSwitch_raw == "None" else int(attitudeSwitch_raw)

    address = parts[5]

    return user_id, currentAcc, attitude, futureAttitude, attitudeSwitch, address

def parse_gas_stats(line: str) -> GasStats:
    if line is None:
        return
    line = line.replace("#", "")
    # Split on commas *only* between top-level lists
    parts = []
    buf = ""
    depth = 0

    for ch in line.strip():
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1

        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch

    if buf:
        parts.append(buf.strip())

    if len(parts) != 7:
        raise ValueError(f"Expected 7 list fields, got {len(parts)} in {line}")

    lists = [ast.literal_eval(p) for p in parts]

    return GasStats(*lists)