import csv
from dataclasses import dataclass
from io import StringIO

@dataclass
class ExperimentSpec:
    # ---- Author ----
    author: str

    # ---- System Specs ----
    cpu_name: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    os_name: str = ""

    # ---- Experiment Config ----
    good_contributors: int = 0
    bad_contributors: int = 0
    freeriders: int = 0
    inactive: int = 0

    reward: int = 0
    minimum_rounds: int = 0

    min_buy_in: int = 0
    max_buy_in: int = 0
    standard_buy_in: int = 0

    epochs: int = 0
    batch_size: int = 0

    punish_factor: int = 0
    first_round_fee: int = 0
    fork: bool = False

    contribution_score_strategy: str = ""
    use_outlier_detection: bool = False

    freerider_start_round: int = 0
    freerider_noise_scale: float = 0.0

def parse_experiment_spec(csv_text: list[str]) -> ExperimentSpec:
    data = {}

    actualLines = [x for sub in csv_text for x in sub.split("#")]

    for line in actualLines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            line = line[1:].strip()

        if ":" not in line:
            continue

        key, rest = line.split(":", 1)
        value = rest.strip().split()[0]

        data[key.strip()] = value
    
    if data.get('TOTAL EXPERIMENT TIME') is None:
        raise ValueError("TOTAL EXPERIMENT TIME is missing, experiment probably failed")

    return ExperimentSpec(
        author=data.get("author", "Nykjaer"),

        cpu_name=data.get("cpu_name", ""),
        cpu_cores=int(data.get("cpu_cores", 0)),
        ram_gb=float(data.get("ram_gb", 0.0)),
        gpu_name=data.get("gpu_name", ""),
        os_name=data.get("os_name", ""),

        good_contributors=int(data.get("good_contributors", 0)),
        bad_contributors=int(data.get("bad_contributors", 0)),
        freeriders=int(data.get("freeriders", 0)),
        inactive=int(data.get("inactive", 0)),

        reward=int(data.get("reward", 0)),
        minimum_rounds=int(data.get("minimum_rounds", 0)),

        min_buy_in=int(data.get("min_buy_in", 0)),
        max_buy_in=int(data.get("max_buy_in", 0)),
        standard_buy_in=int(data.get("standard_buy_in", 0)),

        epochs=int(data.get("epochs", 0)),
        batch_size=int(data.get("batch_size", 0)),

        punish_factor=int(data.get("punish_factor", 0)),
        first_round_fee=int(data.get("first_round_fee", 0)),
        fork=data.get("fork", "False").lower() == "true",

        contribution_score_strategy=data.get("contribution_score_strategy", "err"),
        use_outlier_detection=data.get("use_outlier_detection", "False").lower() == "true",

        freerider_start_round=int(data.get("freerider_start_round", 0)),
        freerider_noise_scale=float(data.get("freerider_noise_scale", 0.0)),
    )
