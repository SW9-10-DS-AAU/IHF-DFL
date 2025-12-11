from dataclasses import dataclass
from typing import List

@dataclass
class GasStats:
    gas_feedback: List[int]
    gas_register: List[int]
    gas_slot: List[int]
    gas_weights: List[int]
    gas_close: List[int]
    gas_deploy: List[int]
    gas_exit: List[int]