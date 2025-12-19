from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class Round:
  def __init__(
    self,
    _nr: int,
    _globalAcc: float,
    _globalLoss: float,
    _attitude,
    _time: datetime,
    _GRS: list[float],
    _accAvgPerUser: list[float],
    _lossAvgPerUser: list[float],
    _rewards: list[float],
    _conctractBalanceRewards: list[float],
    _punishments: list[tuple[str, int, int, int]],
    _contributionScores: list[float],
    _feedbackMatrix: list[list[int]],
    _disqualifiedUsers: list[int],
    _gasTransactions: list[tuple[str, str, int]],
    _lastTime: datetime
):
    self.nr = _nr
    self.time = _time
    self.timeDelta = timedelta(0) if _lastTime is None else _time - _lastTime
    self.globalAcc = _globalAcc
    self.globalLoss = _globalLoss
    self.attitude = _attitude
    self.GRS = _GRS
    self.accAvgPerUser = _accAvgPerUser
    self.lossAvgPerUser = _lossAvgPerUser
    self.rewards = _rewards
    self.conctractBalanceRewards = _conctractBalanceRewards
    self.punishments = _punishments
    self.contributionScores = _contributionScores
    self.feedbackMatrix = _feedbackMatrix
    self.disqualifiedUsers = _disqualifiedUsers,
    self.gasTransactions = [GasCostsFormatted(GasType[_gasTransaction[0]], _gasTransaction[1], _gasTransaction[2]) for _gasTransaction in _gasTransactions]

class GasType(Enum):
    register = 0
    slot = 1
    weights = 2
    feedback = 3
    contrib = 4
    close = 5
    exit = 6

@dataclass
class GasCostsFormatted:
  type: GasType
  txHash: str
  amount: int

  def __post_init__(self):
        if isinstance(self.type, str):
            self.type = GasType[self.type]

  def __add__(self, other):
        if not isinstance(other, GasCostsFormatted):
            return NotImplemented
        return GasCostsFormatted(self.type, None, self.amount + other.amount)
  
  def __radd__(self, other):
    if other == 0:
        return self
    return self.__add__(other)