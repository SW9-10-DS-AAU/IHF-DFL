from datetime import datetime, timedelta

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
    self.disqualifiedUsers = _disqualifiedUsers