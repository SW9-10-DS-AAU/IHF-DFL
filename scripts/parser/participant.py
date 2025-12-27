from enum import IntEnum



class Attitude(IntEnum):
    GOOD = 1
    FREERIDER = 2
    BAD = 3

    @classmethod
    def from_string(cls, name: str):
        key = name.upper()

        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid method: {name}")

class MetaAttitude(IntEnum):
    GOOD = 1
    FREERIDER = 2
    BAD = 3
    BOTH = 4

class Participant:
  def __init__(self, _id, _currentAcc, _attitude, _futureAttitude, _attitudeSwitch, _address):
    self.id = _id
    self.attitude = _attitude
    self.futureAttitude = _futureAttitude
    self.attitudeSwitch = _attitudeSwitch
    self.address = _address
    self.states: list[ParticipantState] = []
    
class ParticipantState:
  def __init__(self, _id, _currentAcc, _attitude, _futureAttitude, _attitudeSwitch, _address):
    self.id = _id
    self.currentAcc = _currentAcc
    self.attitude = _attitude
    self.futureAttitude = _futureAttitude
    self.attitudeSwitch = _attitudeSwitch
    self.address = _address


def parse_attitude(s: str) -> Attitude:
    s = s.strip().lower()
    if s == "good":
        return Attitude.GOOD
    if s == "freerider":
        return Attitude.FREERIDER
    if s == "bad":
        return Attitude.BAD
    raise ValueError(f"Unknown attitude: {s}")

