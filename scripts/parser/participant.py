from enum import IntEnum



class Attitude(IntEnum):
    GOOD = 1
    FREERIDER = 2
    MALICIOUS = 3

    @classmethod
    def from_string(cls, name: str):
        key = name.upper()

        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid method: {name}")
        
    @property
    def display_name(self) -> str:
        return {
            Attitude.GOOD: "Honest",
            Attitude.FREERIDER: "Freerider",
            Attitude.MALICIOUS: "Malicious",
        }[self]

class MetaAttitude(IntEnum):
    GOOD = 1
    FREERIDER = 2
    MALICIOUS = 3
    BOTH = 4

    @classmethod
    def from_string(cls, name: str, use_outlier: bool):
        key = name.upper()

        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid method: {name}")

    @property
    def display_name(self) -> str:
        return {
            MetaAttitude.GOOD: "Honest",
            MetaAttitude.FREERIDER: "Freerider",
            MetaAttitude.MALICIOUS: "Malicious",
            MetaAttitude.BOTH: "Both have been kicked",
        }[self]

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
    if s == "bad" or s == "malicious":
        return Attitude.MALICIOUS
    raise ValueError(f"Unknown attitude: {s}")

