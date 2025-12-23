
from enum import Enum


class Method(Enum):
    ACCURACY = 1
    NAIVE = 2
    DOTPRODUCT = 3
    DOTPRODUCTANDOUTLIER = 4

    @classmethod
    def from_string(cls, name: str, use_outlier: bool):
        key = name.upper()

        if key == "dotproduct":
            return cls.DOTPRODUCTANDOUTLIER if use_outlier else cls.DOTPRODUCT

        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid method: {name}")