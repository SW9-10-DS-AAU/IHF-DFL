
from enum import Enum

from parser.experiment_specs import ExperimentSpec


class Method(Enum):
    ACCURACY = 1
    NAIVE = 2
    DOTPRODUCT = 3
    DOTPRODUCTANDOUTLIER = 4

    @classmethod
    def from_string(cls, name: str, use_outlier: bool):
        key = name.upper()
        

        if key == "DOTPRODUCT":
            return cls.DOTPRODUCTANDOUTLIER if use_outlier else cls.DOTPRODUCT

        if key == "DOTPRODUCTANDOUTLIER":
            return cls.DOTPRODUCTANDOUTLIER

        try:
            return cls[key]
        except KeyError:
            raise ValueError(f"Invalid method: {name}")
    
    @classmethod
    def from_config(cls, config: ExperimentSpec):
        return cls.from_string(
            config.contribution_score_strategy,
            config.use_outlier_detection,
        )

    @property
    def display_name(self) -> str:
        return {
            Method.ACCURACY: "Accuracy",
            Method.NAIVE: "Naive",
            Method.DOTPRODUCT: "Dot product (no outlier detection)",
            Method.DOTPRODUCTANDOUTLIER: "Dot product (with outlier detection)",
        }[self]
    
    @property
    def short_name(self) -> str:
        return {
            Method.ACCURACY: "Accuracy",
            Method.NAIVE: "Naive",
            Method.DOTPRODUCT: "Without Outlier Detection",
            Method.DOTPRODUCTANDOUTLIER: "With Outlier Detection",
        }[self]
    