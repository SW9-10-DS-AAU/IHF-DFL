import numpy as np


def getVariances(values: list):
    return _minMax(values)


def _stdDeviation(values: list):
    if not values:
        return {"low": 0, "avg": 0, "high": 0, "NoValues": True}

    std = np.std(values, ddof=1)
    mean = np.mean(values)

    return {
        "low": std,
        "avg": mean,
        "high": std,
        "NoValues": False,
    }
    
def _percentiles(values):
    if not values:
        return {"low": 0, "avg": 0, "high": 0, "NoValues": True}

    mean = np.mean(values)

    return {
        "low": mean - np.percentile(values, 25),
        "avg": mean,
        "high": np.percentile(values, 75) - mean,
        "NoValues": False,
    }


def _minMax(values: list):
    if not values:
        return {"low": 0, "avg": 0, "high": 0, "NoValues": True}

    mean = np.mean(values)

    return {
        "low": mean - min(values),
        "avg": mean,
        "high": max(values) - mean,
        "NoValues": False,
    }