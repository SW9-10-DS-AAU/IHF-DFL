import copy
import math
import torch
import numpy as np
import ml.training as training
from utils import aggregation_strategy_parser
from utils.colors import yellow, red, b
from ml.runtime import DEVICE
from typing import Callable


def the_merge(pm, _users, aggregation_rule: str, merge_weight_collector=None, agg_switch_collector=None,
              avg_prior_losses=None, warning_collector=None):
    # No qualified users → skip merge this round
    if not _users:
        msg = f"[Round {pm.round - 1}] No participants qualified for merge – skipping aggregation"
        print("-----------------------------------------------------------------------------------")
        print(red(msg))
        print("-----------------------------------------------------------------------------------\n")
        if warning_collector is not None:
            warning_collector.append(msg)
        return

    pre_merge_snapshot = copy.deepcopy(pm.global_model)

    pm.two_previous_global_model = pm.previous_global_model
    pm.previous_global_model = pre_merge_snapshot

    client_models, users_contribution_scores, users_merge_weights = [], {}, {}

    for u in _users:
        client_models.append(u.model)
        # print("Account {} participating in merge".format(u.address[0:16] + "..."))
        users_contribution_scores[u.address] = u.contribution_score

    print("Using aggregation rule: {}".format(aggregation_rule))

    # -------------------------
    # Compute weights (UNIFIED)
    # -------------------------
    n_clients = len(client_models)

    # Agg. strategies not using users_contrib_scores are fixed with a lambda capturing the input each function need.
    _fedavg_fn = lambda _: {u.address: 1.0 / n_clients for u in _users}
    _fedavg_fn.__name__ = "FedAVG"
    _grs_fn = lambda _: GRS_aggregation(_users)
    _grs_fn.__name__ = "GRS_aggregation"
    agg_rules = {
        "FedAVG": _fedavg_fn,
        "positives_only": positives_only,
        "plus_one_normalize": plus_one_normalize,
        "plus_more_than_one_normalize": plus_more_than_one_normalize,
        "GRS_aggregation": _grs_fn,
    }

    _low_contributor_fallback = len(_users) <= 3 and aggregation_rule != "FedAVG"
    # Note: We include FedAVG here so in case of partial/binary_switch paired with GRS_agg. don't become 2xGRS_agg.
    if _low_contributor_fallback:
        msg = f"[Round {pm.round - 1}] Too few contributors ({len(_users)}) – defaulting to GRS_aggregation"
        print(yellow(msg))
        if warning_collector is not None:
            warning_collector.append(msg)
        aggregation_rule = "GRS_aggregation"

    if aggregation_rule in agg_rules:
        users_merge_weights = agg_rules[aggregation_rule](users_contribution_scores)
        if agg_switch_collector is not None:
            agg_switch_collector.update({
                "func_1": aggregation_rule,
                "weight_1": 1.0,
                "func_2": None,
                "weight_2": None,
            })

    else:
        parsed_agg_strategy_value = aggregation_strategy_parser.parse_values(aggregation_rule)
        switch_type = parsed_agg_strategy_value[0]
        # Resolve func name strings to callables using agg_rules
        func1_name = parsed_agg_strategy_value[1]
        func2_name = parsed_agg_strategy_value[2]
        if func1_name not in agg_rules:
            raise ValueError(f"Unknown aggregation function '{func1_name}' in switch rule: {aggregation_rule}")
        if func2_name not in agg_rules:
            raise ValueError(f"Unknown aggregation function '{func2_name}' in switch rule: {aggregation_rule}")
        func1 = agg_rules[func1_name]
        func2 = agg_rules[func2_name]

        if switch_type == "binary_switch":
            users_merge_weights = binary_switch(pm, users_contribution_scores, func1, func2, agg_switch_collector)

        elif switch_type.startswith("partial_switch"):
            users_merge_weights = invoke_partial_switch(pm, users_contribution_scores,
                                                             switch_type, func1, func2, avg_prior_losses,
                                                             agg_switch_collector)
        else:
            raise ValueError(f"Unknown merge strategy: {aggregation_rule}")

    for u in _users:
        u.merge_weight = users_merge_weights[u.address]

    if merge_weight_collector is not None:
        merge_weight_collector.update(users_merge_weights)

    assert abs(sum(users_merge_weights.values()) - 1.0) < 1e-6, "Aggregation weights must sum to 1"

    # -------------------------
    # Cache client state_dicts (IMPORTANT OPTIMIZATION)
    # -------------------------
    client_state_dicts = [m.state_dict() for m in client_models]
    ordered_weights = [users_merge_weights[u.address] for u in _users]

    with torch.no_grad():
        global_dict = pm.global_model.state_dict()

        for k in global_dict.keys():
            # Stack all client parameters
            stacked = torch.stack([
                client_state_dicts[i][k].to(
                    device=global_dict[k].device,
                    dtype=global_dict[k].dtype
                )
                for i in range(n_clients)
            ], dim=0)

            # Prepare weights tensor (once per param for correct device/dtype)
            w = torch.tensor(ordered_weights, device=stacked.device, dtype=stacked.dtype)
            w = w.view(-1, *([1] * (stacked.dim() - 1)))

            # Weighted aggregation (covers ALL rules including FedAvg)
            global_dict[k] = (stacked * w).sum(0)

        pm.global_model.load_state_dict(global_dict)

    # -------------------------
    # Evaluation
    # -------------------------
    loss, accuracy = training.test(pm.global_model, pm.test, DEVICE)
    pm.accuracy.append(accuracy)
    pm.loss.append(loss)

    for u in _users:
        print(f"User {u.address[0:16]}... merge_weight: {users_merge_weights[u.address]:.4f}")

    print("-----------------------------------------------------------------------------------")
    print(b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy * 100, loss)))

    # -------------------------
    # Distribute global model
    # -------------------------
    for u in pm.participants:
        u.previousModel = copy.deepcopy(u.model)
        u.model.load_state_dict(pm.global_model.state_dict())

    print("-----------------------------------------------------------------------------------\n")


def models_are_equal(model_a, model_b):
    previous_model_params = model_a.state_dict()
    pre_previous_model_params = model_b.state_dict()

    if previous_model_params.keys() != pre_previous_model_params.keys():
        return False

    return all(
        torch.equal(previous_model_params[x], pre_previous_model_params[x])
        for x in previous_model_params
    )


def invoke_partial_switch(pm, users_contrib_scores, switch_type: str, func1: Callable, func2: Callable, avg_prior_losses=None, agg_switch_collector=None):
    loss_based = {
        "partial_switch_fixed_loss": partial_switch_fixed_loss,
        "partial_switch_retrospective": partial_switch_loss_retrospective,
    }

    if switch_type == "partial_switch_accuracy":
        return partial_switch_accuracy(pm, users_contrib_scores, func1, func2, agg_switch_collector)

    if switch_type in loss_based:
        if not avg_prior_losses:  # None or empty list (early rounds)
            print(yellow(f"Warning: Missing prior losses for {switch_type}. Defaulting to func1."))
            if agg_switch_collector is not None:
                agg_switch_collector.update({"func_1": func1.__name__, "weight_1": 1.0, "func_2": func2.__name__, "weight_2": 0.0})
            return func1(users_contrib_scores)
        if switch_type == "partial_switch_fixed_loss":
            return partial_switch_fixed_loss(users_contrib_scores, avg_prior_losses[0], func1, func2, agg_switch_collector=agg_switch_collector)
        return partial_switch_loss_retrospective(users_contrib_scores, avg_prior_losses, func1, func2, agg_switch_collector=agg_switch_collector)

    raise ValueError(f"Unknown partial switch type: {switch_type}")


def binary_switch(pm, users_contrib_scores, func_1, func_2, agg_switch_collector):
    if not pm.has_switched and pm.round > 1 and pm.two_previous_global_model is not None:
        if models_are_equal(pm.previous_global_model, pm.two_previous_global_model):
            pm.has_switched = True
            print(
                f"  [binary_switch] Convergence detected at round {pm.round - 1}: Switching from {func_1.__name__} to {func_2.__name__}")

    use_func_2 = pm.has_switched
    active_func = func_2 if use_func_2 else func_1

    print(f"  [binary_switch] At round {pm.round - 1}: Using {active_func.__name__}")

    if agg_switch_collector is not None:
        agg_switch_collector.update({
            "func_1": func_1.__name__,
            "weight_1": 0.0 if use_func_2 else 1.0,
            "func_2": func_2.__name__,
            "weight_2": 1.0 if use_func_2 else 0.0,
        })

    return active_func(users_contrib_scores)

def partial_switch_accuracy(pm, users_contrib_scores, func_1, func_2, agg_switch_collector):

    accuracy_measure = pm.accuracy[-1]  # latest accuracy
    # Weight for func_2 (e.g. 0.47 → 47% func_2, 53% func_1)

    weights_1 = func_1(users_contrib_scores)
    weights_2 = func_2(users_contrib_scores)

    mixed_weights = {
        key: (1 - accuracy_measure) * weights_1[key] + accuracy_measure * weights_2[key]
        for key in users_contrib_scores
    }

    # Normalizing
    total = sum(mixed_weights.values())
    mixed_weights = {key: value / total for key, value in mixed_weights.items()}

    if agg_switch_collector is not None:
        agg_switch_collector.update({
            "func_1": func_1.__name__,
            "weight_1": 1 - accuracy_measure,
            "func_2": func_2.__name__,
            "weight_2": accuracy_measure,
        })

    return mixed_weights

def partial_switch_fixed_loss(users_contrib_scores, avg_prior_losses, func_1, func_2, threshold=100,
                              agg_switch_collector=None):
    # ratio = loss / threshold: high when loss is high (early training), low when loss is low (late training).
    # alpha = sin(ratio * 90°): weight for func_1 (strict/conservative, e.g. positives_only).
    #   sin(90°) = 1 → loss at or above threshold → fully func_1
    #   sin(0°)  = 0 → loss near zero             → fully func_2
    # Using ratio = loss/threshold (not 1 - loss/threshold) keeps func_1 dominant for longer —
    # the sine curve stays high across most of the loss range and only drops sharply near zero.
    # This matches the intent: use positives_only throughout training, switch to plus_one_normalize
    # only once loss is genuinely low.
    if avg_prior_losses is None or avg_prior_losses >= threshold:
        ratio = 1.0
        alpha = 1.0
    else:
        ratio = avg_prior_losses / threshold
        alpha = math.sin(math.radians(ratio * 90.0))

    beta = 1.0 - alpha  # weight for func_2 (soft/rewarding, e.g. plus_one_normalize)

    print(f"partial_switch_fixed_loss: prior_loss={avg_prior_losses}, threshold={threshold}, ratio={ratio:.3f}, "
          f"alpha(func1)={alpha:.3f}, beta(func2)={beta:.3f}")

    if agg_switch_collector is not None:
        agg_switch_collector.update({
            "func_1": func_1.__name__,
            "weight_1": alpha,
            "func_2": func_2.__name__,
            "weight_2": beta,
        })

    res1 = func_1(users_contrib_scores)
    res2 = func_2(users_contrib_scores)

    combined = {
        addr: alpha * res1[addr] + beta * res2[addr]
        for addr in users_contrib_scores
    }

    total = sum(combined.values())
    if total <= 0:
        n = len(combined)
        return {addr: 1.0 / n for addr in combined}

    return {addr: w / total for addr, w in combined.items()}


# takes an array of accuracy/losses. Only losses since we do not store accuracy.
def partial_switch_loss_retrospective(users_contrib_scores, avg_prior_losses, func_1, func_2,
                        agg_switch_collector=None):
    res1 = func_1(users_contrib_scores)  # e.g. positives_only    — strict, filters noise
    res2 = func_2(users_contrib_scores)  # e.g. plus_one_normalize — soft, rewards broadly

    # How much did loss improve from the prior round to the previous round?
    # Normalise by prior loss so the ratio is scale-independent, clamp to [0, 1].

    if len(avg_prior_losses) >= 2:
        x = np.arange(len(avg_prior_losses))
        slope, _ = np.polyfit(x, list(reversed(avg_prior_losses)), 1)  # now negative = improving
        # normalize slope relative to mean loss so it's scale-independent
        mean_loss = np.mean(avg_prior_losses)
        if mean_loss == 0:
            improvement_ratio = 1.0  # loss already zero → maximum improvement
        else:
            improvement_ratio = max(0.0, min(1.0, -slope / mean_loss))
    else:
        improvement_ratio = 1.0  # not enough data, go strict

    # Map ratio → angle [0°, 90°] → sine blend weight.
    # sin(0°) = 0  → loss plateauing  → fully func_2 (soft/rewarding)
    # sin(90°) = 1 → max improvement  → fully func_1 (strict/conservative)
    # Sine gives a slow start and fast finish: cautious early, more aggressive as convergence grows.
    alpha = math.sin(math.radians(
        improvement_ratio * 90.0))  # weight for func_1   0.33 → 30° → sin(30°)=0.5
    beta = 1.0 - alpha  # weight for func_2  1 - 0.5 = 0.5

    print(f"partial_switch: losses={[f'{l:.2f}' for l in avg_prior_losses]}, "
          f"improvement={improvement_ratio:.3f}, alpha(func1)={alpha:.3f}, beta(func2)={beta:.3f}")

    if agg_switch_collector is not None:
        agg_switch_collector.update({
            "func_1": func_1.__name__,
            "weight_1": alpha,
            "func_2": func_2.__name__,
            "weight_2": beta,
        })

    # Blend the two score dicts linearly by (alpha, beta)
    combined = {
        addr: alpha * res1[addr] + beta * res2[addr]
        for addr in users_contrib_scores
    }

    # Re-normalise: each func's output sums to 1 individually, but their
    # linear combination may not — especially if func_1 zeroed some users out.
    # 0.5+0.87=1,37
    total = sum(combined.values())
    if total <= 0:
        n = len(combined)
        return {addr: 1.0 / n for addr in combined}
    return {addr: w / total for addr, w in combined.items()}


# OLD — BUG: division by zero when all scores <= 0
def positives_only(users_contrib_scores: dict):
     positive_sum = sum(score for score in users_contrib_scores.values() if score > 0)
     if positive_sum <= 0:
         raise Exception("No positive contribution scores; cannot normalize")
     aggregation_scores = {user_id: score / positive_sum if score > 0 else 0 for user_id, score in users_contrib_scores.items()}
     return aggregation_scores


# OLD — BUG: denominator n+1 is only correct when scores already sum to 1
def plus_one_normalize(users_contrib_scores: dict):
     # normalized_scores = [score + 1 for score in users_contrib_scores]
     normalized_scores = {user_id: score + 1 for user_id, score in users_contrib_scores.items()}
     sum_ = sum(normalized_scores.values())
     aggregation_scores = {user_id: score / sum_ for user_id, score in normalized_scores.items()}
     return aggregation_scores


# OLD — BUG: denominator n*more_than_one+1 is only correct when scores already sum to 1
def plus_more_than_one_normalize(users_contrib_scores: dict, more_than_one=1.1):
     normalized_scores = {user_id: score + more_than_one for user_id, score in users_contrib_scores.items()}
     sum_ = sum(normalized_scores.values())
     aggregation_scores = {user_id: score / sum_ for user_id, score in normalized_scores.items()}
     return aggregation_scores


def GRS_aggregation(users):
    total_grs = sum(user._globalrep[-1] for user in users)
    aggregation_scores = {user.address: user._globalrep[-1] / total_grs for user in users}
    return aggregation_scores
