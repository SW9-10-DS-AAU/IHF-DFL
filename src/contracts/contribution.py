import math
import numpy as np
import torch
import warnings
from termcolor import colored
from decimal import Decimal
from utils.colors import green, red
from utils.shapley import check_shapley_compliance
from contracts import logging

_runtime_warnings = []


def contribution_score(challenge, _users, _current_round_no): # pragma: no cover
    """
    Compute contribution scores for all merging users, submit them to the
    contract, and log them. Strategy is chosen by _get_contribution_score_calculator:
      - legacy: simple dot-product
      - mad: MAD-based outlier filtering of weights
      - naive: equal-share (1 / num_mergers)
    """

    # Guard: no users → nothing to score
    if not _users:
        print("-----------------------------------------------------------------------------------")
        print("No users passed to contribution_score – skipping.")
        print("-----------------------------------------------------------------------------------")
        return

    print("Calculating contribution scores...\n")

    strategy = challenge.experiment_config.contribution_score_strategy

    if len(_users) <= 3:
        share = 1.0 / len(_users)
        msg = f"[Round {_current_round_no}] Too few contributors({len(_users)}) for contribution scoring – using equal shares({share:.4f} each)"
        print(colored(msg, "yellow"))
        logging.log_warning(challenge, msg)
        scores = [share] * len(_users)
        logging.log_contribution_scores(challenge, _users, scores, None, None, None)
        # for u in _users: u.evaluation_reward = 1
    else:
        if strategy not in _STRATEGIES:
            raise ValueError(f"Unknown contribution score strategy '{strategy}'. Available: {sorted(_STRATEGIES)}")
        scores = _STRATEGIES[strategy](challenge, _users, _current_round_no)

    challenge.scores = scores

    # if challenge.experiment_config.contribution_score_strategy != "loss_only":
    #     for u in _users: u.evaluation_reward = 1

    txs = []
    for u, score in zip(_users, challenge.scores):
        u.contribution_score = score
        scaled_contribution_score = int(Decimal(score) * Decimal("1e18"))
        # scaled_evaluation_score = int(Decimal(str(u.evaluation_reward)) * Decimal("1e18"))
        # if u.evaluation_reward == 0:
        #     raise ValueError(f"Evaluation reward for user {u.address} is zero, which will fail a require on the smart contract. User data: {u.__dict__}")

        if challenge.fork:
            tx = challenge.build_tx(u.address, challenge.modelAddress)
            tx_hash = challenge.model.functions.submitContributionScore(
                scaled_contribution_score
            ).transact(tx)
        else:
            nonce = challenge.w3.eth.get_transaction_count(u.address)
            cl = challenge.build_non_fork_tx(
                u.address,
                nonce,
            )
            cl = challenge.model.functions.submitContributionScore(
                scaled_contribution_score
            ).build_transaction(cl)
            pk = u.privateKey
            signed = challenge.w3.eth.account.sign_transaction(cl, private_key=pk)
            tx_hash = challenge.w3.eth.send_raw_transaction(signed.raw_transaction)
        txs.append(tx_hash)

    for i, txHash in enumerate(txs):
        challenge.track_transaction(i, txHash, len(txs), "contrib")

    for u in _users:
        print(green(f"\nUSER @ {u.address}"))
        print(green(f"{'CONTRIBUTION SCORE:':25}{u.contribution_score}"))
        # print(green(f"{'EVALUATION REWARD:':25}{u.evaluation_reward}")) if u.evaluation_reward is not None else None
    print("-----------------------------------------------------------------------------------\n")


def print_shapley_warnings():
    print(f"Number of Shapley Axioms violated: {len(_runtime_warnings)}\n")
    if _runtime_warnings:
        print("\n" + red("!" * 30 + " SHAPLEY WARNINGS " + "!" * 30))
        for warn in _runtime_warnings:
            print(colored(warn, "yellow"))
        print(red("!" * 78))


# ===== Strategy implementations =====

def _calculate_scores_dotproduct(challenge, users): # pragma: no cover
    """
    MAD-based scoring: robust per-weight outlier filtering before scoring.
    """
    merged_model = users[0].model
    # Both sides must use the same source (state_dict) so vector lengths stay aligned if buffers are added.
    global_update = torch.cat([v.view(-1) for v in merged_model.state_dict().values()])
    local_updates = [
        torch.cat([v.view(-1) for v in u.previousModel.values()]) for u in users
    ]
    local_updates = torch.stack(local_updates)

    use_outlier_detection = challenge.experiment_config.use_outlier_detection

    if use_outlier_detection:
        print("using mad")
        filtered_global_update, per_user_outlier_info = trim_global_update_using_mad(local_updates, global_update)
        scores = calc_contribution_scores_dotproduct(local_updates, filtered_global_update)

        # Raw dot product per user (pre-normalization), analogous to avg_acc/avg_loss in other strategies
        dots = torch.mv(local_updates, filtered_global_update)
        raw_values = [float(d.item()) for d in dots]
        logging.log_contribution_scores(challenge, users, scores, raw_values, per_user_outlier_info, None)
    else:
        print("not using mad")
        scores = calc_contribution_scores_dotproduct(local_updates, global_update)
        logging.log_contribution_scores(challenge, users, scores, None, None, None)

    return scores


def _calculate_scores_naive(challenge, users): # pragma: no cover
    """
    Equal-share scoring: everyone contributing gets 1 / num_mergers.
    """  # unused; included for signature consistency
    num_mergers = len(users)
    scores = [calc_contribution_score_naive(num_mergers) for _ in users]

    logging.log_contribution_scores(challenge, users, scores, None, None, None)

    return scores


def _calculate_scores_accuracy_loss(challenge, users, mad_threshold=1.1): # pragma: no cover
    """
    Accuracy-Loss-based scoring: use accuracy and loss directly as contribution score.
    """

    # accuracies: 1d array
    # losses: 1d array
    # prev_acc, prev_loss: int

    # Array of previous accuracies and losses from all users: A tuple of arrays
    prev_accuracies, prev_losses = challenge.get_all_previous_accuracies_and_losses()

    # use mad on these and average them

    mad_prev_accuracies = remove_outliers_mad(prev_accuracies, mad_threshold)
    mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold)

    avg_prev_acc = np.mean(mad_prev_accuracies)
    avg_prev_loss = np.mean(mad_prev_losses)

    avg_accuracies = []  # after loop: [30, 20, 30, 40]
    avg_losses = []  # after loop: [60, 70, 50, 80]

    for u in users:  # For loop to extract accuracies and loses.

        # All accuracies and loses per user
        _, accuracies, losses = challenge.get_all_accuracies_and_losses_about(u.address)

        try:
            # Multiple accuracies and losses per user
            mad_accuracies = remove_outliers_mad(accuracies, mad_threshold)
            mad_losses = remove_outliers_mad(losses, mad_threshold)

            # One average accuracy and loss per user
            avg_acc = np.mean(mad_accuracies)
            avg_loss = np.mean(mad_losses)

            avg_accuracies.append(avg_acc)  # int
            avg_losses.append(avg_loss)  # int
        except ValueError:
            print("An error occured")

    scores = []

    norm_accuracies = normalize_contribution_scores_old(avg_accuracies, avg_prev_acc)
    print(f"normalized accuracies: {norm_accuracies}")

    norm_losses = normalize_contribution_scores_old(avg_losses, avg_prev_loss)
    print(f"normalized losses: {norm_losses}")

    sum_na = sum(norm_accuracies)
    sum_nl = sum(norm_losses)

    print(f"sum_na: {sum_na}")
    print(f"sum_nl: {sum_nl}")

    for i in range(len(norm_accuracies)):
        res = (norm_accuracies[i] + norm_losses[i]) / (sum_na + sum_nl)
        score = res
        scores.append(score)

    print(f"scores = {scores}")
    return scores


# Output: An array of user scores
# Find out who was merged


def _calculate_scores_accuracy_only(challenge, users, _current_round_no, mad_threshold=1.1):
    """
    Accuracy-based scoring: use accuracy directly as contribution score.
    """

    # accuracies: 1d array
    # prev_acc: int

    # Array of previous accuracies from all users: A tuple of arrays
    prev_accuracies, _ = challenge.get_all_previous_accuracies_and_losses()

    # use mad on these and average them
    prev_info = {}
    mad_prev_accuracies = remove_outliers_mad(prev_accuracies, mad_threshold, collector=prev_info, label="previous")
    avg_prev_acc = np.mean(mad_prev_accuracies)
    avg_accuracies = []  # after loop: [30, 20, 30, 40]
    per_user_outlier_info = []

    for u in users:  # For loop to extract accuracies.
        # All accuracies per user
        _, accuracies = challenge.get_all_accuracies_about(u.address)

        try:
            # Multiple accuracies per user
            info = {}
            mad_accuracies = remove_outliers_mad(accuracies, mad_threshold, collector=info, label="current")
            # One average accuracy per user
            if len(mad_accuracies) == 0:
                raise ValueError("No accuracies left after MAD filtering for user {}".format(u.address))
            avg_acc = np.mean(mad_accuracies)
            avg_accuracies.append(avg_acc)  # int
            per_user_outlier_info.append({**prev_info,
                                          **info})  # Merge prev (global baseline) and current (per-user) MAD info into one dict; keys are prefixed ("previous_*" / "current_*") so they don't collide
        except Exception as e:
            per_user_outlier_info.append({})
            raise type(e)(f"Failed while processing user data: {e}") from e


    norm_accuracies = normalize_contribution_scores_new(avg_accuracies, avg_prev_acc, 'accuracy')
    print(f"normalized accuracies: {norm_accuracies}")

    # Validating Shapley Axioms (Runtime Guard)
    diffs = [v - avg_prev_acc for v in avg_accuracies]
    success, errors = check_shapley_compliance(diffs, norm_accuracies)

    if not success:
        msg = f"[Round {_current_round_no}] Axiom Violation: {errors}"
        _runtime_warnings.append(msg)
        print(colored(f"{msg}", "yellow"))
        logging.log_warning(challenge, msg)

    scores = norm_accuracies
    print(f"scores = {scores}")

    logging.log_contribution_scores(challenge, users, scores, avg_accuracies, per_user_outlier_info, avg_prev_acc)

    return scores


def _calculate_scores_loss_only(challenge, users, _current_round_no, mad_threshold=1.1):
    """
    Loss-based scoring: use loss directly as contribution score.
    """

    # losses: 1d array
    # prev_loss: int

    # Array of previous losses from all users: A tuple of arrays
    _, prev_losses = challenge.get_all_previous_accuracies_and_losses()

    # use mad on these and average them
    prev_info = {}
    mad_prev_losses = remove_outliers_mad(prev_losses, mad_threshold, collector=prev_info, label="previous")
    avg_prev_loss = np.mean(mad_prev_losses)
    avg_losses = []  # after loop: [60, 70, 50, 80]
    per_user_outlier_info = []
    # softmax_records = []
    # user_map = {u.address: u for u in users}
    # for u in users: u.evaluation_reward = 0

    for u in users:  # For loop to extract losses.
        # All loses per user
        voters, losses = challenge.get_all_losses_about(u.address)

        try:
            # Multiple accuracies and losses per user
            info = {}
            mad_losses = remove_outliers_mad(losses, mad_threshold, collector=info, label="current")
            # One average accuracy and loss per user
            if len(mad_losses) == 0:
                raise ValueError("No losses left after MAD filtering for user {}".format(u.address))
            avg_loss = np.mean(mad_losses)
            avg_losses.append(avg_loss)  # int
            per_user_outlier_info.append({**prev_info,
                                          **info})  # Merge prev (global baseline) and current (per-user) MAD info into one dict; keys are prefixed ("previous_*" / "current_*") so they don't collide
        except Exception as e:
            per_user_outlier_info.append({})
            raise type(e)(f"Failed while processing user data: {e}") from e


        # # Evaluation voting: convert loss votes into rewards using softmax, assign to users, and log
        # rewards = softmax_rewards(losses, avg_loss, 1, 0.01)
        # for voter_addr, loss_vote, reward in zip(voters, losses, rewards):
        #     if voter_addr in user_map:
        #         user_map[voter_addr].evaluation_reward += reward
        #         softmax_records.append({
        #             "evaluated_user":      u,
        #             "voter_user":          user_map[voter_addr],
        #             "loss_vote":           loss_vote,
        #             "avg_loss_true_value": avg_loss,
        #             "softmax_reward":      reward,
        #         })
        #     else:
        #         warnings.warn("Voter {} not found among merging users".format(voter_addr))

    norm_losses = normalize_contribution_scores_new(avg_losses, avg_prev_loss, 'loss')
    # print(f"normalized losses: {norm_losses}")

    # sum_nl = sum(norm_losses)

    # print(f"sum_nl: {sum_nl}")

    # Validating Shapley Axioms (Runtime Guard)
    diffs = [v - avg_prev_loss for v in avg_losses]
    diffs = [-1 * d for d in diffs]
    success, errors = check_shapley_compliance(diffs, norm_losses)

    if not success:
        msg = f"[Round {_current_round_no}] Axiom Violation: {errors}"
        _runtime_warnings.append(msg)
        print(colored(f"{msg}", "yellow"))
        logging.log_warning(challenge, msg)

    scores = norm_losses

    # print(f"scores = {scores}")
    # logging.log_evaluation_votes(challenge, softmax_records)
    logging.log_contribution_scores(challenge, users, scores, avg_losses, per_user_outlier_info, avg_prev_loss)

    return scores


# ===== Helper functions =====

def calc_contribution_score_naive(num_mergers) -> int: # pragma: no cover
    score = Decimal(1) / Decimal(num_mergers)
    return int(score * Decimal('1e18'))


def calc_contribution_scores_dotproduct(local_updates: torch.Tensor,
                                        global_update: torch.Tensor,
                                        eps: float = 1e-12): # pragma: no cover
    """
    Compute contribution scores solely using dot-product similarity
    between local updates and the global update.

    Args:
        local_updates: Tensor of shape (num_mergers, D)
                       flattened parameters for each user's local model.
        global_update: Tensor of shape (D,)
                       flattened parameters for the global model.
        eps:           Small tolerance to avoid division by zero.

    Returns:
        List[int]: contribution scores scaled by 1e18.
    """

    num_mergers, D = local_updates.shape

    # ||U||^2
    norm_U_sq = torch.dot(global_update, global_update)

    if norm_U_sq.abs() < eps:
        # If the global update has no magnitude → equal contribution
        score = Decimal(1) / Decimal(num_mergers)
        equal_int_score = int(score * Decimal('1e18'))
        return [equal_int_score for _ in range(num_mergers)]

    # Dot product for each user vs global update
    dots = torch.mv(local_updates, global_update)  # (num_mergers,)
    scores = dots / (num_mergers * norm_U_sq)

    # Convert to integer fixed-point (×1e18)
    return [
        int(Decimal(score.item()) * Decimal('1e18'))
        for score in scores
    ]


def normalize_contribution_scores_old(arr, prev_val): # pragma: no cover
    # This method takes a 1d array of an array (accuracy or loss), a scalar of previous accuracy or loss
    # Output is an array of normalized (according to sum) input array values
    # Takes a list of values
    # Subtracts a baseline (prev_val)
    # Normalizes them so they sum to 1
    # Example:
    # -- arr - prev_val => norm_arr = [2, 1, 0]
    # -- sum = 3
    # -- [2/3, 1/3, 0/3]

    norm_arr = []
    sum_val = 0.0

    for i in range(len(arr)):
        norm_arr.append(arr[i] - prev_val)
        sum_val += norm_arr[i]

    if len(norm_arr) == 0:
        raise Exception("No values to normalize")
    for i in range(len(norm_arr)):
        if sum_val == 0.0:
            return [1.0 / len(norm_arr)] * len(norm_arr)
        norm_arr[i] /= sum_val
    return norm_arr


def normalize_contribution_scores_new(vals: list, prev_val: float, evaluation_metric: str) -> list:
    """
    4-step normalization for contribution scores.

    1. Subtract baseline, then negate if metric is 'loss' (lower=better → flip sign).
    2. Edge cases: if max==0 replace zeros with 1; if all negative compute sum/val ratios.
    3. Clamp negatives so the minimum is exactly -1.
    4. Final normalization to sum=1: divide by sum if all-positive, otherwise
       redistribute the excess proportionally across positive values.
    """

    # Step 1: subtract baseline, flip sign for loss
    vals = [v - prev_val for v in vals] # Handle the subtraction of new minus prev here
    if evaluation_metric == "loss":
        vals = [-1 * val for val in vals]
    sum_ = sum(vals)

    # Step 2: edge cases
    max_val = max(vals)
    if max_val == 0:
        vals = [1 if val == 0 else val for val in vals]
    elif max_val < 0:
        vals = [sum_ / val for val in vals]
    # elif sum_ < 0:
    # vals = [val / -sum_ for val in vals]


    # Step 3: clamp negatives to minimum -1
    if min(vals) < -1:
        divisor = -min(vals)
        vals = [val / divisor if val < 0 else val for val in vals]
    sum_ = sum(vals)

    # Step 4: normalize to sum = 1
    if not sum_ == 1:  #
        if min(vals) >= 0:  # if all positive
            vals = [val / sum_ for val in vals]
        else:
            sum_of_positives = sum(val for val in vals if val > 0)
            excess_sum = sum_ - 1
            vals = [val + (val / sum_of_positives) * -excess_sum if val > 0 else val for val in vals]
    return vals


def softmax_rewards(values, true_value, total_reward, alpha):
    distances = [abs(v - true_value) for v in values]
    weights = [math.exp(-alpha * d) for d in distances]
    total_weight = sum(weights)
    rewards = [total_reward * w / total_weight for w in weights]
    return rewards


def remove_outliers_mad(arr, threshold=0.70, return_mask=False, collector=None, label=None): # pragma: no cover
    # Keep original dtype (int from contract uint256). np.median returns float64
    # automatically, so all intermediate MAD arithmetic stays in float without
    # needing to cast the input array.
    arr = np.asarray(arr)

    # always flatten
    flat = arr.ravel()

    median = np.median(flat)
    abs_dev = np.abs(flat - median)
    mad = np.median(abs_dev)

    prefix = f"{label}_" if label else "" # Set label if provided else empty string

    # SPECIAL CASE: MAD == 0
    if mad == 0:
        mask = abs_dev <= threshold
        if collector is not None:
            collector[f"{prefix}median"]   = float(median)
            collector[f"{prefix}mad"]      = 0.0
            collector[f"{prefix}removed"]  = flat[~mask].tolist()
            collector[f"{prefix}accepted"] = flat[mask].tolist()
            collector[f"{prefix}boundary"] = None
        if return_mask:
            return arr, mask
        return flat[mask]

    # proper modified z-score
    z_val = 0.6745
    modified_z = z_val * (flat - median) / mad

    mask = np.abs(modified_z) <= threshold

    if collector is not None:
        collector[f"{prefix}median"]   = float(median)
        collector[f"{prefix}mad"]      = float(mad)
        collector[f"{prefix}removed"]  = flat[~mask].tolist()
        collector[f"{prefix}accepted"] = flat[mask].tolist()
        collector[f"{prefix}boundary"] = float(threshold * mad / z_val)

    if return_mask:
        return arr, mask
    return flat[mask]


def trim_global_update_using_mad(local_updates: torch.Tensor, # pragma: no cover
                                 global_update: torch.Tensor,
                                 mad_thresh: float = 3.5,
                                 eps: float = 1e-12):
    """
    Trim the global update by removing (zeroing) weights where
    all clients are outliers according to MAD filtering.

    Args:
        local_updates: Tensor (num_mergers, D)
        global_update: Tensor (D,)
        mad_thresh: MAD robust z-score threshold
        eps: avoid divide-by-zero

    Returns:
        filtered_global_update: Tensor (D,)
        per_user_outlier_info: list of dicts (one per user) with MAD stats
    """

    num_mergers, D = local_updates.shape

    # Per-weight median
    median = local_updates.median(dim=0).values  # (D,)

    # Per-weight absolute deviation
    abs_dev = (local_updates - median).abs()  # (num_mergers, D)

    # MAD per weight
    mad = abs_dev.median(dim=0).values  # (D,)
    safe_mad = mad.clone()
    safe_mad[safe_mad < eps] = eps

    # Per weight/user robust z-score
    robust_z = 0.6745 * abs_dev / safe_mad

    # Non-outlier mask (True = keep)
    mask = robust_z <= mad_thresh  # (num_mergers, D)

    # Collapse user dimension: keep weight if ANY user is non-outlier
    global_mask = mask.any(dim=0)  # (D,)

    # Zero out outlier-only weights in global update
    filtered_global_update = global_update * global_mask

    # Per-user summary stats for logging
    mad_mean = float(mad.mean().item())
    median_mean = float(median.mean().item())
    per_user_outlier_info = [
        {
            "current_median": median_mean,
            "current_mad": mad_mean,
            "current_boundary": mad_thresh,
            # Weight-space outlier counts (not scalar value lists — stored under distinct keys)
            "dotproduct_outlier_weight_count": int((~mask[i]).sum().item()),
            "dotproduct_outlier_weight_fraction": float((~mask[i]).float().mean().item()),
        }
        for i in range(num_mergers)
    ]

    return filtered_global_update, per_user_outlier_info


# ===== Strategy registry =====

_STRATEGIES = {
     "dotproduct":    _calculate_scores_dotproduct,
     "naive":         _calculate_scores_naive,
     "accuracy_loss": _calculate_scores_accuracy_loss,
     "accuracy_only": _calculate_scores_accuracy_only,
     "loss_only":     _calculate_scores_loss_only,
 }
