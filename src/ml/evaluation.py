import numpy as np
import ml.training as training
import math
from utils.colors import green, red, yellow, b, rb
from web3 import Web3
from ml.runtime import DEVICE


def exchange_models(pm):
    print("Users exchanging models...")
    for user in pm.participants:
        user.userToEvaluate = []
        for j in pm.participants:
            if user.model == j.model:
                continue
            if j.model in user.userToEvaluate:
                continue
            user.userToEvaluate.append(j)
    print("-----------------------------------------------------------------------------------")


def verify_models(pm, on_chain_hashes):
    print("Users verifying models...")
    for _user in pm.participants:
        _user.cheater = []
        for user in _user.userToEvaluate:
            if not get_hash(user.model.state_dict()) == on_chain_hashes[user.id]:
                print(
                    red(f"Account {_user.id}: Account {user.address[0:16]}... could not provide the registered model"))
                _user.cheater.append(user)

    print("-----------------------------------------------------------------------------------")


def get_hash(_state_dict):
    if not isinstance(_state_dict, dict):
        _state_dict = dict(_state_dict)

    parts = []
    for k, v in sorted(_state_dict.items(), key=lambda x: x[0]):
        t = v.detach()
        if t.is_cuda:
            t = t.cpu()
        t = t.contiguous()
        parts.append(k.encode("utf-8"))
        parts.append(b"|")
        # include shape to avoid accidental collisions
        parts.append(np.asarray(t.shape, dtype=np.int64).tobytes())
        parts.append(b"|")
        parts.append(t.numpy().tobytes())
        parts.append(b"\n")
    blob = b"".join(parts)
    return Web3.keccak(blob)  # remove hex to match old, with improved algo.


def evaluate_peers(pm):
    print("Users evaluating models...")

    scalar = 100  # Adds more decimals for precision (Adding 0 gives another decimal, vice versa)
    MAX_UINT16_SIZE = 65535
    count_dq = len(pm.disqualified)

    feedback_matrix = np.zeros((1, len(pm.participants) + count_dq, len(pm.participants) + count_dq))[0]
    n = len(pm.participants) + count_dq
    accuracy_matrix = [[0 for _ in range(n)] for _ in range(n)]
    loss_matrix = [[0 for _ in range(n)] for _ in range(n)]
    prev_accs = [0 for _ in range(n)]
    prev_losses = [0 for _ in range(n)]

    for feedbackGiver in pm.participants:
        valloader = feedbackGiver.val
        bad_att = feedbackGiver.attitude == "bad"
        free_att = feedbackGiver.attitude == "freerider"
        accuracy_last_round = -1

        for ix, user in enumerate(feedbackGiver.userToEvaluate):
            if not bad_att and not free_att:
                loss, accuracy = training.test(user.model, valloader, DEVICE)
                prev_loss, prev_acc = training.test(pm.global_model, valloader, DEVICE)
                prev_acc = round(prev_acc * 100 * scalar)
                prev_loss = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

            if bad_att:
                feedback_matrix[feedbackGiver.id][user.id] = -1
                accuracy_matrix[feedbackGiver.id][user.id] = 0
                loss_matrix[feedbackGiver.id][user.id] = 65535
                prev_loss, prev_acc = training.test(pm.global_model, valloader, DEVICE)
                prev_accs[feedbackGiver.id] = round(prev_acc * 100 * scalar)
                prev_losses[feedbackGiver.id] = safe_scale(prev_loss, scalar, MAX_UINT16_SIZE)

            elif free_att:
                feedback_matrix[feedbackGiver.id][user.id] = 0
                if accuracy_last_round == -1:
                    loss_last_round, accuracy_last_round = training.test(pm.global_model, valloader, DEVICE)
                    accuracy_last_round = round(accuracy_last_round * 100 * scalar)
                    loss_last_round = safe_scale(loss_last_round, scalar, MAX_UINT16_SIZE)
                accuracy_matrix[feedbackGiver.id][user.id] = accuracy_last_round
                loss_matrix[feedbackGiver.id][user.id] = min(loss_last_round, MAX_UINT16_SIZE)
                prev_accs[feedbackGiver.id] = accuracy_last_round
                prev_losses[feedbackGiver.id] = loss_last_round

            elif user in feedbackGiver.cheater:
                feedback_matrix[feedbackGiver.id][user.id] = -1
                accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                prev_accs[feedbackGiver.id] = prev_acc
                prev_losses[feedbackGiver.id] = prev_loss

            elif accuracy > feedbackGiver.currentAcc - 0.07:  # 7% Worse
                feedback_matrix[feedbackGiver.id][user.id] = 1
                accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                prev_accs[feedbackGiver.id] = prev_acc
                prev_losses[feedbackGiver.id] = prev_loss

            elif accuracy > feedbackGiver.currentAcc - 0.14:  # 14% Worse
                feedback_matrix[feedbackGiver.id][user.id] = 0
                accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                prev_accs[feedbackGiver.id] = prev_acc
                prev_losses[feedbackGiver.id] = prev_loss

            else:
                feedback_matrix[feedbackGiver.id][user.id] = -1
                accuracy_matrix[feedbackGiver.id][user.id] = round(accuracy * 100 * scalar)
                loss_matrix[feedbackGiver.id][user.id] = safe_scale(loss, scalar, MAX_UINT16_SIZE)
                prev_accs[feedbackGiver.id] = prev_acc
                prev_losses[feedbackGiver.id] = prev_loss

            if pm.force_merge_all:
                feedback_matrix[feedbackGiver.id][user.id] = 0

        # Reset
        feedbackGiver.userToEvaluate = []
    # acc_mat = [[x / 10 for x in sublist] for sublist in accuracy_matrix]
    # loss_mat = [[x / 10 for x in sublist] for sublist in loss_matrix]
    # prev_accs_divided = [x / 10 for x in prev_accs]
    # prev_losses_divided = [x / 10 for x in prev_losses]

    # print("FEEDBACK MATRIX:")
    # print(feedback_matrix)
    # print("-----------------------------------------------------------------------------------")
    # print("ACCURACY MATRIX:")
    # print(accuracy_matrix)
    # print("-----------------------------------------------------------------------------------")
    # print("LOSS MATRIX:")
    # print(loss_matrix)
    # print("-----------------------------------------------------------------------------------")
    # print("PREVIOUS ACCURACIES:")
    # print(prev_accs)
    # print("-----------------------------------------------------------------------------------")
    # print("PREVIOUS LOSSES:")
    # print(prev_losses)
    # print("-----------------------------------------------------------------------------------")

    return feedback_matrix, accuracy_matrix, loss_matrix, prev_accs, prev_losses


def finalize_user_evaluation(pm, user): # Same as lines 294-296,306 in original code.
    loss, acc = training.test(user.model, pm.test, DEVICE) # TODO: Investigate if this should be user.val instead.
    user._accuracy.append(acc) # Line 295 in original code # TODO: Investigate if this should be test and not validation accuracy.
    user._loss.append(loss) # Line 296 in original code # TODO: Investigate if this should be test and not validation loss.
    user.hashedModel = get_hash(user.model.state_dict())


def safe_scale(value, scalar, max_val):
    if not math.isfinite(value):
        return max_val

    scaled = value * scalar

    if not math.isfinite(scaled):
        return max_val

    return min(round(scaled), max_val)
