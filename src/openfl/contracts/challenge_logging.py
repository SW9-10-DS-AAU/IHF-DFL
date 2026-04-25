"""
Bridges FLChallenge state to the ExperimentLogger interface: adapts shapes
for simple events (receipt dicts, User objects -> flat kwargs), orchestrates
multi-call sequences for round events (per-user logging, vote-matrix
iteration, addr->reward derivation) that would otherwise leak into
simulate().

The trivial functions stay for consistency and to keep the ExperimentLogger
interface decoupled from FLChallenge call sites.
"""


def log_receipt(challenge, receipt, receipt_type, round=None):
    if challenge._logger is None:
        return
    challenge._logger.receipt(
        round=challenge.pytorch_model.round if round is None else round,
        tx_type=receipt_type,
        tx_hash=receipt["transactionHash"].hex(),
        gas_used=receipt["gasUsed"],
    )


def log_warning(challenge, msg, round=None):
    if challenge._logger is None:
        return
    r = round if round is not None else challenge.pytorch_model.round
    challenge._logger.warning(r, msg)


def log_contribution_scores(challenge, users, scores, raw_values, outlier_info, previous_avg):
    if challenge._logger is None:
        return
    challenge._logger.contribution_scores(
        round=challenge.pytorch_model.round,
        user_ids=[u.id for u in users],
        user_addresses=[u.address for u in users],
        scores=scores,
        raw_values=raw_values,
        outlier_info=outlier_info,
        previous_avg=previous_avg,
    )


def log_round_zero(challenge):
    if challenge._logger is None:
        return
    challenge._logger.global_round(
        round=0,
        round_time=0.0,
        obj_global_acc=challenge.pytorch_model.accuracy[-1] if challenge.pytorch_model.accuracy else None,
        obj_global_loss=challenge.pytorch_model.loss[-1] if challenge.pytorch_model.loss else None,
        reward_pool=challenge._reward_balance[-1],
        punishment_pool=0,
    )
    all_users = challenge.pytorch_model.participants + challenge.pytorch_model.disqualified
    for _user in all_users:
        challenge._logger.user_round(
            round=0,
            user_id=_user.id,
            state="active",
            behavior=_user.attitude,
            role=_user.futureAttitude,
            grs=_user._globalrep[-1],
            sub_personal_acc=None,
            sub_personal_loss=None,
            sub_global_acc=None,
            sub_global_loss=None,
            round_reputation_assigned=None,
            reward_delta=None,
            is_reward=None,
            merged=None,
            merge_weight=None
        )


def log_global_round(challenge, round, round_time, punishment_pool, agg_switch_collector=None):
    if challenge._logger is None:
        return
    challenge._logger.global_round(
        round=round,
        round_time=round_time,
        obj_global_acc=challenge.pytorch_model.accuracy[-1] if challenge.pytorch_model.accuracy else 0,
        obj_global_loss=challenge.pytorch_model.loss[-1] if challenge.pytorch_model.loss else 0,
        reward_pool=challenge._reward_balance[-1],
        punishment_pool=punishment_pool,
        agg_func_1=agg_switch_collector.get("func_1") if agg_switch_collector else None,
        agg_weight_1=agg_switch_collector.get("weight_1") if agg_switch_collector else None,
        agg_func_2=agg_switch_collector.get("func_2") if agg_switch_collector else None,
        agg_weight_2=agg_switch_collector.get("weight_2") if agg_switch_collector else None,
    )


def log_round(challenge, current_round, round_time,
               accuracy_matrix, loss_matrix, prev_accs, prev_losses,
               contributors, receipt, users_weight_collector, agg_switch_collector=None):
    if challenge._logger is None:
        return

    # ---- votes ----
    fbm = challenge.feedback_matrix
    for _idx, _giver in enumerate(challenge.pytorch_model.participants):
        _user_acc = prev_accs[_idx] if prev_accs and _idx < len(prev_accs) else None
        _user_loss = prev_losses[_idx] if prev_losses and _idx < len(prev_losses) else None
        for _receiver in challenge.pytorch_model.participants:
            if _giver.id == _receiver.id:
                continue
            try:
                _feedback_vote = int(fbm[_giver.id][_receiver.id])
            except (IndexError, TypeError):
                continue
            challenge._logger.vote(
                round=current_round,
                giver_id=_giver.id,
                receiver_id=_receiver.id,
                giver_address=_giver.address,
                receiver_address=_receiver.address,
                vote_feedback_score=_feedback_vote,
                vote_prev_accuracy=_user_acc,
                vote_prev_loss=_user_loss,
                vote_accuracy=accuracy_matrix[_giver.id][_receiver.id] if accuracy_matrix is not None else None,
                vote_loss=loss_matrix[_giver.id][_receiver.id] if loss_matrix is not None else None,
            )

    # ---- per-user round ----
    _round_rewards = challenge.get_round_rewards(receipt) if receipt is not None else []
    _addr_to_reward = {addr: win for addr, _rs, win, _nr, _ir in _round_rewards}
    _addr_to_ir = {addr: _ir for addr, _rs, win, _nr, _ir in _round_rewards}

    for _user in challenge.pytorch_model.participants:
        challenge._logger.user_round(
            round=current_round, user_id=_user.id, state="active",
            behavior=_user.attitude, role=_user.futureAttitude,
            grs=_user._globalrep[-1],
            sub_personal_acc=_user.currentAcc,
            sub_personal_loss=_user.currentLoss,
            sub_global_acc=_user._accuracy[-1],
            sub_global_loss=_user._loss[-1],
            round_reputation_assigned=_user._roundrep[-1] if _user._roundrep else None,
            reward_delta=_addr_to_reward.get(_user.address, None),
            is_reward=_addr_to_ir.get(_user.address, None),
            merged=any(u.id == _user.id for u in contributors),
            merge_weight=users_weight_collector.get(_user.address, None),
            attack_type=_user.last_attack_type,
        )
    for _user in challenge.pytorch_model.disqualified:
        challenge._logger.user_round(
            round=current_round, user_id=_user.id, state="disqualified",
            behavior=_user.attitude, role=_user.futureAttitude,
            grs=_user._globalrep[-1],
            sub_personal_acc=_user.currentAcc,
            sub_personal_loss=_user.currentLoss,
            sub_global_acc=_user._accuracy[-1],
            sub_global_loss=_user._loss[-1],
            round_reputation_assigned=_user._roundrep[-1] if _user._roundrep else None,
            reward_delta=_addr_to_reward.get(_user.address, None),
            is_reward=_addr_to_ir.get(_user.address, None),
            merged=False,
            merge_weight=None,
            attack_type=_user.last_attack_type,
        )

    # ---- global round ----
    _punishment_total = sum(p[1] for p in challenge._punishments if p[0] == current_round)
    log_global_round(challenge, current_round, round_time, _punishment_total, agg_switch_collector)