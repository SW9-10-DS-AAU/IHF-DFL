import torch
import random
import ml.training as training
import ml.evaluation as evaluation
from ml.visualization import get_color
from utils.colors import green, red, yellow, rb
from ml.runtime import DEVICE
from collections import OrderedDict


def manipulate(model, scale: float = 1.0) -> OrderedDict:
    sd = OrderedDict()
    with torch.no_grad():
        for k, v in model.state_dict().items():
            t = v.clone()
            if t.is_floating_point():
                # uniform noise in [-scale, scale]
                noise = torch.empty_like(t).uniform_(-scale, scale)
                t.add_(noise)
            sd[k] = t
    return sd


def add_noise(model, offset_from_end: int = 5) -> OrderedDict:
    """
    GPU-friendly: keep tensors on their original device/dtype and add a tiny scalar
    to the tensor at index len(state_dict)-offset_from_end.
    """
    items = list(model.state_dict().items())
    target_idx = max(0, len(items) - offset_from_end)

    new_sd = OrderedDict()
    with torch.no_grad():
        for idx, (k, v) in enumerate(items):
            t = v.clone()
            if t.is_floating_point() and idx == target_idx:
                # Match original magnitude: 9e-6 or 1e-5
                eps = 1e-5 if random.randint(9, 10) == 10 else 9e-6
                t.add_(eps)  # in-place scalar add on the same device (CPU/GPU)
            new_sd[k] = t
    return new_sd


def let_malicious_users_do_their_work(pm):
    for i in range(len(pm.participants)):
        if pm.participants[i].attitude == "bad":

            if pm.malicious_attack_type == "byzantine":
                print(red("Address {} executing Byzantine Attack".format(pm.participants[i].address[0:16] + "...")))
                manipulated_state_dict = byzantine_attack(pm, pm.participants[i])
                fallback = pm.previous_global_model is None
                pm.participants[i].last_attack_type = "noise" if fallback else "byzantine"
            else:
                print(red("Address {} going to provide random weights".format(
                pm.participants[i].address[0:16] + "...")))
                manipulated_state_dict = manipulate(pm.participants[i].model, scale=pm.malicious_noise_scale)
                pm.participants[i].last_attack_type = "noise"

            pm.participants[i].model.load_state_dict(manipulated_state_dict)
            pm.participants[i].hashedModel = evaluation.get_hash(pm.participants[i].model.state_dict())
            loss, accuracy = training.test(pm.participants[i].model, pm.test, DEVICE)

            pm.participants[i].currentAcc = accuracy
            pm.participants[i].currentLoss = loss

            print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                  pm.participants[i].address[
                                                                                      0:16] + "...",
                                                                                  accuracy * 100, loss))


def let_freerider_users_do_their_work(pm):
    for i in range(len(pm.participants)):
        if pm.participants[i].attitude == "freerider":

            if pm.freerider_attack_type == "delta_weight":
                print(
                    red("Address {} executing Delta Weights Attack".format(pm.participants[i].address[0:16] + "...")))
                manipulated_state_dict = delta_weight_attack(pm, pm.participants[i])
                fallback = pm.previous_global_model is None
                pm.participants[i].last_attack_type = "noise" if fallback else "delta_weight"
            else:
                print(red("Address {} going to provide random weights".format(
                    pm.participants[i].address[0:16] + "...")))
                manipulated_state_dict = manipulate(pm.participants[i].model, scale=pm.freerider_noise_scale)
                pm.participants[i].last_attack_type = "noise"

            pm.participants[i].model.load_state_dict(manipulated_state_dict)
            pm.participants[i].hashedModel = evaluation.get_hash(pm.participants[i].model.state_dict())
            loss, accuracy = training.test(pm.participants[i].model, pm.test, DEVICE)

            pm.participants[i].currentAcc = accuracy
            pm.participants[i].currentLoss = loss

            print("{:<17} {} |  Testing  | Accuracy {:>3.0f} % | Loss ∞\n".format("Account testing:   ",
                                                                                  pm.participants[i].address[
                                                                                      0:16] + "...",
                                                                                  accuracy * 100, loss))


def update_users_attitude(pm):
    for user in pm.participants:
        if user.attitudeSwitch == pm.round and user.attitude != user.futureAttitude:
            print(rb("Address {} going to switch attitude to {}".format(user.address[0:16] + "...",
                                                                        user.futureAttitude)))
            user.attitude = user.futureAttitude
            user.color = get_color(None, user.attitude)


def _freerider_submit_with_noise(pm, user):
    """Freerider reuses the global model with configurable noise."""

    if pm.freerider_noise_scale < 0:
        raise ValueError("freerider_noise_scale must be non-negative")

    if pm.freerider_noise_scale == 0:  # Copy global model if noise is zero
        print(yellow("Address {} resubmitting original model".format(user.address[0:16] + "...")))
        # Changed from deepcopy(user.model).state_dict(): only tensor copies are required for the submission payload.
        return OrderedDict((k, v.clone()) for k, v in user.model.state_dict().items())

    print(red(
        "Address {} adding noise (scale={}) to global weights".format(
            user.address[0:16] + "...",
            pm.freerider_noise_scale,
        )
    ))
    # Changed from manipulate(deepcopy(user.model), ...): manipulate() already clones each tensor internally.
    return manipulate(user.model, scale=pm.freerider_noise_scale)


def byzantine_attack(pm, user):
    """Byzantine Attack.

    Computes the global model's recent trajectory (delta between the current
    and previous global checkpoint) and crafts an update that pushes the
    global model in the opposite direction of learning, actively harming training.

    Falls back to noise in round 1 when there is not yet enough history to
    estimate the trajectory.
    """
    if pm.previous_global_model is None:
        print(red("  [Byzantine] Not enough history yet – falling back to noise attack"))
        # Same deepcopy removal as above: fallback path only needs a crafted tensor dict, not an nn.Module copy.
        return manipulate(user.model, scale=pm.malicious_noise_scale)

    crafted_weights = OrderedDict()
    with torch.no_grad():
        # Changed from pm.previous_global_model.state_dict(): previous_global_model is now already a state_dict snapshot.
        prev_weights = pm.previous_global_model
        current_weights = pm.global_model.state_dict()

        for key in current_weights.keys():
            value = current_weights[key]
            if value.is_floating_point():
                # Trajectory: direction the global model has been moving
                learning_direction = current_weights[key] - prev_weights[key]
                # Push against the trajectory to reverse learning
                crafted_weights[key] = value - pm.malicious_noise_scale * learning_direction
            else:
                crafted_weights[key] = value.clone()

    return crafted_weights


def delta_weight_attack(pm, user):
    """Delta Weights Attack

    A free-rider constructs fake gradient updates by subtracting two
    consecutively received global models, and creates a small deviation from the global model,
    which should be hard to detect.
    Falls back to noise in round 1 when no previous global model exists.
    """
    if pm.previous_global_model is None:
        print(red("  [DeltaWeight] Not enough history yet – falling back to noise attack"))
        # Same deepcopy removal as above: fallback path only needs a crafted tensor dict, not an nn.Module copy.
        return manipulate(user.model, scale=pm.freerider_noise_scale)

    crafted_weights = OrderedDict()
    with torch.no_grad():
        # Changed from pm.previous_global_model.state_dict(): previous_global_model is now already a state_dict snapshot.
        prev_weights = pm.previous_global_model
        current_weights = pm.global_model.state_dict()

        for key in current_weights.keys():
            value = current_weights[key]
            if value.is_floating_point():
                fake_gradient = prev_weights[key] - value  # G_fake = M_{j-1} - M_j
                crafted_weights[key] = value + fake_gradient  # = M_{j-1}
            else:
                crafted_weights[key] = value.clone()

    return crafted_weights
