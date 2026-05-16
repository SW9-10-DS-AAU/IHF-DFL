import torch
from collections import OrderedDict
import ml.training as training
from utils.colors import yellow, red, b
from utils.printer import print_divider
from ml.runtime import DEVICE


def the_merge(pm, _current_round_no, _users, warning_collector=None):

    # No qualified users → skip merge this round
    if not _users:
        msg = f"[Round {_current_round_no}] No participants qualified for merge – skipping aggregation"
        print_divider()
        print(red(msg))
        print_divider(blank_line_after=True)
        if warning_collector is not None:
            warning_collector.append(msg)
        return

    client_models, users_contribution_scores = [], {}

    for u in _users:
        client_models.append(u.model)
        print("Account {} participating in merge".format(u.address[0:16] + "..."))
        # users_contribution_scores[u.address] = u.contribution_score

    n_clients = len(client_models)

    client_state_dicts = [m.state_dict() for m in client_models]

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

            # unweighted averaging
            global_dict[k] = stacked.mean(0)

        pm.global_model.load_state_dict(global_dict)

    # -------------------------
    # Evaluation
    # -------------------------
    loss, accuracy = training.test(pm.global_model, pm.test, DEVICE)
    pm.accuracy.append(accuracy)
    pm.loss.append(loss)

    print_divider()
    print(b("Merged Model: Accuracy {:>3.0f} % | Loss {:>6,.2f}".format(accuracy * 100, loss)))

    # -------------------------
    # Distribute global model
    # -------------------------
    for u in pm.participants:
        # Changed from deepcopy(u.model): keep only the pre-merge tensor state needed by scoring/attacks.
        # This prevents accidental aliasing while avoiding the overhead of cloning full nn.Module objects.
        u.previousModel = OrderedDict((k, v.detach().clone()) for k, v in u.model.state_dict().items())
        u.model.load_state_dict(pm.global_model.state_dict())

    print_divider(blank_line_after=True)