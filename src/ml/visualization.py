import os
import numpy as np
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
bad_c = "#d62728"
free_c = "#9467bd"
colors.remove(bad_c)
colors.remove(free_c)


def get_color(i, a):
    if a == "bad":
        return bad_c
    if a == "freerider":
        return free_c
    try:
        return colors[i]
    except:
        return None


def visualize_simulation(challenge, output_folder_path): # pragma: no cover
    os.makedirs(output_folder_path, exist_ok=True)
    accuracy = [0] + challenge.pytorch_model.accuracy
    loss = [challenge.pytorch_model.loss[0]] + challenge.pytorch_model.loss

    f, axs = plt.subplots(1, 4, figsize=(16, 3), gridspec_kw={'width_ratios': [0.8, 2, 2, 2],
                                                              'height_ratios': [1]})
    plot_colors = ["#00629b", "#629b00", "#000000", "#d93e6a"]

    participants = challenge.pytorch_model.participants + challenge.pytorch_model.disqualified

    use_step_grs = False

    rounds = list(range(len(accuracy)))

    y = accuracy
    acc_line = axs[1].plot(rounds, y, color=plot_colors[0], linewidth=2.5, label="Avg. Accuracy")[0]
    twin = axs[1].twinx()
    y = loss
    loss_line = twin.plot(rounds, y, color=plot_colors[1], linewidth=2.5, linestyle="--", label="Avg. Loss")[0]

    grs_rounds = list(range(len(participants[0]._globalrep)))
    if use_step_grs:
        grs_x = [item for sublist in zip(grs_rounds, (np.array(grs_rounds) + 1).tolist()) for item in sublist]
        grs_ticks = grs_rounds
        for i, user in enumerate(participants):
            grs_y = [item for sublist in zip(user._globalrep, user._globalrep) for item in sublist]
            axs[2].plot(grs_x, grs_y, linewidth=2.5, color=user.color)
    else:
        grs_x = grs_rounds
        grs_ticks = grs_rounds
        for i, user in enumerate(participants):
            axs[2].plot(
                grs_x,
                user._globalrep,
                linewidth=2.5,
                color=user.color,
                alpha=0.9,
                marker="o",
                markersize=4,
                markevery=1,
            )

    pun = {}
    for i, j, y in challenge._punishments:
        if i in pun.keys():
            pun[i] += j
        else:
            pun[i] = j

    rew = list()
    for i, j in enumerate(challenge._reward_balance):
        if i in pun.keys():
            rew.append(j + pun[i])
        else:
            rew.append(j)

    y_reward = [item for sublist in zip(challenge._reward_balance, challenge._reward_balance) for item in sublist]
    y2_reward = [item for sublist in zip(rew, rew) for item in sublist]
    x_reward = list(range(len(challenge._reward_balance)))
    x_reward = [item for sublist in zip(x_reward, (np.array(x_reward) + 1).tolist()) for item in sublist]

    axs[3].plot(x_reward, y_reward, label="reward", color=plot_colors[0], linewidth=2.5)
    axs[3].plot(x_reward, y2_reward, label="reward +\npunishments", color=plot_colors[1], linewidth=2.5)
    axs[3].fill_between(x_reward, y_reward, y2_reward, alpha=0.2, hatch=r"//", color=plot_colors[1])

    axs[0].text(0, 0.1, f'dataset={challenge.pytorch_model.DATASET}\n'
                + f'epochs={challenge.pytorch_model.EPOCHS}\n'
                + f'rounds={challenge.pytorch_model.round - 1}\n'
                + f'users={challenge.pytorch_model.NUMBER_OF_CONTRIBUTORS}\n'
                + f'malicious={challenge.pytorch_model.NUMBER_OF_BAD_CONTRIBUTORS}\n'
                + f'copycat={challenge.pytorch_model.NUMBER_OF_FREERIDER_CONTRIBUTORS}', fontsize=15)
    axs[0].set_axis_off()

    axs[1].set_xlabel('rounds\n(a)', fontsize=14)
    axs[2].set_xlabel('rounds\n(b)', fontsize=14)
    axs[3].set_xlabel('rounds\n(c)', fontsize=14)
    axs[1].set_ylabel('Avg. Accuracy', fontsize=14)
    twin.set_ylabel('Avg. Loss', fontsize=14)
    axs[1].tick_params(axis='both', which='major', labelsize=14)

    axs[2].set_ylabel('GRS', fontsize=14)
    axs[3].set_ylabel('Contract Balance', fontsize=14)

    axs[2].tick_params(axis='both', which='major', labelsize=14)
    axs[3].tick_params(axis='both', which='major', labelsize=14)

    if len(rounds) > 20:
        axs[1].set_xticks([i for i in rounds if i % 5 == 0 or i == 0])
    else:
        axs[1].set_xticks([i for i in rounds])

    if len(grs_ticks) > 20:
        axs[2].set_xticks([i for i in grs_ticks if i % 5 == 0 or i == 0])
    else:
        axs[2].set_xticks([i for i in grs_ticks])

    if len(x_reward) > 20:
        axs[3].set_xticks([i for i in x_reward if i % 5 == 0 or i == 0])
    else:
        axs[3].set_xticks([i for i in x_reward])

    axs[1].set_xlim(0, max(rounds) if rounds else 0)

    axs[2].yaxis.get_offset_text().set_fontsize(14)
    axs[3].yaxis.get_offset_text().set_fontsize(14)

    axs[1].grid()
    axs[2].grid()
    axs[3].grid()

    twin_lines = [acc_line, loss_line]
    axs[1].legend(twin_lines, [l.get_label() for l in twin_lines], loc="lower right", fontsize=10)

    axs[3].legend(fontsize=10)

    print(output_folder_path)
    plt.tight_layout(pad=1)
    plt.savefig(os.path.join(output_folder_path, f"{challenge.pytorch_model.DATASET}_simulation.pdf"), bbox_inches='tight')
    return plt
