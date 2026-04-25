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
