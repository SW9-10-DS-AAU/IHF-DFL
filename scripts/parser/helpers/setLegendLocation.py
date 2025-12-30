from enum import Enum

class LegendPosition(Enum):
    INSIDE_LOWER_LEFT = 1
    INSIDE_UPPER_RIGHT = 2
    BELOW_RIGHT = 3
    BELOW_CENTER = 4
    OUTSIDE_RIGHT = 5

def set_legend(ax, position: LegendPosition, group_names=None):
    if position == LegendPosition.INSIDE_LOWER_LEFT:
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            bbox_transform=ax.transAxes,
            frameon=True,
        )

    elif position == LegendPosition.INSIDE_UPPER_RIGHT:
        ax.legend(
            loc="upper right",
            frameon=True,
        )

    elif position == LegendPosition.BELOW_RIGHT:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.0, -0.15),
            bbox_transform=ax.transAxes,
            frameon=True,
        )

    elif position == LegendPosition.BELOW_CENTER:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            bbox_transform=ax.transAxes,
            frameon=True,
            ncol=len(group_names) if group_names else 1,
        )

    elif position == LegendPosition.OUTSIDE_RIGHT:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            bbox_transform=ax.transAxes,
            frameon=True,
        )

    else:
        raise ValueError(f"Unknown LegendPosition: {position}")