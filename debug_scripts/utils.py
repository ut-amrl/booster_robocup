import numpy as np
import matplotlib.pyplot as plt

def plot_actions(actions, actions2=None, name=None, ymin=None, ymax=None, xmax=None):
    actions = np.array(actions, dtype=float)  # shape: (n_steps1, n_actions)
    n_steps1, n_actions = actions.shape

    if actions2 is not None:
        actions2 = np.array(actions2, dtype=float)
        n_steps2 = actions2.shape[0]

    # grid layout
    n_cols = int(np.ceil(np.sqrt(n_actions)))
    n_rows = int(np.ceil(n_actions / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), sharex=True)
    axes = axes.flatten()

    for i in range(n_actions):
        time1 = np.arange(n_steps1)
        axes[i].plot(time1, actions[:, i], label="actions1")

        if actions2 is not None:
            time2 = np.arange(n_steps2)
            min_len = min(len(time1), len(time2))
            axes[i].plot(time2[:min_len], actions2[:min_len, i], label="actions2", linestyle="--")
            if len(time2) > min_len:
                axes[i].plot(time2[min_len:], actions2[min_len:, i], linestyle="--", color="orange")

        axes[i].set_title(f"Idx {i}", fontsize=10)
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel("Value")
        axes[i].grid(True)
        if actions2 is not None:
            axes[i].legend(fontsize=8)

        # apply y-limits if provided
        if ymin is not None or ymax is not None:
            axes[i].set_ylim(ymin, ymax)
        if xmax is not None:
            axes[i].set_xlim(0, xmax)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    name = name or "plot"
    fig.suptitle(name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave room for suptitle

    plt.savefig(name)
    print(f"saved to {name}")
