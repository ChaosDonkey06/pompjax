
import matplotlib.pyplot as plt

def convergence_plot(p_mean, posterior_list_df, p_range, param_label=None, param_truth=None, title=None, path_to_save = None, ax=None, fig=None):
    p, Nif  = p_mean.shape

    if param_label is None:
        param_label = [f"param{i}" for i in range(1, p+1)]

    if ax is None:
        fig, ax = plt.subplots(p, 1, figsize=(15.5, 12.2), sharex=True)

    for idx, axi in enumerate(ax.flatten()):
        param_range = p_range[idx, :]
        p_lab       = param_label[idx]
        param_df    = posterior_list_df[idx]

        axi.plot(range(Nif), p_mean[idx,:], color="k", lw=3, label="Mean")
        axi.fill_between(param_df.index.values+1, param_df["low_95"], param_df["high_95"], color="gray", alpha=0.2, label="95% CI")
        axi.fill_between(param_df.index.values+1, param_df["low_50"], param_df["high_50"], color="gray", alpha=0.4, label="50% CI")

        if param_truth:
            axi.axhline(y=param_truth[idx], color="red", linestyle="--", lw=2, label="Truth")

        axi.set_ylabel(p_lab)
        axi.legend(loc="upper right", ncol=1)
        axi.set_ylim(param_range)

    ax[-1].set_xlabel("IF iteration")
    fig.suptitle(title)

    plt.tight_layout()
    if path_to_save:
        fig.savefig(path_to_save, dpi=300, transparent=True)