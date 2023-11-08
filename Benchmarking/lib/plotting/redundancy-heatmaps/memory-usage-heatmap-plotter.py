import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec


def plot_memory_4sets(data1, data2, data3, data4, step_size, levels, cbar_levels, titles, suptitle, fontsize):
    fig = plt.figure(figsize=(12, 3))

    plt.subplots_adjust(left=.1, right=.95)

    width = 1.1
    gs = gridspec.GridSpec(1, 5, width_ratios=[width, .5, width, width, width])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[2])
    ax2 = plt.subplot(gs[3])
    ax3 = plt.subplot(gs[4])
    axs = [ax0, ax1, ax2, ax3]

    ax = axs[0]
    c = ax.contourf(data1, levels=levels[1], cmap='jet_r', origin="lower", norm=mcolors.TwoSlopeNorm(.55))
    ax.set_title(titles[0], fontsize=fontsize+2)
    
    
    ax.grid(False)
    cbar = plt.colorbar(c, ax=ax, ticks=cbar_levels[0], fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Redundancy", fontsize=fontsize)

    def plot_shared(data, ax, levels, title):
        ax.contour(data, levels=[1], alpha=.2, colors='k')
        c = ax.contourf(data, levels=levels, cmap='RdYlGn_r', origin="lower", norm=mcolors.TwoSlopeNorm(1))
        ax.set_title(title, fontsize=fontsize+2)
        return c

    vals = [data1, data2, data3, data4]
    tats = titles

    for i in [1, 3, 2]:
        d, title = vals[i], tats[i]
        ax = axs[i]
        c = plot_shared(d, ax, levels[1], title)

    cbar = plt.colorbar(c, ax=axs[1:], ticks=cbar_levels[1], fraction=0.046, pad=0.04, shrink=0.79)
    cbar.set_label("Memory Cost Over Dense", size=fontsize)

    i = 0

    labs = ["0", "250k", "500k", "750k", "1000k"]

    ticks = axs[0].get_xticks()
    for ax in axs:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("Nonzero Elements", fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize-2)
        
        if i == 0:
            ax.set_yticks(ticks[:-1])
            ax.set_yticklabels(labs)
            ax.set_ylabel(r"Unique Elements", fontsize=fontsize)
        elif i == 1:
            ax.set_ylabel(r"Unique Elements", fontsize=fontsize)
            ax.set_yticks(ticks[:-1])
            ax.set_yticklabels(labs)
        else:
            ax.set_yticks([])

        ax.set_xticks(ticks[:-1])
        ax.set_xticklabels(labs)
        i += 1

    axs[2].set_yticks([])
    axs[3].set_yticks([])

    plt.suptitle(suptitle)
    plt.savefig("redundancy_memory_plot.pdf", bbox_inches='tight')


size = 1_000_000
step = 1_000
ivcsc = np.load(f"ivcsc_{size}_{step}.npy")
vcsc = np.load(f"vcsc_{size}_{step}.npy")
csc = np.load(f"csc_{size}_{step}.npy")
redundancy = np.load(f"redundancy.npy")

ivcsc *= (size + 1)
ivcsc += 4
ivcsc /= size
vcsc *= (size + 1)
vcsc /= size
csc *= (size + 1)
csc /= size

plot_memory_4sets(redundancy, csc, vcsc, ivcsc,
                  step_size=step, levels=[None, 30],
                  cbar_levels=[np.linspace(0, .9, 10), None],
                  titles=['Redundancy', 'CSC Memory Use', 'VCSC Memory Use', 'IVCSC Memory Use'],
                  suptitle='',
                  fontsize=12)
