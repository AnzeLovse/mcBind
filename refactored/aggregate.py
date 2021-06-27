"""Aggregate all plots for supplementary."""
import os
import pickle
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from constants import protein_list, protein_names

style.use("seaborn-notebook")


def plot_aggregated(protein, protein_name, order, export_pdf):
    """Aggregate plots."""

    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    gs = fig.add_gridspec(8, 10)

    fig.suptitle(f"Experiment: {protein_name}", fontsize=16)

    ax = fig.add_subplot(gs[:5, 5:])
    img = mpimg.imread(f"results/calibration_curves/{protein}_calibration_curve.png")
    ax.imshow(img)
    ax.set_title(f"b", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[:4, :5])
    img = mpimg.imread(f"results/ROC_all/{protein}.png")
    img = img[89:, 0:680, :]
    ax.imshow(img)
    ax.set_title(f"a", loc="left", fontweight="bold")

    ax = fig.add_subplot(gs[4, 0])
    img = mpimg.imread(f"results/set_{SET_NUMBER}/{protein}/saliency.png")
    ax.set_title(f"c", loc="left", fontweight="bold")
    ax.imshow(img)

    ax = fig.add_subplot(gs[-1, :])
    img = mpimg.imread(f"results/set_{SET_NUMBER}/{protein}/max_activation.png")
    ax.imshow(img)
    ax.set_title(f"f", loc="left", fontweight="bold")

    for grid_num, filter in enumerate(order):
        activation_fname = f"results/filter_motifs/{protein}/motif{filter}.png"
        if Path(activation_fname).is_file():
            ax = fig.add_subplot(gs[-3, grid_num])
            img = mpimg.imread(activation_fname)
            ax.imshow(img)
        if grid_num == 0:
            ax.set_title(f"d", loc="left", fontweight="bold")

        ax2 = fig.add_subplot(gs[-2, grid_num])
        img2 = mpimg.imread(f"results/filters/SEQ_{protein}_{filter}.png")
        ax2.imshow(img2)

        if grid_num == 0:
            ax2.set_title(f"e", loc="left", fontweight="bold")

    for ax in fig.get_axes():
        ax.set_axis_off()
    export_pdf.savefig()

    if not os.path.exists("results/aggregated"):
        os.makedirs("results/aggregated")
    plt.savefig(f"results/aggregated/{protein}")
    plt.close()


SET_NUMBER = 0
with open("results/filter_order.pkl", "rb") as pkl:
    filter_order = pickle.load(pkl)

name_mapping = {protein: name for (protein, name) in zip(protein_list, protein_names)}

with PdfPages("results/aggregated_plots.pdf") as export_pdf:
    for protein, order in filter_order.items():
        plot_aggregated(protein, name_mapping[protein], order, export_pdf)
