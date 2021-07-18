"""Visualise first layer filters.

This creates weblogos of filter motifs. Steps are as follows:
- get filter activation for samples with binding
- calculate threshold based on maximum activation in the dataset
- write sequence to file for locations that pass the threshold
- visualise sequences using weblogo
"""
import os

import numpy as np
from Bio import SeqIO
from keras.models import model_from_json, Model
from Bio.Seq import Seq

from utils import get_class, load_data, get_positive_samples, reset_seeds

reset_seeds()


def create_filter_motifs(protein, set_number, experiment_set, max_pct=0.5):
    """Create weblogos of filter motifs.

    If the argument `max_pct` isn't passed in, the default 0.5 is used.

    Parameters
    ----------
    protein : str
        Name of protein
    set_number : int
        Dataset number
    experiment_set : list
        List of strings with names of datasources to use.
    max_pct : float, optional
        Threshold factor for filter actvation (default is 0.5)
    """
    X_samples = get_positive_samples(protein, "test", set_number, *experiment_set)

    with open(f"results/set_{set_number}/{protein}/model.json", "r") as json_file:
        json = json_file.read()
        model = model_from_json(json)
        model.load_weights(f"results/set_{set_number}/{protein}/weights.h5")
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    if not os.path.exists("results/filter_motifs"):
        os.mkdir("results/filter_motifs")

    first_layer_outputs = model.layers[0].output
    activation_model = Model(inputs=model.input, outputs=first_layer_outputs)
    activations = activation_model.predict(X_samples)

    test_sequences = load_data(
        f"datasets/clip/{protein}/30000/test_sample_{set_number}"
    )
    y_test = get_class(protein, "test", set_number)
    y_test = y_test[:, 1]
    positive_seq = [
        seq
        for seq, label in zip(SeqIO.parse(test_sequences, "fasta"), y_test)
        if label == 1
    ]

    filters_sequences = {}
    for filter_number in range(10):
        filter_activations = activations[:, :, filter_number]

        all_activations = np.ravel(filter_activations)
        activations_mean = all_activations.mean()
        activations_norm = all_activations - activations_mean
        threshold = max_pct * activations_norm.max() + activations_mean

        for case_num, act in enumerate(filter_activations):
            locations = np.argwhere(act > threshold)
            if locations.size != 0:
                for loc in locations.flatten():
                    loc = int(loc)
                    nucleotides = list(positive_seq[case_num].seq)[loc : loc + 6]
                    motif = Seq("".join(nucleotides).replace("T", "U"))

                    if filter_number in filters_sequences:
                        filters_sequences[filter_number].append(motif)
                    else:
                        filters_sequences[filter_number] = [motif]

    for filter_number, fil_seq in filters_sequences.items():
        if not os.path.exists("results/filter_motifs/{}".format(protein)):
            os.mkdir("results/filter_motifs/{}".format(protein))
        with open(
            f"results/filter_motifs/{protein}/filter{filter_number}_motif.txt", "w"
        ) as motif_file:
            for motif in fil_seq:
                motif_file.write(str(motif) + "\n")

    for filter_number, fil_seq in filters_sequences.items():
        cmd = (
            f"weblogo -f results/filter_motifs/{protein}/filter{filter_number}_motif.txt -F png_print -o results/filter_motifs/{protein}/motif{filter_number}.png -P '' "
            f"--title 'Filter motif {filter_number} ({len(fil_seq)})' --size large --errorbars NO --show-xaxis YES --show-yaxis YES -A rna "
            f"--composition none --color '#00CC00' 'A' 'A' --color '#0000CC' 'C' 'C' --color '#FFB300' 'G' 'G' --color '#CC0000' 'U' 'U'"
        )
        os.system(cmd)


if __name__ == "__main__":
    set_number = 0
    from constants import protein_list, experiment_set

    if not os.path.exists("filter_motifs"):
        os.mkdir("filter_motifs")

    for protein in protein_list:
        create_filter_motifs(protein, set_number, experiment_set, max_pct=0.5)
