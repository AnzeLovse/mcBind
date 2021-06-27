import os

import numpy as np
import seqlogo
from Bio import motifs

# sys.path.append("~/miniconda3/pkgs/ghostscript-9.18-1/bin")
# print(sys.path)

if not os.path.exists("results/filters"):
    os.makedirs("results/filters")

with open("results/first_layer_filters.jaspar", "r") as handle:
    records = motifs.parse(handle, "jaspar")
    for motif in records:
        pfm = motif.format("pfm")
        # Motifs need to be converted which returns strings
        # Strings are split, whitespaces removed and converted to floats
        motif_str_list = [l.split("   ") for l in pfm.split("\n")]
        motif_matrix = np.array(
            [[float(j.strip(" ")) for j in i if j != ""] for i in motif_str_list][0:4]
        ).T
        logo_matrix = seqlogo.Pfm(motif_matrix, alphabet_type="RNA")
        seqlogo.seqlogo(
            logo_matrix,
            ic_scale=False,
            format="png",
            size="large",
            units="probability",
            show_xaxis=False,
            show_yaxis=False,
            filename="results/filters/{}.png".format(motif.name),
        )
