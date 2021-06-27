from ionmf.factorization.model import iONMF
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
import pickle
import gzip
import os
import sys

np.set_printoptions(precision=5)


def load_data(path, kmer=True, rg=True, clip=True, rna=True, go=False):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if go:
        data["X_GO"] = np.loadtxt(
            gzip.open(os.path.join(path, "matrix_GeneOntology.tab.gz")), skiprows=1
        )
    if kmer:
        data["X_KMER"] = np.loadtxt(
            gzip.open(os.path.join(path, "matrix_RNAkmers.tab.gz")), skiprows=1
        )
    if rg:
        data["X_RG"] = np.loadtxt(
            gzip.open(os.path.join(path, "matrix_RegionType.tab.gz")), skiprows=1
        )
    if clip:
        data["X_CLIP"] = np.loadtxt(
            gzip.open(os.path.join(path, "matrix_Cobinding.tab.gz")), skiprows=1
        )
    if rna:
        data["X_RNA"] = np.loadtxt(
            gzip.open(os.path.join(path, "matrix_RNAfold.tab.gz")), skiprows=1
        )
    data["Y"] = np.loadtxt(
        gzip.open(os.path.join(path, "matrix_Response.tab.gz")), skiprows=1
    )
    data["Y"] = data["Y"].reshape((len(data["Y"]), 1))

    return data


def load_labels(path, kmer=True, rg=True, clip=True, rna=True, go=True):
    """
        Load column labels for data matrices.
    """

    labels = dict()
    if go:
        labels["X_GO"] = (
            gzip.open(os.path.join(path, "matrix_GeneOntology.tab.gz"))
            .readline()
            .split("\t")
        )
    if kmer:
        labels["X_KMER"] = (
            gzip.open(os.path.join(path, "matrix_RNAkmers.tab.gz"))
            .readline()
            .split("\t")
        )
    if rg:
        labels["X_RG"] = (
            gzip.open(os.path.join(path, "matrix_RegionType.tab.gz"))
            .readline()
            .split("\t")
        )
    if clip:
        labels["X_CLIP"] = (
            gzip.open(os.path.join(path, "matrix_Cobinding.tab.gz"))
            .readline()
            .split("\t")
        )
    if rna:
        labels["X_RNA"] = (
            gzip.open(os.path.join(path, "matrix_RNAfold.tab.gz"))
            .readline()
            .split("\t")
        )
    return labels


def run(protein):

    # Select example protein folder from the dataset
    protein = protein

    # Load training data and column labels
    training_data = load_data(
        "datasets/clip/%s/30000/training_sample_0" % protein, go=False, kmer=False
    )
    training_labels = load_labels(
        "datasets/clip/%s/30000/training_sample_0" % protein, go=False, kmer=False
    )
    model = iONMF(rank=5, max_iter=100, alpha=10.0)

    # Fit all training data
    model.fit(training_data)

    # Make predictions about class on all training data
    # delete class from dictionary
    test_data = load_data(
        "datasets/clip/%s/30000/test_sample_0" % protein, go=False, kmer=False
    )
    true_y = test_data["Y"].copy()
    del test_data["Y"]
    results = model.predict(test_data)

    # Evaluate prediction on holdout test set
    predictions = results["Y"]
    auc = roc_auc_score(true_y, predictions)
    print "Test AUC: ", auc

    return auc, predictions


if __name__ == "__main__":
    protein_list = [
        "1_PARCLIP_AGO1234_hg19",
        "2_PARCLIP_AGO2MNASE_hg19",
        "3_HITSCLIP_Ago2_binding_clusters",
        "4_HITSCLIP_Ago2_binding_clusters_2",
        "5_CLIPSEQ_AGO2_hg19",
        "6_CLIP-seq-eIF4AIII_1",
        "7_CLIP-seq-eIF4AIII_2",
        "8_PARCLIP_ELAVL1_hg19",
        "9_PARCLIP_ELAVL1MNASE_hg19",
        "10_PARCLIP_ELAVL1A_hg19",
        "11_CLIPSEQ_ELAVL1_hg19",
        "12_PARCLIP_EWSR1_hg19",
        "13_PARCLIP_FUS_hg19",
        "14_PARCLIP_FUS_mut_hg19",
        "15_PARCLIP_IGF2BP123_hg19",
        "16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters",
        "17_ICLIP_HNRNPC_hg19",
        "18_ICLIP_hnRNPL_Hela_group_3975_all-hnRNPL-Hela-hg19_sum_G_hg19--ensembl59_from_2337-2339-741_bedGraph-cDNA-hits-in-genome",
        "19_ICLIP_hnRNPL_U266_group_3986_all-hnRNPL-U266-hg19_sum_G_hg19--ensembl59_from_2485_bedGraph-cDNA-hits-in-genome",
        "20_ICLIP_hnRNPlike_U266_group_4000_all-hnRNPLlike-U266-hg19_sum_G_hg19--ensembl59_from_2342-2486_bedGraph-cDNA-hits-in-genome",
        "21_PARCLIP_MOV10_Sievers_hg19",
        "22_ICLIP_NSUN2_293_group_4007_all-NSUN2-293-hg19_sum_G_hg19--ensembl59_from_3137-3202_bedGraph-cDNA-hits-in-genome",
        "23_PARCLIP_PUM2_hg19",
        "24_PARCLIP_QKI_hg19",
        "25_CLIPSEQ_SFRS1_hg19",
        "26_PARCLIP_TAF15_hg19",
        "27_ICLIP_TDP43_hg19",
        "28_ICLIP_TIA1_hg19",
        "29_ICLIP_TIAL1_hg19",
        "30_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters",
        "31_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters",
    ]
    scores = {}
    for dataset in range(3):
        for protein in protein_list:
            score, predictions = run(protein)
            with open(
                "results/set_%s/%s_predictions" % (dataset, protein), "wb"
            ) as predictions_file:
                np.save(predictions_file, predictions)
            if protein in scores:
                scores[protein].append(score)
            else:
                scores[protein] = [score]

    pd.DataFrame.from_dict(scores, orient="index").to_csv(
        "results/ionmf_scores.tsv", sep="\t"
    )
