"""Utility functions."""
import gzip
import os
import random

import numpy as np
from Bio import SeqIO
from keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
import tensorflow as tf


def reset_seeds():
    np.random.seed(4)
    random.seed(5)
    tf.set_random_seed(4)


reset_seeds()


def load_data(path):
    data = gzip.open(os.path.join(path, "sequences.fa.gz"), "rt")
    return data


def get_seq(protein, t_data, training_set_number):
    if t_data == "train":
        training_data = load_data(
            "datasets/clip/%s/30000/training_sample_%s" % (protein, training_set_number)
        )
        x_train = np.zeros((30000, 101, 4))

    elif t_data == "test":
        training_data = load_data(
            "datasets/clip/%s/30000/test_sample_%s" % (protein, training_set_number)
        )
        x_train = np.zeros((10000, 101, 4))
    r = 0
    for record in SeqIO.parse(training_data, "fasta"):
        sequence = list(record.seq)
        nucleotide = {"A": 0, "T": 1, "G": 2, "C": 3, "N": 4}
        num_seq = list()  # Encoded sequence
        for i in range(0, len(sequence)):
            num_seq.append(nucleotide[sequence[i]])

        X = np.zeros((1, len(num_seq), 4))
        for i in range(len(num_seq)):
            if num_seq[i] <= 3:
                X[:, i, num_seq[i]] = 1
        x_train[r, :, :] = X
        r = r + 1

    return x_train


def get_class(protein, t_data, training_set_number):
    y_train = []
    if t_data == "train":
        data = load_data(
            "datasets/clip/%s/30000/training_sample_%s" % (protein, training_set_number)
        )
    elif t_data == "test":
        data = load_data(
            "datasets/clip/%s/30000/test_sample_%s" % (protein, training_set_number)
        )
    for record in SeqIO.parse(data, "fasta"):
        v = int((record.description).split(":")[1])
        # [1,0] if there was no observed binding and [0,1] for sequences where binding was observed.
        y_train.append([int(v == 0), int(v != 0)])

    y_train = np.array(y_train)
    return y_train


def get_cobinding(protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(
            (
                "datasets/clip/%s/30000/training_sample_%s/matrix_Cobinding.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            cobinding_data = np.loadtxt(f, skiprows=1)
        cobinding = np.zeros(
            (30000, 101, int(cobinding_data.shape[1] / 101)), dtype=np.int
        )
    elif t_data == "test":
        with gzip.open(
            (
                "datasets/clip/%s/30000/test_sample_%s/matrix_Cobinding.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            cobinding_data = np.loadtxt(f, skiprows=1)
        cobinding = np.zeros(
            (10000, 101, int(cobinding_data.shape[1] / 101)), dtype=np.int
        )
    for n in range(0, cobinding_data.shape[1], 101):
        a = cobinding_data[:, n : (n + 101)]
        cobinding[:, :, int(n / 101)] = a

    return cobinding


def get_region(protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(
            (
                "datasets/clip/%s/30000/training_sample_%s/matrix_RegionType.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            region_data = np.loadtxt(f, skiprows=1)
        region = np.zeros((30000, 101, int(region_data.shape[1] / 101)), dtype=np.int)

    elif t_data == "test":
        with gzip.open(
            (
                "datasets/clip/%s/30000/test_sample_%s/matrix_RegionType.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            region_data = np.loadtxt(f, skiprows=1)
        region = np.zeros((10000, 101, int(region_data.shape[1] / 101)), dtype=np.int)
    for n in range(0, region_data.shape[1], 101):
        a = region_data[:, n : (n + 101)]
        region[:, :, int(n / 101)] = a

    return region


def get_fold(protein, t_data, training_set_number):
    if t_data == "train":
        with gzip.open(
            (
                "datasets/clip/%s/30000/training_sample_%s/matrix_RNAfold.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            fold_data = np.loadtxt(f, skiprows=1)
        fold = np.zeros((30000, 101, int(fold_data.shape[1] / 101)), dtype=np.float64)

    elif t_data == "test":
        with gzip.open(
            (
                "datasets/clip/%s/30000/test_sample_%s/matrix_RNAfold.tab.gz"
                % (protein, training_set_number)
            ),
            "rt",
        ) as f:
            fold_data = np.loadtxt(f, skiprows=1)
        fold = np.zeros((10000, 101, int(fold_data.shape[1] / 101)), dtype=np.float64)

    for n in range(0, fold_data.shape[1], 101):
        a = fold_data[:, n : (n + 101)]
        fold[:, :, int(n / 101)] = a

    return fold


def load_data_sources(protein, t_data, training_set_number, *args):
    X = np.array([])
    data_sources = []
    for arg in args:

        if arg == "SEQ":
            if X.size == 0:
                X = get_seq(protein, t_data, training_set_number)
            else:
                X = np.dstack((X, get_seq(protein, t_data, training_set_number)))
        if arg == "RNA":
            if X.size == 0:
                X = get_fold(protein, t_data, training_set_number)
            else:
                X = np.dstack((X, get_fold(protein, t_data, training_set_number)))
        if arg == "RG":
            if X.size == 0:
                X = get_region(protein, t_data, training_set_number)
            else:
                X = np.dstack((X, get_region(protein, t_data, training_set_number)))
        if arg == "CLIP":
            if X.size == 0:
                X = get_cobinding(protein, t_data, training_set_number)
            else:
                X = np.dstack((X, get_cobinding(protein, t_data, training_set_number)))
        data_sources.append(arg)

    data_sources = ",".join(data_sources)
    return data_sources, X


def get_model(size):
    model = Sequential()
    model.add(
        Conv1D(
            10,
            6,
            data_format="channels_last",
            input_shape=(101, size),
            strides=1,
            padding="valid",
        )
    )
    model.add(MaxPooling1D(pool_size=20, strides=1, padding="valid"))
    model.add(Conv1D(10, 4, activation="relu"))
    model.add(Dropout(0.1, seed=4))
    model.add(MaxPooling1D(pool_size=40, strides=1, padding="valid"))
    model.add(Conv1D(15, 4, activation="relu"))
    model.add(MaxPooling1D(pool_size=30, strides=1, padding="valid"))
    model.add(Conv1D(15, 3, activation="relu"))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.1, seed=5))
    model.add(Dense(2, activation="softmax"))
    return model


def get_positive_samples(protein, t_data, training_set_number, *args):
    """Get positive samples."""
    _, X_test = load_data_sources(protein, t_data, training_set_number, *args)
    y_test = get_class(protein, t_data, training_set_number)
    y_test = y_test[:, 1]
    positive_samples = np.zeros((2000, X_test.shape[1], X_test.shape[2]))
    n = 0

    for i, value in enumerate(y_test):
        if value == 1:
            positive_samples[n] = X_test[i]
            n += 1

    return positive_samples
