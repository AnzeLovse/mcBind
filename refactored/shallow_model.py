# Before training add "env": {"PYTHONHASHSEED":"0"} to kernel.json file
# Use `jupyter kernelspec list` to see a list of installed kernels
# For more info see:
# https://stackoverflow.com/questions/58067359/is-there-a-way-to-set-pythonhashseed-for-a-jupyter-notebook-session

import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, MaxPooling1D
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json


from constants import experiment_set, protein_list
from utils import get_class, load_data_sources, reset_seeds

# Set random seeds for reproducibility.

reset_seeds()

# When running on a GPU, some operations have non-deterministic outputs.
# Force the code to run on a single core of the CPU.
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def train_model(protein, training_set, sources):
    print(f"Training model for {protein} from set {training_set}")
    _, X = load_data_sources(protein, "train", training_set, *sources)
    y = get_class(protein, "train", training_set)
    size = X.shape[2]
    # ideep receptive = 9
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
    model.add(MaxPooling1D(pool_size=2, strides=1, padding="valid"))
    model.add(Conv1D(10, 3, activation="relu", strides=1, padding="valid"))
    # Removed layers
    # -------------------------------------------------------------------
    # model.add(Dropout(0.1))
    # model.add(MaxPooling1D(pool_size=40, strides=1, padding='valid'))
    # model.add(Conv1D(15, 4, activation='relu'))
    # model.add(MaxPooling1D(pool_size=30, strides=1, padding='valid'))
    # model.add(Conv1D(15, 3, activation='relu'))
    # -------------------------------------------------------------------
    model.add(GlobalAveragePooling1D())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    checkpointer = ModelCheckpoint(
        filepath=f"shallow_model/set_{training_set}/{protein}/weights.h5",
        verbose=1,
        save_best_only=True,
    )
    earlystopper = EarlyStopping(monitor="val_loss", patience=5, verbose=1)
    metrics = model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=12,
        batch_size=200,
        verbose=1,
        callbacks=[earlystopper, checkpointer],
    )

    # Save model and weights to .json file.
    json_model = model.to_json()
    with open(
        "shallow_model/set_%s/%s/model.json" % (training_set, protein), "w"
    ) as fn:
        fn.write(json_model)

    # Save the model metrics generated with model fit.
    with open(
        "shallow_model/set_%s/%s/metrics" % (training_set, protein), "wb"
    ) as pickle_file:
        pickle.dump(metrics, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def run_predictions(protein, training_set, sources):
    with open(
        "shallow_model/set_%s/%s/model.json" % (training_set, protein), "r"
    ) as json_file:
        json = json_file.read()
        loaded_model = model_from_json(json)
        loaded_model.load_weights(
            "shallow_model/set_%s/%s/weights.h5" % (training_set, protein)
        )

    # Load data for testing purposes.
    _, X_test = load_data_sources(protein, "test", training_set, *sources)
    y_test = get_class(protein, "test", training_set)

    # Run predictions on test dataset and save them.
    predictions = loaded_model.predict(X_test)
    y_scores = predictions[:, 0:1]
    y_test = y_test[:, 0:1]

    score = roc_auc_score(y_test, y_scores)

    with open(
        "shallow_model/set_%s/%s/predictions" % (training_set, protein), "wb"
    ) as predictions_file:
        np.save(predictions_file, predictions)

    print(f"Model for {protein}({training_set}) achieved AUC {score}.")

    return score


if __name__ == "__main__":

    scores = {}
    for training_set in range(3):
        if not os.path.exists("shallow_model/set_%s" % training_set):
            os.makedirs("shallow_model/set_%s" % training_set)
        for protein in protein_list:
            if not os.path.exists("shallow_model/set_%s/%s" % (training_set, protein)):
                os.makedirs("shallow_model/set_%s/%s" % (training_set, protein))
                print(f"Created new directory results/set_{training_set}/{protein}.")

            train_model(protein, training_set, experiment_set)
            score = run_predictions(protein, training_set, experiment_set)

            if protein in scores:
                scores[protein].append(score)
            else:
                scores[protein] = [score]
    pd.DataFrame.from_dict(scores, orient="index").to_csv(
        "shallow_model/scores.tsv", sep="\t"
    )
